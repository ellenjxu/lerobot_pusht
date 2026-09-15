"""Read-only Koch arm discovery and telemetry. This entry point cannot move motors."""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import io
import json
import os
import time
from pathlib import Path

# ID-to-joint order is the repository's Koch configuration, not inferred from USB.
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
EXPECTED_MODELS = {
    "follower": (1060, 1060, 1200, 1200, 1200, 1200),
    "leader": (1190,) * 6,
}
MODELS = {1060: "xl430-w250", 1190: "xl330-m077", 1200: "xl330-m288"}
# address, width, signed, scale, unit. Profile units depend on Drive_Mode.
FIELDS = {
    "firmware": (6, 1, False, 1, "raw"),
    "id": (7, 1, False, 1, "raw"),
    "baud_index": (8, 1, False, 1, "raw"),
    "drive_mode": (10, 1, False, 1, "raw"),
    "operating_mode": (11, 1, False, 1, "raw"),
    "homing_offset": (20, 4, True, 1, "tick"),
    "temperature_limit": (31, 1, False, 1, "degC"),
    "max_voltage_limit": (32, 2, False, 0.1, "V"),
    "min_voltage_limit": (34, 2, False, 0.1, "V"),
    "pwm_limit": (36, 2, False, 0.113, "% PWM"),
    "velocity_limit": (44, 4, False, 0.229, "rpm"),
    "max_position_limit": (48, 4, False, 1, "tick"),
    "min_position_limit": (52, 4, False, 1, "tick"),
    "startup_configuration": (60, 1, False, 1, "raw"),
    "shutdown": (63, 1, False, 1, "bitmask"),
    "torque_enable": (64, 1, False, 1, "boolean"),
    "hardware_error": (70, 1, False, 1, "bitmask"),
    "bus_watchdog": (98, 1, True, 1, "raw; 20 ms per positive count"),
    "profile_acceleration": (108, 4, False, 1, "raw; depends on drive mode"),
    "profile_velocity": (112, 4, False, 1, "raw; depends on drive mode"),
    "goal_position": (116, 4, True, 1, "tick"),
    "realtime_tick": (120, 2, False, 1, "ms modulo 32768"),
    "present_pwm": (124, 2, True, 0.113, "% PWM"),
    "present_velocity": (128, 4, True, 0.229, "rpm"),
    "present_position": (132, 4, True, 1, "tick; uncalibrated"),
    "input_voltage": (144, 2, False, 0.1, "V"),
    "temperature": (146, 1, False, 1, "degC"),
}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def readonly_writer(write):
    """Reject every motor instruction except unicast PING/READ to configured IDs."""

    def guarded(packet):
        if (
            len(packet) < 10
            or bytes(packet[:4]) != b"\xff\xff\xfd\x00"
            or packet[4] not in range(1, 7)
            or packet[7] not in (1, 2)
        ):
            raise PermissionError(
                "Follower diagnostics permit only unicast PING/READ to IDs 1..6"
            )
        return write(packet)

    return guarded


def read_field(handler, port, motor_id, spec):
    address, width, signed, scale, unit = spec
    start = time.monotonic()
    raw, comm, error = getattr(handler, f"read{width}ByteTxRx")(port, motor_id, address)
    end = time.monotonic()
    valid = comm == 0 and error & 0x7F == 0
    value = raw
    if signed and value & (1 << (8 * width - 1)):
        value -= 1 << (8 * width)
    return {
        "raw_unsigned": raw if valid else None,
        "raw": value if valid else None,
        "value": value * scale if valid else None,
        "unit": unit,
        "available": valid,
        "comm": comm,
        "status_error": error,
        "hardware_alert": bool(error & 0x80),
        "utc": utc_now(),
        "read_start_monotonic": start,
        "read_end_monotonic": end,
    }


def observe(handler, port, role="follower"):
    """Retain partial observations and faults. Missing data is never zero-filled."""
    expected = EXPECTED_MODELS[role]
    start = time.monotonic()
    motors = []
    for motor_id, joint in enumerate(JOINTS, 1):
        model, comm, error = handler.ping(port, motor_id)
        record = {
            "id": motor_id,
            "configured_joint": joint,
            "comm": comm,
            "status_error": error,
            "model_number": model if comm == 0 else None,
            "responding": comm == 0,
            "fields": {},
        }
        motors.append(record)
        if comm != 0 or error & 0x7F:
            continue
        record["model"] = MODELS.get(model)
        record["matches_koch_config"] = model == expected[motor_id - 1]
        if model not in MODELS:
            continue  # Do not apply an unverified control table to an unknown model.
        fields = dict(FIELDS)
        if model == 1060:
            fields["present_load"] = (126, 2, True, 0.1, "% estimated load")
            unavailable = ("present_current", "current_limit")
        else:
            fields["present_current"] = (126, 2, True, 1, "mA (input supply current)")
            fields["current_limit"] = (38, 2, False, 1, "mA")
            unavailable = ("present_load",)
        record["fields"] = {
            key: read_field(handler, port, motor_id, spec)
            for key, spec in fields.items()
        }
        for key in (*unavailable, "joint_torque", "torque_limit_nm"):
            record["fields"][key] = {
                "available": False,
                "value": None,
                "reason": "No documented measurement/conversion for this model",
            }
    return {
        "utc": utc_now(),
        "start_monotonic": start,
        "end_monotonic": time.monotonic(),
        "protocol": 2.0,
        "baud": 1_000_000,
        "register_writes": 0,
        "declared_role": role,
        "motion_supported": False,
        "calibration_applied": False,
        "motors": motors,
    }


def capture_frame(index, destination):
    """Capture through LeRobot; host timestamps do not claim exposure timing or extrinsics."""
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
    from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
    from PIL import Image

    camera = OpenCVCamera(OpenCVCameraConfig(camera_index=index))
    try:
        camera.connect()
        first = camera.read()
        time.sleep(0.1)
        start = time.monotonic()
        frame = camera.read()
        end = time.monotonic()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as image_file:
            Image.fromarray(frame).save(image_file, format="PNG")
        return {
            "utc": utc_now(),
            "camera_index": index,
            "shape": list(frame.shape),
            "read_start_monotonic": start,
            "read_end_monotonic": end,
            "timestamp_kind": "host read timing; exposure timestamp unavailable",
            "frames_differ": bool((first != frame).any()),
            "rotation": None,
            "extrinsics": None,
            "frame": str(destination),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    finally:
        if camera.is_connected:
            camera.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List serial devices without opening them"
    )
    parser.add_argument("--port")
    parser.add_argument("--role", choices=("follower", "leader"), default="follower")
    parser.add_argument(
        "--serial", help="Required USB serial identity for the confirmed follower"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--frame-output", type=Path)
    args = parser.parse_args()
    if args.camera_index is not None:
        if args.frame_output is None or args.output is None:
            parser.error("Camera capture requires --frame-output and --output")
        if args.list or args.port or args.serial:
            parser.error("Capture and motor observation are separate calls")
        with contextlib.redirect_stdout(io.StringIO()):
            result = capture_frame(args.camera_index, args.frame_output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(result, output, indent=2)
        print(json.dumps(result))
        return
    from serial.tools.list_ports import comports

    ports = list(comports())
    if args.list:
        print(
            json.dumps(
                [
                    {
                        "port": p.device,
                        "serial": p.serial_number,
                        "vid": p.vid,
                        "pid": p.pid,
                        "description": p.description,
                    }
                    for p in ports
                ]
            )
        )
        return
    if not args.port or not args.serial or not args.output:
        parser.error("--port, --serial, and --output are required for observation")
    matches = [
        p for p in ports if p.device == args.port and p.serial_number == args.serial
    ]
    if len(matches) != 1:
        parser.error("Port does not match the confirmed USB serial identity")
    from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
    from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

    motors = {
        name: (i, MODELS[model])
        for i, (name, model) in enumerate(zip(JOINTS, EXPECTED_MODELS[args.role]), 1)
    }
    bus = DynamixelMotorsBus(DynamixelMotorsBusConfig(port=args.port, motors=motors))
    # Same lock for cu/tty aliases and independent checkouts; scoped to the USB identity.
    lock_dir = Path.home() / ".cache" / "lerobot" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = hashlib.sha256(args.serial.encode()).hexdigest()
    fd = os.open(lock_dir / f"{key}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            bus.connect()  # Inspected: opens the serial port, no motor configuration/torque writes.
            bus.port_handler.writePort = readonly_writer(bus.port_handler.writePort)
            with contextlib.redirect_stdout(io.StringIO()):
                bus.set_bus_baudrate(1_000_000)  # Changes host UART only.
            result = observe(bus.packet_handler, bus.port_handler, args.role)
            result.update(port=args.port, usb_serial=args.serial)
        finally:
            if bus.is_connected:
                bus.disconnect()  # Closes the port, does not change torque.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create preserves prior evidence rather than silently overwriting it.
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
