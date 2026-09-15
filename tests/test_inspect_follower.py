"""Hardware-free tests for the diagnostic command's write boundary and telemetry."""

from unittest.mock import Mock

import pytest
from lerobot.scripts.inspect_follower import (
    FIELDS,
    observe,
    read_field,
    readonly_writer,
)


def packet(op, motor_id=1):
    return [255, 255, 253, 0, motor_id, 3, 0, op, 0, 0]


@pytest.mark.parametrize("op", [1, 2])
def test_only_ping_and_read_are_sent(op):
    send = Mock(return_value=10)
    assert readonly_writer(send)(packet(op)) == 10
    send.assert_called_once()


@pytest.mark.parametrize("op", [3, 4, 5, 6, 8, 0x10, 0x82, 0x83, 0x92, 0x93])
def test_mutating_and_broadcast_instructions_never_reach_port(op):
    send = Mock()
    with pytest.raises(PermissionError):
        readonly_writer(send)(packet(op))
    send.assert_not_called()


@pytest.mark.parametrize("motor_id", [0, 7, 252, 254])
def test_other_motor_ids_are_blocked(motor_id):
    send = Mock()
    with pytest.raises(PermissionError):
        readonly_writer(send)(packet(1, motor_id))
    send.assert_not_called()


@pytest.mark.parametrize("data", [[], [1], [0] * 10])
def test_malformed_packet_is_blocked(data):
    send = Mock()
    with pytest.raises(PermissionError):
        readonly_writer(send)(data)
    send.assert_not_called()


@pytest.mark.parametrize(
    "name,width,raw,expected",
    [
        ("present_position", 4, 0xFFFFFFFF, -1),
        ("present_velocity", 4, 0xFFFFFFFE, -2),
        ("present_pwm", 2, 0xFFFF, -1),
        ("bus_watchdog", 1, 255, -1),
    ],
)
def test_signed_raw_feedback(name, width, raw, expected):
    handler = Mock()
    getattr(handler, f"read{width}ByteTxRx").return_value = (raw, 0, 0)
    r = read_field(handler, None, 1, FIELDS[name])
    assert r["raw"] == expected and r["raw_unsigned"] == raw
    assert r["read_start_monotonic"] <= r["read_end_monotonic"]


@pytest.mark.parametrize("comm,error", [(-3001, 0), (0, 2), (0, 0x82)])
def test_failed_reads_do_not_fabricate_zero_or_stale_feedback(comm, error):
    handler = Mock()
    handler.read4ByteTxRx.return_value = (12345, comm, error)
    r = read_field(handler, None, 1, FIELDS["present_position"])
    assert not r["available"]
    assert r["value"] is r["raw"] is r["raw_unsigned"] is None


def test_hardware_alert_preserves_diagnostic_value_and_alert():
    handler = Mock()
    handler.read1ByteTxRx.return_value = (1, 0, 128)
    r = read_field(handler, None, 1, FIELDS["hardware_error"])
    assert r["available"] and r["hardware_alert"] and r["value"] == 1


class FakeHandler:
    def __init__(self):
        self.reads = []

    def ping(self, port, motor_id):
        return {1: (1060, 0, 128), 2: (9999, 0, 0), 3: (1200, 0, 0)}.get(
            motor_id, (0, -3001, 0)
        )

    def read(self, port, motor_id, address):
        self.reads.append((motor_id, address))
        return (0xFFFF if address == 126 else 0, 0, 0)

    read1ByteTxRx = read2ByteTxRx = read4ByteTxRx = read


def test_model_specific_units_unknown_model_and_missing_motors():
    h = FakeHandler()
    result = observe(h, None)
    xl430, unknown, xl330, *missing = result["motors"]
    assert xl430["fields"]["present_load"]["value"] == -0.1
    assert not xl430["fields"]["present_current"]["available"]
    assert not xl430["fields"]["current_limit"]["available"]
    assert (1, 38) not in h.reads
    assert xl330["fields"]["present_current"]["value"] == -1
    assert "mA" in xl330["fields"]["present_current"]["unit"]
    assert (3, 38) in h.reads
    assert not xl330["fields"]["joint_torque"]["available"]
    assert unknown["fields"] == {}
    assert not any(mid == 2 for mid, address in h.reads)
    assert all(not m["responding"] and m["fields"] == {} for m in missing)
    assert result["register_writes"] == 0 and not result["motion_supported"]


def test_camera_capture_uses_lerobot_and_releases_device(monkeypatch, tmp_path):
    import numpy as np
    from lerobot.common.robot_devices.cameras import opencv
    from lerobot.scripts.inspect_follower import capture_frame

    camera = Mock(is_connected=True)
    camera.read.side_effect = [
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.ones((2, 3, 3), dtype=np.uint8),
    ]
    factory = Mock(return_value=camera)
    monkeypatch.setattr(opencv, "OpenCVCamera", factory)
    output = tmp_path / "frame.png"
    result = capture_frame(0, output)
    assert output.is_file() and result["frames_differ"]
    assert result["extrinsics"] is None and len(result["sha256"]) == 64
    camera.disconnect.assert_called_once()


def test_camera_capture_releases_after_failed_read(monkeypatch, tmp_path):
    from lerobot.common.robot_devices.cameras import opencv
    from lerobot.scripts.inspect_follower import capture_frame

    camera = Mock(is_connected=True)
    camera.read.side_effect = OSError("camera disconnected")
    monkeypatch.setattr(opencv, "OpenCVCamera", Mock(return_value=camera))
    with pytest.raises(OSError):
        capture_frame(0, tmp_path / "frame.png")
    camera.disconnect.assert_called_once()
