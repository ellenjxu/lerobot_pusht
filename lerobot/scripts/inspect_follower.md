# Read-only Koch arm inspection

This is the discovery/observation stage of a follower control tool. It reuses this
checkout's LeRobot motor-bus connection and camera classes. The optional
`--role leader` mode supports explicitly requested leader observations with the
same read-only transport boundary. It does not instantiate
`ManipulatorRobot`, calibrate, home, enable torque, change motor settings, or offer
motion/hold/disarm commands. Keep other robot-control programs stopped during
inspection. The process lock coordinates instances of this tool; it cannot exclude
unrelated software that ignores the lock.

## Run

From the repository root, with its dependencies installed:

```sh
PYTHONPATH=. python -m lerobot.scripts.inspect_follower --list
PYTHONPATH=. python -m lerobot.scripts.inspect_follower \
  --port /dev/cu.usbmodemYOUR_DEVICE --serial YOUR_USB_SERIAL \
  --output /absolute/path/state.json
PYTHONPATH=. python -m lerobot.scripts.inspect_follower \
  --camera-index 0 --frame-output /absolute/path/frame.png \
  --output /absolute/path/frame.json
PYTHONPATH=. python -m pytest -q tests/test_inspect_follower.py
```

Use `--role leader` only after the operator has identified and authorized reading
that USB-connected arm. It expects six XL330-M077 (model 1190) servos; their current,
voltage and position units match the XL330-M288 column below.

Use a new output filename for each invocation. The tool refuses to overwrite
existing evidence. Identify the follower's USB serial with the local operator
before opening it. USB identity does not prove joint identity: reported joint names
are the expected ID mapping from `lerobot/configs/robot/koch.yaml`.

The serial command uses only protocol 2.0 at the repository's 1,000,000 baud and
unicast IDs 1–6. It changes the host UART baud, never the motors' baud registers.
A final transport guard rejects every instruction except unicast PING and READ to
those IDs. Discovery does not sweep addresses or configure absent motors. It
returns partial diagnostic data even when hardware is faulted. `motion_supported`
is always false; this output is not a motion-readiness certificate.

For camera capture, macOS camera permission must be granted. Place the follower,
base mounting and full proposed movement region in view. The returned timestamps
measure host reads, not exposure time. Different successive frames provide a basic
liveness check, not proof of bounded sensor latency. No camera-to-robot calibration,
mirroring convention or coordinate transform is asserted. Existing LeRobot camera
connection may enumerate cameras when the selected index fails.

## Feedback semantics

Model numbers are checked before reading model-specific registers. Unknown models
receive identity-only results. Register reads with communications errors or protocol
errors return null values; hardware alert bit 0x80 is retained separately so fault
registers remain readable.

| Field | XL430-W250 (1060) | XL330-M288 (1200) |
| --- | --- | --- |
| Present position | Signed ticks, uncalibrated | Signed ticks, uncalibrated |
| Present velocity | Signed raw × 0.229 rpm | Signed raw × 0.229 rpm |
| Address 126 | Signed load estimate × 0.1% | Signed input-supply current × 1 mA |
| Current limit | Unavailable | Register 38, mA |
| PWM limit | Register 36, output duty limit | Register 36, output duty limit |
| Voltage / temperature | 0.1 V / 1 °C | 0.1 V / 1 °C |
| Torque enabled / hardware errors | State bit / error bitmask | State bit / error bitmask |
| Joint torque / torque limit in Nm | Unavailable | Unavailable |

The raw bit pattern and signed interpretation are both retained. XL430 load is
inferred from motor output, not measured torque. XL330 measures supply current,
which is not a joint torque sensor or direct phase-current measurement. No Nm
conversion is made. A torque-enable flag is never an effort measurement.

Sources: [XL430 control table](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/)
[XL330-M288 control table](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/),
and [XL330-M077 control table](https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/).

Important startup details for the eventual controller: zero profile velocity or
acceleration can mean unlimited profiles; EEPROM min/max position limits are not
joint safety limits and are ignored in extended-position mode. A software read
watchdog alone cannot protect a disconnected bus. These diagnostics neither set nor
test a hardware watchdog.

## Control integration boundary

Inspected sources:

- This repo: `bbe3588fd28feb49097eb788f815cd04366ef9ac`.
- `robocurve/inspect-robots`: `7e4d1b7aee1c0d3cfc3a05a7492b9d12cda666f9`.
- `robocurve/inspect-robots-so101`: `096d89e12a9e31228fa57a7e5b7d236b09f3db6e`.

Inspect Robots' `Embodiment.step()` owns physical pacing. Its SO-ARM adapter targets
SO-100/SO-101 and the modern LeRobot SOFollower API, not this Koch/Dynamixel checkout.
Use a thin Koch edge adapter around this repository's bus/calibration support for
future bounded control. Do not copy its robot framework or use an SO controller
against these servos.

Before extending this command to motion, establish all six models and joint mapping,
normal power/health, the correct existing calibration, calibrated joint limits,
stable mounting, visible clear workspace and locally reachable power disconnect.
Then implement and mock-test limits on position/velocity/acceleration/displacement/
duration, one command owner, current-position initialization before torque enable,
freshness and communication-loss watchdogs, and support-aware hold/disable behavior.
Only after those checks should a concrete small movement be presented for local
readiness confirmation. No first-motion bounds are approved by this document.
