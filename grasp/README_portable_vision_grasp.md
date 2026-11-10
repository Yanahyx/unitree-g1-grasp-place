# G1 Portable Vision Grasp

`g1_portable_vision_grasp.py` is a minimal entry script for RGB-D object detection and right-hand grasping on the current G1 setup.

It captures one aligned RealSense RGB-D frame from the robot, runs YOLO-World open-vocabulary detection, estimates the object position in the pelvis frame, applies a fixed grasp offset, then calls the tested right-arm + RH56E2 hand grasp routine.

## Run

From the repo root:

```bash
cd /home/jpf/unitree_sdk2_python

python3 example/g1/grasp/g1_portable_vision_grasp.py \
  --camera-pitch 0.35 \
  --grasp-offset -0.22 0.04 0.25 \
  --object bottle \
  --announce-text "抓取完成"
```

For a cup:

```bash
python3 example/g1/grasp/g1_portable_vision_grasp.py \
  --camera-pitch 0.35 \
  --grasp-offset -0.22 0.04 0.25 \
  --object cup \
  --announce-text "杯子抓取完成"
```

## Arguments

Only three command-line arguments are exposed:

```text
--camera-pitch FLOAT
```

Approximate camera pitch in radians. Current tested value: `0.35`.

```text
--grasp-offset X Y Z
```

Offset added after converting the detected point into the pelvis frame. Current tested value:

```text
-0.22 0.04 0.25
```

Meaning:

```text
x: -0.22 m  move target back toward the robot
y: +0.04 m  move target toward robot left
z: +0.25 m  move target upward
```

```text
--object NAME
```

Open-vocabulary object name for YOLO-World, for example `bottle` or `cup`.

```text
--announce-text TEXT
```

TTS text spoken by the robot after the grasp routine completes successfully.
Default: `抓取完成`.

```text
--no-announce
```

Disable the success TTS announcement.

## Fixed Defaults

The script keeps the rest of the tested configuration fixed in code:

```text
DDS interface: enx68da73a54ce1
Robot SSH: unitree@192.168.123.164
SSH password: set ROBOT_SSH_PASSWORD locally, or use the --password option
RealSense serial: 243322071448
YOLO model: yolov8s-world.pt
Hand IP: 192.168.124.210
Hand port: 6000
Wrist pitch offset: -0.5
TTS speaker id: 0
```

## Files Needed

For this repo environment, keep:

```text
example/g1/grasp/g1_portable_vision_grasp.py
yolov8s-world.pt
```

The script still calls the existing tested grasp implementation:

```text
example/g1/grasp/g1_right_arm_body_ik_then_hand.py
```

So a migration target also needs the Unitree SDK Python environment, the G1 IK/grasp support files, and the RH56E2 hand helper files available in the same repo layout.

## Output

The script does not save RGB or depth images. It prints a JSON summary including:

```text
detection
point_camera_m
point_pelvis_before_offset_m
grasp_offset_m
grasp_target_m
workspace_ok
```

Then it executes the grasp command.

## Dependencies

Local computer:

```text
python3
numpy
pexpect
ultralytics
unitree_sdk2_python dependencies
```

Robot side:

```text
python3
numpy
pyrealsense2
RealSense connected and visible
SSH reachable at unitree@192.168.123.164
```
