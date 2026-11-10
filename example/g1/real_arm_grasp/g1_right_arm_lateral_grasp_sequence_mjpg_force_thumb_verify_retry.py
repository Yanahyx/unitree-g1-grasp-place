#!/usr/bin/env python3
"""Lateral grasp sequence with contact feedback, verification, and one retry.

Sequence:
1. Close RH56E2 hand before lifting.
2. Move right hand to PREGRASP_TARGET in pelvis frame.
3. Open RH56E2 hand and rotate the thumb outward.
4. Move the whole right hand to the final grasp target.
5. On force/current feedback, rotate the thumb inward, then close the fingers.
6. Verify finger obstruction, lift briefly, and confirm the grasp is retained.
7. On failure, lower safely, redetect, and retry at most once.
8. Move back to PREGRASP_TARGET.
9. Move laterally to a safer side-return pose.
10. Return the right arm to its start posture and release arm_sdk.

This experimental file is copied from
g1_right_arm_lateral_grasp_sequence_mjpg_force_thumb.py. The source file is
left untouched.
"""

import argparse
import os
import io
import json
import select
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
G1_DIR = THIS_DIR.parents[0]
GRASP_DIR = G1_DIR / "grasp"
HANDS_DIR = G1_DIR / "hands"
HIGH_LEVEL_DIR = G1_DIR / "high_level"
for path in (THIS_DIR, GRASP_DIR, HANDS_DIR, HIGH_LEVEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from g1_right_arm_body_ik_example import (  # noqa: E402
    G1JointIndex,
    format_joint_delta_table,
    format_joint_table,
    model_to_motor_q,
    motor_to_model_q,
    right_arm_joints,
    right_hand_pose,
    solve_position_ik,
    solve_position_ik_for,
    validate_target,
)
from g1_portable_vision_grasp import (  # noqa: E402
    camera_to_pelvis,
    depth_point_for_detection,
    detect_object,
    load_model,
    run_capture,
)
from g1_right_arm_official_style_reach import (  # noqa: E402
    LEFT_ARM_7DOF,
    WAIST_JOINTS,
    OfficialStyleArmSdkController,
    detect_interface,
    hold_group,
    interpolate_positions,
    joint_positions,
    joint_vector,
    joint_velocity_abs_max,
    release_group,
    run_timed_phase,
)
from rh56e2_hand_only import write_hand_pose  # noqa: E402
from rh56e2_modbus_move_once import write_registers  # noqa: E402
from rh56e2_modbus_probe import ModbusTcpClient, REG, fmt  # noqa: E402
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient  # noqa: E402


MJPG_URL = "http://192.168.1.103:18082/color.mjpg"
DEPTH_SERIAL = "243322071448"
PREGRASP_TARGET = np.asarray([0.30, -0.38, 0.20], dtype=float)
FIXED_GRASP_Z = 0.12
DEFAULT_VISION_XY_OFFSET = [-0.12, 0.0]
DEFAULT_SIDE_STANDOFF_Y = 0.08
DEFAULT_HAND_OPEN_ANGLE = [1000, 1000, 1000, 1000, 1000, 1000]
DEFAULT_HAND_GRASP_ANGLE = [0, 0, 0, 0, 400, 300]
DEFAULT_HAND_SAFE_CLOSE_ANGLE = [0, 0, 0, 0, 400, 1000]
DEFAULT_WRIST_ROLL = 0.30
DEFAULT_WRIST_PITCH = -0.80
DEFAULT_WRIST_YAW = 0.50
DEFAULT_RETURN_SIDE_OFFSET = 0.03
CURRENT_ADDR = 1594
HAND_CHANNEL_LABELS = [
    "A0/finger0",
    "A1/finger1",
    "A2/finger2",
    "A3/finger3",
    "A4/thumb_bend",
    "A5/thumb_rotation",
]
WRIST_LIMITS = {
    G1JointIndex.RightWristRoll: (-1.9722, 1.9722),
    G1JointIndex.RightWristPitch: (-1.6144, 1.6144),
    G1JointIndex.RightWristYaw: (-1.6144, 1.6144),
}
WRIST_NAMES = {
    G1JointIndex.RightWristRoll: "RightWristRoll",
    G1JointIndex.RightWristPitch: "RightWristPitch",
    G1JointIndex.RightWristYaw: "RightWristYaw",
}
HAND_NETWORK_HINT = """Hand TCP preflight failed.

The RH56E2 hand is expected at 192.168.124.210:6000. If the robot was rebooted,
add the temporary secondary address on eth1, then probe the hand:

  sudo ip addr add 192.168.124.164/24 dev eth1

  cd /home/unitree/unitree_sdk2_python
  python3 example/g1/high_level/rh56e2_modbus_probe.py \\
    --ip 192.168.124.210 \\
    --port 6000

Expected probe output includes: TCP connected and ID: 1.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-point lateral right-arm grasp sequence.")
    parser.add_argument("network_interface", nargs="?", default=None, help="DDS interface, e.g. eth1.")
    parser.add_argument(
        "--pregrasp-target",
        nargs=3,
        type=float,
        default=PREGRASP_TARGET.tolist(),
        metavar=("X", "Y", "Z"),
        help="Saved first point in pelvis frame. Default: 0.30 -0.38 0.20.",
    )
    parser.add_argument("--left-distance", type=float, default=0.10, help="Pelvis-frame +Y lateral move before grasp.")
    parser.add_argument("--use-vision", action="store_true", help="Use RGB-D vision to override final grasp XY.")
    parser.add_argument("--object", default="bottle", help="Open-vocabulary object name for YOLO-World.")
    parser.add_argument("--camera-pitch", type=float, default=0.8)
    parser.add_argument("--fixed-grasp-z", type=float, default=FIXED_GRASP_Z)
    parser.add_argument("--vision-xy-offset", nargs=2, type=float, default=DEFAULT_VISION_XY_OFFSET, metavar=("X", "Y"))
    parser.add_argument(
        "--side-standoff-y",
        type=float,
        default=DEFAULT_SIDE_STANDOFF_Y,
        help="Keep the lateral grasp target this many meters on the approach side of the detected object.",
    )
    parser.add_argument("--vision-host", default="unitree@192.168.1.103")
    parser.add_argument("--vision-password", default=os.environ.get("ROBOT_SSH_PASSWORD", ""))
    parser.add_argument(
        "--vision-source",
        choices=("mjpg", "realsense"),
        default="mjpg",
        help="Vision source used when --use-vision is set. Default: mjpg.",
    )
    parser.add_argument("--mjpg-url", default=MJPG_URL, help="HTTP MJPEG RGB stream URL.")
    parser.add_argument(
        "--mjpg-assumed-depth",
        type=float,
        default=0.57,
        help="Fallback assumed object depth in meters if --mjpg-depth-source assumed is used.",
    )
    parser.add_argument(
        "--mjpg-depth-source",
        choices=("http-rgbd", "ros", "realsense", "assumed"),
        default="http-rgbd",
        help="Depth source paired with MJPEG RGB. Default: second RealSense HTTP rgbd.npz.",
    )
    parser.add_argument("--rgbd-npz-url", default="http://192.168.1.103:18082/rgbd.npz")
    parser.add_argument("--ros-depth-topic", default="/d455/aligned_depth_to_color/image_raw")
    parser.add_argument("--ros-camera-info-topic", default="/d455/color/camera_info")
    parser.add_argument("--ros-timeout", type=float, default=3.0)
    parser.add_argument("--depth-serial", default=DEPTH_SERIAL, help="RealSense serial for depth-only capture.")
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument("--depth-warmup-frames", type=int, default=15)
    parser.add_argument("--mjpg-fx", type=float, default=612.0, help="RGB camera fx at reference size.")
    parser.add_argument("--mjpg-fy", type=float, default=612.0, help="RGB camera fy at reference size.")
    parser.add_argument("--mjpg-cx", type=float, default=320.0, help="RGB camera cx at reference size.")
    parser.add_argument("--mjpg-cy", type=float, default=240.0, help="RGB camera cy at reference size.")
    parser.add_argument("--mjpg-reference-width", type=float, default=640.0)
    parser.add_argument("--mjpg-reference-height", type=float, default=480.0)
    parser.add_argument(
        "--detector-server-url",
        default=None,
        help="Optional persistent detector endpoint, e.g. http://127.0.0.1:8765/detect.",
    )
    parser.add_argument("--detector-timeout", type=float, default=30.0)
    parser.add_argument("--detector-retries", type=int, default=3)
    parser.add_argument("--detector-retry-delay", type=float, default=0.5)
    parser.add_argument(
        "--vision-fallback-source",
        choices=("none", "realsense"),
        default="realsense",
        help="Fallback vision source when detector/RGB-D HTTP URL is unavailable. Default: realsense.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually move the arm and hand.")
    parser.add_argument("--yes", action="store_true", help="Confirm the robot and hand have clear space.")
    parser.add_argument(
        "--approach-only",
        action="store_true",
        help="Move to the planned safe approach target and return without closing the hand.",
    )
    parser.add_argument("--weight", type=float, default=0.75)
    parser.add_argument("--allow-full-weight", action="store_true")
    parser.add_argument("--pre-duration", type=float, default=6.0)
    parser.add_argument("--lateral-duration", type=float, default=3.0)
    parser.add_argument("--return-duration", type=float, default=5.0)
    parser.add_argument(
        "--return-side-offset",
        type=float,
        default=DEFAULT_RETURN_SIDE_OFFSET,
        help="Additional pelvis-frame lateral offset used to move farther outside the table before returning to the start posture.",
    )
    parser.add_argument("--return-side-duration", type=float, default=2.0)
    parser.add_argument("--hold-before-open", type=float, default=0.5)
    parser.add_argument("--hold-after-open", type=float, default=0.5)
    parser.add_argument("--hold-before-grasp", type=float, default=0.5)
    parser.add_argument("--hold-after-grasp", type=float, default=0.8)
    parser.add_argument(
        "--wrist-roll-target",
        type=float,
        default=DEFAULT_WRIST_ROLL,
        help="RightWristRoll target used for the grasp wrist pose.",
    )
    parser.add_argument(
        "--wrist-pitch-target",
        type=float,
        default=DEFAULT_WRIST_PITCH,
        help="RightWristPitch target used for the grasp wrist pose.",
    )
    parser.add_argument(
        "--wrist-yaw-target",
        type=float,
        default=DEFAULT_WRIST_YAW,
        help="RightWristYaw target used for the grasp wrist pose.",
    )
    parser.add_argument(
        "--no-vertical-wrist",
        action="store_true",
        help="Disable the explicit wrist roll/pitch/yaw override.",
    )
    parser.add_argument("--release-duration", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.025)
    parser.add_argument("--max-joint-delta", type=float, default=2.2)
    parser.add_argument("--max-joint-speed", type=float, default=0.35)
    parser.add_argument("--max-start-arm-dq", type=float, default=0.8)
    parser.add_argument("--max-start-waist-dq", type=float, default=0.4)
    parser.add_argument("--kp-right", type=float, default=60.0)
    parser.add_argument("--kd-right", type=float, default=1.5)
    parser.add_argument("--kp-left", type=float, default=60.0)
    parser.add_argument("--kd-left", type=float, default=1.5)
    parser.add_argument("--kp-waist", type=float, default=60.0)
    parser.add_argument("--kd-waist", type=float, default=1.5)
    parser.add_argument("--hand-ip", default="192.168.124.210")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--hand-unit-id", type=int, default=1)
    parser.add_argument("--hand-timeout", type=float, default=2.0)
    parser.add_argument("--hand-speed", type=int, default=200)
    parser.add_argument("--hand-force", type=int, default=2000)
    parser.add_argument("--hand-hold", type=float, default=3.0)
    parser.add_argument("--hand-open-angle", nargs=6, type=int, default=DEFAULT_HAND_OPEN_ANGLE)
    parser.add_argument("--hand-grasp-angle", nargs=6, type=int, default=DEFAULT_HAND_GRASP_ANGLE)
    parser.add_argument(
        "--hand-safe-close-angle",
        nargs=6,
        type=int,
        default=DEFAULT_HAND_SAFE_CLOSE_ANGLE,
        help="Closed non-grasp travel posture. Default keeps thumb at maximum rotation to avoid leg collision.",
    )
    parser.add_argument(
        "--no-force-feedback-grasp",
        action="store_true",
        help="Use the original one-shot hand grasp command instead of force/current feedback staging.",
    )
    parser.add_argument(
        "--thumb-rotate-open",
        type=int,
        default=None,
        help="Optional A5 override for thumb rotation during open. Default uses --hand-open-angle A5.",
    )
    parser.add_argument(
        "--thumb-rotate-close",
        type=int,
        default=None,
        help=(
            "Optional A5 override for thumb rotation during close. Default uses --hand-grasp-angle A5."
        ),
    )
    parser.add_argument(
        "--feedback-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for force/current feedback. Default 0 monitors for the whole approach.",
    )
    parser.add_argument("--feedback-poll", type=float, default=0.05, help="Seconds between hand feedback reads.")
    parser.add_argument(
        "--thumb-inward-delay",
        type=float,
        default=0.45,
        help="Seconds to wait after rotating the thumb inward before closing the fingers.",
    )
    parser.add_argument(
        "--thumb-bend-delay",
        type=float,
        default=0.20,
        help="Seconds to wait after closing A0-A3 before bending the thumb closed.",
    )
    parser.add_argument(
        "--thumb-outward-delay",
        type=float,
        default=0.8,
        help="Seconds to wait after rotating the thumb outward before opening the fingers.",
    )
    parser.add_argument(
        "--thumb-stage-channel",
        type=int,
        default=5,
        help="Hand angle channel used for staged thumb rotation. Default A5.",
    )
    parser.add_argument(
        "--thumb-rotate-speed",
        type=int,
        default=500,
        help="Speed used for staged thumb rotation.",
    )
    parser.add_argument(
        "--thumb-rotate-force",
        type=int,
        default=2000,
        help="Force used for staged thumb rotation.",
    )
    parser.add_argument(
        "--thumb-rotate-hold",
        type=float,
        default=0.03,
        help="Seconds to wait after each thumb rotation interpolation step.",
    )
    parser.add_argument(
        "--thumb-rotate-step",
        type=int,
        default=40,
        help="Maximum A5 change per thumb rotation interpolation step.",
    )
    parser.add_argument(
        "--thumb-close-tolerance",
        type=int,
        default=120,
        help="Allowed A5 error before closing fingers after staged thumb rotation.",
    )
    parser.add_argument(
        "--feedback-force-delta",
        type=int,
        default=60,
        help="Trigger when max abs signed forceAct delta reaches this value.",
    )
    parser.add_argument(
        "--feedback-current-delta",
        type=int,
        default=35,
        help="Trigger when max abs current delta reaches this value.",
    )
    parser.add_argument(
        "--feedback-min-phase",
        type=float,
        default=0.85,
        help="Ignore hand feedback before this fraction of the approach motion to avoid early false triggers.",
    )
    parser.add_argument(
        "--close-on-feedback-timeout",
        action="store_true",
        default=False,
        help="Close the hand even if no force/current feedback is detected before timeout.",
    )
    parser.add_argument(
        "--no-close-on-feedback-timeout",
        dest="close_on_feedback_timeout",
        action="store_false",
        help="Do not close the hand if no force/current feedback is detected.",
    )
    parser.add_argument(
        "--close-at-approach-end",
        action="store_true",
        default=False,
        help="If no hand feedback triggers, close at the final approach point using the safe thumb-first sequence.",
    )
    parser.add_argument(
        "--verify-grasp",
        action="store_true",
        help="Verify finger obstruction, lift briefly, and confirm the obstruction is retained.",
    )
    parser.add_argument(
        "--verify-lift-distance",
        type=float,
        default=0.04,
        help="Pelvis-frame +Z lift used for grasp verification. Default: 0.04 m.",
    )
    parser.add_argument("--verify-lift-duration", type=float, default=1.5)
    parser.add_argument("--verify-hold", type=float, default=0.7)
    parser.add_argument(
        "--verify-finger-gap",
        type=int,
        default=120,
        help="Minimum A0-A3 actual-minus-commanded angle treated as object obstruction.",
    )
    parser.add_argument(
        "--verify-min-blocked-fingers",
        type=int,
        default=2,
        help="Minimum obstructed A0-A3 channels required before and after the verification lift.",
    )
    parser.add_argument(
        "--verify-empty-max-gap",
        type=int,
        default=80,
        help="Maximum A0-A3 close-command gap required to declare the hand confidently empty for retry.",
    )
    parser.add_argument(
        "--verify-max-finger-drop",
        type=int,
        default=80,
        help="Maximum decrease of a blocked finger angle during lift before declaring a slip.",
    )
    parser.add_argument(
        "--max-grasp-retries",
        type=int,
        choices=(0, 1),
        default=1,
        help="Maximum retries after failed verification. This experimental version allows at most one.",
    )
    parser.add_argument(
        "--retry-approach-extra",
        type=float,
        default=0.015,
        help="Extra meters along the lateral approach direction after retry redetection.",
    )
    parser.add_argument(
        "--test-first-attempt-shortfall",
        type=float,
        default=0.0,
        help="TEST ONLY: stop the first approach this many meters short; retry uses the normal redetected target.",
    )
    parser.add_argument(
        "--clearance-lift-distance",
        type=float,
        default=0.12,
        help="Actual palm +Z clearance before moving laterally across the table edge. Default: 0.12 m.",
    )
    parser.add_argument("--clearance-lift-duration", type=float, default=2.5)
    parser.add_argument("--clearance-side-duration", type=float, default=3.0)
    parser.add_argument(
        "--clearance-settle-duration",
        type=float,
        default=2.0,
        help="Maximum hold time for each clearance waypoint to reach the tracking tolerance.",
    )
    parser.add_argument(
        "--clearance-joint-tolerance",
        type=float,
        default=0.18,
        help="Maximum joint tracking error after each clearance segment before aborting.",
    )
    parser.add_argument(
        "--motion-tracking-error-limit",
        type=float,
        default=0.30,
        help="Abort a tracked retreat segment if any joint persistently lags its in-flight command by this many radians.",
    )
    parser.add_argument(
        "--motion-tracking-error-duration",
        type=float,
        default=0.30,
        help="Seconds an in-flight tracking error must persist before entering safety hold.",
    )
    parser.add_argument(
        "--no-clearance-retreat",
        action="store_true",
        help="Disable the high vertical-then-sideways table-clearance retreat.",
    )
    parser.add_argument("--skip-hand", action="store_true", help="Run arm motion only; do not command RH56E2.")
    parser.add_argument("--announce-start-text", default="检测到物体，正在执行抓取。")
    parser.add_argument("--announce-text", default="抓取完成")
    parser.add_argument("--announce-speaker-id", type=int, default=0)
    parser.add_argument("--no-announce", action="store_true")
    return parser.parse_args()


def hand_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        ip=args.hand_ip,
        port=args.hand_port,
        unit_id=args.hand_unit_id,
        timeout=args.hand_timeout,
        speed=args.hand_speed,
        force=args.hand_force,
        hold=args.hand_hold,
        read_before=False,
        read_after=False,
    )


def signed16(values: Sequence[int]) -> np.ndarray:
    return np.asarray([value - 65536 if value >= 32768 else value for value in values], dtype=int)


def hand_channel_label(index: int) -> str:
    if 0 <= index < len(HAND_CHANNEL_LABELS):
        return HAND_CHANNEL_LABELS[index]
    return f"A{index}"


def hand_feedback_snapshot(client: ModbusTcpClient) -> Dict[str, np.ndarray]:
    force_raw = client.read_holding_registers(REG["forceAct"], 6)
    current_raw = client.read_holding_registers(CURRENT_ADDR, 6)
    angle_act = client.read_holding_registers(REG["angleAct"], 6)
    return {
        "force": signed16(force_raw),
        "current": signed16(current_raw),
        "angle_act": np.asarray(angle_act, dtype=int),
    }


def feedback_deltas(snapshot: Dict[str, np.ndarray], baseline: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.abs(snapshot["force"] - baseline["force"]),
        np.abs(snapshot["current"] - baseline["current"]),
    )


def max_delta_channel(delta: np.ndarray) -> Tuple[int, int]:
    index = int(np.argmax(delta))
    return index, int(delta[index])


def print_feedback_delta_detail(label: str, snapshot: Dict[str, np.ndarray], baseline: Dict[str, np.ndarray]) -> None:
    force_delta, current_delta = feedback_deltas(snapshot, baseline)
    force_index, force_value = max_delta_channel(force_delta)
    current_index, current_value = max_delta_channel(current_delta)
    print(f"{label}_force_delta[6]: {fmt(force_delta.tolist())}")
    print(
        f"{label}_force_delta_max_channel: "
        f"{hand_channel_label(force_index)} index={force_index} delta={force_value}"
    )
    print(f"{label}_current_delta[6]: {fmt(current_delta.tolist())}")
    print(
        f"{label}_current_delta_max_channel: "
        f"{hand_channel_label(current_index)} index={current_index} delta={current_value}"
    )


def validate_angle_values(values: Sequence[int]) -> None:
    if len(values) != 6:
        raise ValueError("expected six hand angle values")
    if any(value < -1 or value > 1000 for value in values):
        raise ValueError("hand angle values must be -1 or in 0..1000")


def with_thumb_rotation(values: Sequence[int], thumb_rotation: Optional[int]) -> list:
    updated = list(values)
    if thumb_rotation is None:
        validate_angle_values(updated)
        return updated
    if thumb_rotation < -1 or thumb_rotation > 1000:
        raise ValueError("thumb rotation must be -1 or in 0..1000")
    updated[5] = int(thumb_rotation)
    validate_angle_values(updated)
    return updated


def thumb_inward_pose(open_angles: Sequence[int], close_angles: Sequence[int], thumb_channel: int) -> list:
    """Keep fingers open while moving the configured thumb channel to its close/grasp angle."""
    validate_angle_values(open_angles)
    validate_angle_values(close_angles)
    if thumb_channel < 0 or thumb_channel > 5:
        raise ValueError("thumb channel must be in 0..5")
    updated = list(open_angles)
    updated[thumb_channel] = int(close_angles[thumb_channel])
    validate_angle_values(updated)
    return updated


def thumb_outward_pose(open_angles: Sequence[int], close_angles: Sequence[int], thumb_channel: int) -> list:
    """Keep fingers closed while moving the configured thumb channel to its open/outward angle."""
    validate_angle_values(open_angles)
    validate_angle_values(close_angles)
    if thumb_channel < 0 or thumb_channel > 5:
        raise ValueError("thumb channel must be in 0..5")
    updated = list(close_angles)
    updated[thumb_channel] = int(open_angles[thumb_channel])
    validate_angle_values(updated)
    return updated


def write_hand_angles_open_client(
    client: ModbusTcpClient,
    args: argparse.Namespace,
    angles: Sequence[int],
    label: str,
    speed_value: Optional[int] = None,
    force_value: Optional[int] = None,
) -> None:
    speed = [args.hand_speed if speed_value is None else speed_value] * 6
    force = [args.hand_force if force_value is None else force_value] * 6
    validate_angle_values(angles)
    print(f"hand_speedSet_{label}: {fmt(speed)}")
    write_registers(client, REG["speedSet"], speed)
    time.sleep(0.1)
    print(f"hand_forceSet_{label}: {fmt(force)}")
    write_registers(client, REG["forceSet"], force)
    time.sleep(0.1)
    print(f"hand_angleSet_{label}: {fmt(angles)}")
    write_registers(client, REG["angleSet"], list(angles))


def thumb_rotation_waypoints(start: int, target: int, step: int) -> list:
    if start == -1 or target == -1:
        return []
    start = int(start)
    target = int(target)
    step = max(1, int(step))
    if start == target:
        return [target]
    direction = 1 if target > start else -1
    values = list(range(start + direction * step, target + direction, direction * step))
    if not values or values[-1] != target:
        values.append(target)
    return [max(0, min(1000, value)) for value in values]


def write_staged_thumb_rotation_client(
    client: ModbusTcpClient,
    args: argparse.Namespace,
    base_angles: Sequence[int],
    start: int,
    target: int,
    label: str,
) -> None:
    validate_angle_values(base_angles)
    waypoints = thumb_rotation_waypoints(start, target, args.thumb_rotate_step)
    if not waypoints:
        return
    speed = [args.thumb_rotate_speed] * 6
    force = [args.thumb_rotate_force] * 6
    print(f"hand_speedSet_{label}: {fmt(speed)}")
    write_registers(client, REG["speedSet"], speed)
    time.sleep(0.1)
    print(f"hand_forceSet_{label}: {fmt(force)}")
    write_registers(client, REG["forceSet"], force)
    time.sleep(0.1)
    for value in waypoints:
        angles = list(base_angles)
        angles[args.thumb_stage_channel] = int(value)
        print(f"hand_angleSet_{label}_A{args.thumb_stage_channel}_{value}: {fmt(angles)}")
        write_registers(client, REG["angleSet"], angles)
        time.sleep(max(0.0, args.thumb_rotate_hold))


def read_hand_angle_act(client: ModbusTcpClient) -> list:
    return client.read_holding_registers(REG["angleAct"], 6)


def force_feedback_grasp(args: argparse.Namespace) -> None:
    open_angles = with_thumb_rotation(args.hand_open_angle, args.thumb_rotate_open)
    close_angles = with_thumb_rotation(args.hand_grasp_angle, args.thumb_rotate_close)
    thumb_outer_angles = thumb_outward_pose(open_angles, close_angles, args.thumb_stage_channel)
    thumb_inner_angles = thumb_inward_pose(open_angles, close_angles, args.thumb_stage_channel)
    print("force_feedback_grasp_thumb_outer_angles:", fmt(thumb_outer_angles))
    print("force_feedback_grasp_open_angles:", fmt(open_angles))
    print("force_feedback_grasp_thumb_inner_angles:", fmt(thumb_inner_angles))
    print("force_feedback_grasp_close_angles:", fmt(close_angles))

    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print(f"hand_connect: {args.hand_ip}:{args.hand_port}")
        print("TCP connected")
        write_staged_thumb_rotation_client(
            client,
            args,
            thumb_outer_angles,
            close_angles[args.thumb_stage_channel],
            open_angles[args.thumb_stage_channel],
            "thumb_outer",
        )
        time.sleep(max(0.0, args.thumb_outward_delay))
        write_hand_angles_open_client(client, args, open_angles, "open")
        time.sleep(max(0.0, args.hand_hold))

        baseline = hand_feedback_snapshot(client)
        print(f"feedback_baseline_forceAct_signed[6]: {fmt(baseline['force'].tolist())}")
        print(f"feedback_baseline_current[6]: {fmt(baseline['current'].tolist())}")

        triggered = False
        trigger_reason = "timeout"
        trigger_snapshot = baseline
        deadline = time.monotonic() + max(0.0, args.feedback_timeout)
        poll = max(0.01, args.feedback_poll)
        while time.monotonic() < deadline:
            snapshot = hand_feedback_snapshot(client)
            reason = feedback_trigger_reason(snapshot, baseline, args)
            if reason is not None:
                triggered = True
                trigger_reason = reason
                trigger_snapshot = snapshot
                break
            time.sleep(poll)

        print(f"feedback_triggered: {triggered}")
        print(f"feedback_trigger_reason: {trigger_reason}")
        print(f"feedback_trigger_forceAct_signed[6]: {fmt(trigger_snapshot['force'].tolist())}")
        print(f"feedback_trigger_current[6]: {fmt(trigger_snapshot['current'].tolist())}")
        print(f"feedback_trigger_angleAct[6]: {fmt(trigger_snapshot['angle_act'].tolist())}")
        print_feedback_delta_detail("feedback_trigger", trigger_snapshot, baseline)

        if triggered or args.close_on_feedback_timeout:
            write_thumb_inward_then_fingers_close_client(client, args, thumb_inner_angles, close_angles, "feedback_grasp")
            final = hand_feedback_snapshot(client)
            print(f"feedback_final_forceAct_signed[6]: {fmt(final['force'].tolist())}")
            print(f"feedback_final_current[6]: {fmt(final['current'].tolist())}")
            print(f"feedback_final_angleAct[6]: {fmt(final['angle_act'].tolist())}")
        else:
            print("feedback_close_skipped: no feedback before timeout")


def write_hand_angles_once(args: argparse.Namespace, angles: Sequence[int], label: str) -> None:
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print(f"hand_connect: {args.hand_ip}:{args.hand_port}")
        print("TCP connected")
        write_hand_angles_open_client(client, args, angles, label)
        time.sleep(max(0.0, args.hand_hold))
        angle_act = read_hand_angle_act(client)
        print(f"hand_angleAct_{label}: {fmt(angle_act)}")
        for index, target in enumerate(angles):
            if target != -1 and abs(int(angle_act[index]) - int(target)) > 100:
                print(
                    f"Warning: A{index} did not follow requested angle. requested={target} actual={angle_act[index]}",
                    file=sys.stderr,
                )


def write_thumb_then_close_once(
    args: argparse.Namespace,
    thumb_inner_angles: Sequence[int],
    close_angles: Sequence[int],
    label: str,
) -> None:
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print(f"hand_connect: {args.hand_ip}:{args.hand_port}")
        print("TCP connected")
        write_thumb_inward_then_fingers_close_client(client, args, thumb_inner_angles, close_angles, label)


def write_thumb_inward_then_fingers_close_client(
    client: ModbusTcpClient,
    args: argparse.Namespace,
    thumb_inner_angles: Sequence[int],
    close_angles: Sequence[int],
    label: str,
) -> None:
    thumb_channel = args.thumb_stage_channel
    thumb_target = int(close_angles[thumb_channel])
    angle_before = read_hand_angle_act(client)
    print(f"hand_angleAct_{label}_before_thumb_inner: {fmt(angle_before)}")
    write_staged_thumb_rotation_client(
        client,
        args,
        thumb_inner_angles,
        int(angle_before[thumb_channel]),
        thumb_target,
        f"{label}_thumb_inner",
    )
    time.sleep(max(0.0, args.thumb_inward_delay))
    angle_after_thumb = read_hand_angle_act(client)
    print(f"hand_angleAct_{label}_after_thumb_inner: {fmt(angle_after_thumb)}")
    if close_angles[thumb_channel] != -1 and abs(int(angle_after_thumb[thumb_channel]) - thumb_target) > args.thumb_close_tolerance:
        print(
            f"Warning: A{thumb_channel} not close enough before finger close; retrying. "
            f"requested={thumb_target} actual={angle_after_thumb[thumb_channel]}",
            file=sys.stderr,
        )
        write_staged_thumb_rotation_client(
            client,
            args,
            thumb_inner_angles,
            int(angle_after_thumb[thumb_channel]),
            thumb_target,
            f"{label}_thumb_inner_retry",
        )
        time.sleep(max(0.0, args.thumb_inward_delay))
        angle_after_thumb = read_hand_angle_act(client)
        print(f"hand_angleAct_{label}_after_thumb_inner_retry: {fmt(angle_after_thumb)}")
    if close_angles[thumb_channel] != -1 and abs(int(angle_after_thumb[thumb_channel]) - thumb_target) > args.thumb_close_tolerance:
        raise RuntimeError(
            f"refusing finger close: A{thumb_channel} requested={thumb_target} actual={angle_after_thumb[thumb_channel]}"
        )
    staged_close_angles = list(close_angles)
    thumb_bend_channel = 4
    if thumb_bend_channel != thumb_channel:
        staged_close_angles[thumb_bend_channel] = int(angle_after_thumb[thumb_bend_channel])
    write_hand_angles_open_client(client, args, staged_close_angles, f"{label}_fingers_close")
    time.sleep(max(0.0, args.thumb_bend_delay))
    write_hand_angles_open_client(client, args, close_angles, f"{label}_thumb_bend_close")
    time.sleep(max(0.0, args.hand_hold))
    angle_act = read_hand_angle_act(client)
    print(f"hand_angleAct_{label}: {fmt(angle_act)}")
    for index, target in enumerate(close_angles):
        if target != -1 and abs(int(angle_act[index]) - int(target)) > 100:
            print(
                f"Warning: A{index} did not follow requested angle. requested={target} actual={angle_act[index]}",
                file=sys.stderr,
            )


def write_thumb_then_open_once(
    args: argparse.Namespace,
    thumb_outer_angles: Sequence[int],
    open_angles: Sequence[int],
    label: str,
) -> None:
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print(f"hand_connect: {args.hand_ip}:{args.hand_port}")
        print("TCP connected")
        write_staged_thumb_rotation_client(
            client,
            args,
            thumb_outer_angles,
            args.hand_grasp_angle[args.thumb_stage_channel],
            args.hand_open_angle[args.thumb_stage_channel],
            f"{label}_thumb_outer",
        )
        time.sleep(max(0.0, args.thumb_outward_delay))
        write_hand_angles_open_client(client, args, open_angles, f"{label}_fingers_open")
        time.sleep(max(0.0, args.hand_hold))
        angle_act = read_hand_angle_act(client)
        print(f"hand_angleAct_{label}: {fmt(angle_act)}")
        for index, target in enumerate(open_angles):
            if target != -1 and abs(int(angle_act[index]) - int(target)) > 100:
                print(
                    f"Warning: A{index} did not follow requested angle. requested={target} actual={angle_act[index]}",
                file=sys.stderr,
            )


def write_safe_close_once(args: argparse.Namespace, safe_close_angles: Sequence[int], label: str) -> None:
    validate_angle_values(safe_close_angles)
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print(f"hand_connect: {args.hand_ip}:{args.hand_port}")
        print("TCP connected")
        angle_before = read_hand_angle_act(client)
        print(f"hand_angleAct_{label}_before: {fmt(angle_before)}")
        write_staged_thumb_rotation_client(
            client,
            args,
            safe_close_angles,
            int(angle_before[args.thumb_stage_channel]),
            int(safe_close_angles[args.thumb_stage_channel]),
            f"{label}_thumb_max",
        )
        write_hand_angles_open_client(client, args, safe_close_angles, f"{label}_fingers_close")
        time.sleep(max(0.0, args.hand_hold))
        angle_act = read_hand_angle_act(client)
        print(f"hand_angleAct_{label}: {fmt(angle_act)}")
        for index, target in enumerate(safe_close_angles):
            if target != -1 and abs(int(angle_act[index]) - int(target)) > 100:
                print(
                    f"Warning: A{index} did not follow requested angle. requested={target} actual={angle_act[index]}",
                    file=sys.stderr,
                )


def start_feedback_monitor(
    args: argparse.Namespace,
    thumb_inner_angles: Sequence[int],
    close_angles: Sequence[int],
) -> Tuple[threading.Event, threading.Thread, Dict[str, object]]:
    stop_event = threading.Event()
    state: Dict[str, object] = {
        "triggered": False,
        "thumb_rotated": False,
        "closed": False,
        "reason": "not_triggered",
        "error": None,
        "baseline_force": None,
        "baseline_current": None,
        "trigger_force": None,
        "trigger_current": None,
        "trigger_angle": None,
    }

    def worker() -> None:
        try:
            with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
                baseline = hand_feedback_snapshot(client)
                state["baseline_force"] = baseline["force"].tolist()
                state["baseline_current"] = baseline["current"].tolist()
                print(f"feedback_monitor_baseline_forceAct_signed[6]: {fmt(state['baseline_force'])}")
                print(f"feedback_monitor_baseline_current[6]: {fmt(state['baseline_current'])}")

                poll = max(0.01, args.feedback_poll)
                deadline = time.monotonic() + args.feedback_timeout if args.feedback_timeout > 0.0 else None
                while not stop_event.is_set():
                    if deadline is not None and time.monotonic() >= deadline:
                        state["reason"] = "feedback_timeout"
                        return
                    snapshot = hand_feedback_snapshot(client)
                    force_delta, current_delta = feedback_deltas(snapshot, baseline)
                    max_force_delta = int(np.max(force_delta))
                    max_current_delta = int(np.max(current_delta))
                    reason = feedback_trigger_reason(snapshot, baseline, args)
                    if reason is not None:
                        state["triggered"] = True
                        state["reason"] = reason

                    if state["triggered"]:
                        state["trigger_force"] = snapshot["force"].tolist()
                        state["trigger_current"] = snapshot["current"].tolist()
                        state["trigger_angle"] = snapshot["angle_act"].tolist()
                        print(f"feedback_monitor_trigger_reason: {state['reason']}")
                        print(f"feedback_monitor_trigger_forceAct_signed[6]: {fmt(state['trigger_force'])}")
                        print(f"feedback_monitor_trigger_current[6]: {fmt(state['trigger_current'])}")
                        print(f"feedback_monitor_trigger_angleAct[6]: {fmt(state['trigger_angle'])}")
                        print_feedback_delta_detail("feedback_monitor_trigger", snapshot, baseline)
                        write_thumb_inward_then_fingers_close_client(
                            client,
                            args,
                            thumb_inner_angles,
                            close_angles,
                            "feedback",
                        )
                        state["thumb_rotated"] = True
                        state["closed"] = True
                        return
                    time.sleep(poll)
        except Exception as exc:
            state["error"] = str(exc)

    thread = threading.Thread(target=worker, name="hand_feedback_monitor", daemon=True)
    thread.start()
    return stop_event, thread, state


def stop_feedback_monitor(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    thread.join(timeout=2.0)


def feedback_trigger_reason(snapshot: Dict[str, np.ndarray], baseline: Dict[str, np.ndarray], args: argparse.Namespace) -> Optional[str]:
    force_delta, current_delta = feedback_deltas(snapshot, baseline)
    force_index, max_force_delta = max_delta_channel(force_delta)
    current_index, max_current_delta = max_delta_channel(current_delta)
    if max_force_delta >= args.feedback_force_delta:
        return f"force_delta={max_force_delta} channel={hand_channel_label(force_index)} index={force_index}"
    if max_current_delta >= args.feedback_current_delta:
        return f"current_delta={max_current_delta} channel={hand_channel_label(current_index)} index={current_index}"
    return None


def run_feedback_limited_phase(
    controller: OfficialStyleArmSdkController,
    duration: float,
    q_left_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
    q_right_start: Dict[int, float],
    q_right_goal: Dict[int, float],
    weight_start: float,
    weight_goal: float,
    args: argparse.Namespace,
    thumb_inner_angles: Sequence[int],
    close_angles: Sequence[int],
) -> Tuple[Dict[int, float], Dict[str, object]]:
    state: Dict[str, object] = {
        "triggered": False,
        "thumb_rotated": False,
        "closed": False,
        "stopped_arm": False,
        "reason": "not_triggered",
        "baseline_force": None,
        "baseline_current": None,
        "trigger_force": None,
        "trigger_current": None,
        "trigger_angle": None,
        "max_force_delta": 0,
        "max_current_delta": 0,
        "final_force": None,
        "final_current": None,
        "final_angle": None,
        "stop_phase": None,
    }
    q_right_last = dict(q_right_start)

    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        baseline = hand_feedback_snapshot(client)
        state["baseline_force"] = baseline["force"].tolist()
        state["baseline_current"] = baseline["current"].tolist()
        print(f"feedback_limited_baseline_forceAct_signed[6]: {fmt(state['baseline_force'])}")
        print(f"feedback_limited_baseline_current[6]: {fmt(state['baseline_current'])}")

        start_time = time.monotonic()
        next_tick = start_time
        next_feedback = start_time
        feedback_deadline = start_time + args.feedback_timeout if args.feedback_timeout > 0.0 else None
        feedback_timed_out = False
        poll = max(0.01, args.feedback_poll)
        while True:
            now = time.monotonic()
            elapsed = now - start_time
            phase = min(max(elapsed / max(duration, 1e-6), 0.0), 1.0)
            ratio = phase * phase * phase * (phase * (phase * 6.0 - 15.0) + 10.0)
            q_right = interpolate_positions(q_right_start, q_right_goal, ratio)
            q_right_last = dict(q_right)
            weight_phase = phase * phase * (3.0 - 2.0 * phase)
            weight = weight_start + weight_phase * (weight_goal - weight_start)
            controller.write_group_command(
                q_left=q_left_hold,
                q_right=q_right,
                q_waist=q_waist_hold,
                weight=weight,
                kp_right=args.kp_right,
                kd_right=args.kd_right,
                kp_left=args.kp_left,
                kd_left=args.kd_left,
                kp_waist=args.kp_waist,
                kd_waist=args.kd_waist,
            )

            now = time.monotonic()
            if not feedback_timed_out and now >= next_feedback:
                if feedback_deadline is not None and now >= feedback_deadline:
                    state["reason"] = "feedback_timeout"
                    feedback_timed_out = True
                else:
                    snapshot = hand_feedback_snapshot(client)
                    force_delta, current_delta = feedback_deltas(snapshot, baseline)
                    state["max_force_delta"] = max(state["max_force_delta"], int(np.max(force_delta)))
                    state["max_current_delta"] = max(state["max_current_delta"], int(np.max(current_delta)))
                    reason = feedback_trigger_reason(snapshot, baseline, args) if phase >= args.feedback_min_phase else None
                    if reason is not None:
                        state["triggered"] = True
                        state["stopped_arm"] = True
                        state["reason"] = reason
                        state["trigger_force"] = snapshot["force"].tolist()
                        state["trigger_current"] = snapshot["current"].tolist()
                        state["trigger_angle"] = snapshot["angle_act"].tolist()
                        state["stop_phase"] = phase
                        print(f"feedback_limited_trigger_reason: {state['reason']}")
                        print(f"feedback_limited_stop_phase: {phase:.3f}")
                        print(f"feedback_limited_trigger_forceAct_signed[6]: {fmt(state['trigger_force'])}")
                        print(f"feedback_limited_trigger_current[6]: {fmt(state['trigger_current'])}")
                        print(f"feedback_limited_trigger_angleAct[6]: {fmt(state['trigger_angle'])}")
                        print_feedback_delta_detail("feedback_limited_trigger", snapshot, baseline)
                        write_thumb_inward_then_fingers_close_client(
                            client,
                            args,
                            thumb_inner_angles,
                            close_angles,
                            "feedback_limited",
                        )
                        state["thumb_rotated"] = True
                        state["closed"] = True
                        return q_right_last, state
                next_feedback = now + poll

            if phase >= 1.0:
                final = hand_feedback_snapshot(client)
                force_delta, current_delta = feedback_deltas(final, baseline)
                state["max_force_delta"] = max(state["max_force_delta"], int(np.max(force_delta)))
                state["max_current_delta"] = max(state["max_current_delta"], int(np.max(current_delta)))
                state["final_force"] = final["force"].tolist()
                state["final_current"] = final["current"].tolist()
                state["final_angle"] = final["angle_act"].tolist()
                print(f"feedback_limited_max_force_delta: {state['max_force_delta']}")
                print(f"feedback_limited_max_current_delta: {state['max_current_delta']}")
                print(f"feedback_limited_final_forceAct_signed[6]: {fmt(state['final_force'])}")
                print(f"feedback_limited_final_current[6]: {fmt(state['final_current'])}")
                print(f"feedback_limited_final_angleAct[6]: {fmt(state['final_angle'])}")
                print_feedback_delta_detail("feedback_limited_final", final, baseline)
                if args.close_at_approach_end:
                    state["triggered"] = True
                    state["stopped_arm"] = True
                    state["reason"] = "approach_end_fallback"
                    state["trigger_force"] = state["final_force"]
                    state["trigger_current"] = state["final_current"]
                    state["trigger_angle"] = state["final_angle"]
                    state["stop_phase"] = phase
                    print("feedback_limited_approach_end_fallback: true")
                    write_thumb_inward_then_fingers_close_client(
                        client,
                        args,
                        thumb_inner_angles,
                        close_angles,
                        "approach_end_fallback",
                    )
                    state["thumb_rotated"] = True
                    state["closed"] = True
                return q_right_last, state
            next_tick += controller.control_dt
            time.sleep(max(0.0, next_tick - time.monotonic()))


def monitor_hand_feedback_window(args: argparse.Namespace, label: str, duration: float) -> None:
    if duration <= 0.0:
        return
    try:
        with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
            baseline = hand_feedback_snapshot(client)
            max_force_delta = 0
            max_current_delta = 0
            deadline = time.monotonic() + duration
            poll = max(0.01, args.feedback_poll)
            last = baseline
            while time.monotonic() < deadline:
                last = hand_feedback_snapshot(client)
                force_delta, current_delta = feedback_deltas(last, baseline)
                max_force_delta = max(max_force_delta, int(np.max(force_delta)))
                max_current_delta = max(max_current_delta, int(np.max(current_delta)))
                time.sleep(poll)
            print(f"{label}_feedback_max_force_delta: {max_force_delta}")
            print(f"{label}_feedback_max_current_delta: {max_current_delta}")
            print(f"{label}_feedback_final_forceAct_signed[6]: {fmt(last['force'].tolist())}")
            print(f"{label}_feedback_final_current[6]: {fmt(last['current'].tolist())}")
            print(f"{label}_feedback_final_angleAct[6]: {fmt(last['angle_act'].tolist())}")
            print_feedback_delta_detail(f"{label}_feedback_final", last, baseline)
    except Exception as exc:
        print(f"Warning: {label} feedback monitor failed: {exc}", file=sys.stderr)


def read_hand_feedback_once(args: argparse.Namespace, label: str) -> Dict[str, np.ndarray]:
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        snapshot = hand_feedback_snapshot(client)
    print(f"{label}_forceAct_signed[6]: {fmt(snapshot['force'].tolist())}")
    print(f"{label}_current[6]: {fmt(snapshot['current'].tolist())}")
    print(f"{label}_angleAct[6]: {fmt(snapshot['angle_act'].tolist())}")
    return snapshot


def classify_finger_obstruction(
    args: argparse.Namespace,
    snapshot: Dict[str, np.ndarray],
    close_angles: Sequence[int],
    label: str,
) -> Dict[str, object]:
    gaps = []
    blocked = []
    for index in range(4):
        target = int(close_angles[index])
        actual = int(snapshot["angle_act"][index])
        gap = 0 if target == -1 else max(0, actual - target)
        gaps.append(gap)
        if target != -1 and gap >= args.verify_finger_gap:
            blocked.append(index)
    result = {
        "gaps": gaps,
        "blocked": blocked,
        "blocked_count": len(blocked),
        "passed": len(blocked) >= args.verify_min_blocked_fingers,
    }
    print(f"{label}_finger_gaps_A0_A3: {fmt(gaps)}")
    print(f"{label}_blocked_fingers: {' '.join(f'A{i}' for i in blocked) if blocked else 'none'}")
    print(f"{label}_blocked_count: {len(blocked)}")
    print(f"{label}_obstruction_passed: {result['passed']}")
    return result


def classification_is_confidently_empty(
    args: argparse.Namespace,
    classification: Dict[str, object],
) -> bool:
    gaps = [int(value) for value in classification["gaps"]]
    confidently_empty = bool(gaps) and max(gaps) <= args.verify_empty_max_gap
    print(f"verification_confidently_empty: {str(confidently_empty).lower()}")
    return confidently_empty


def evaluate_lift_retention(
    args: argparse.Namespace,
    before: Dict[str, np.ndarray],
    after: Dict[str, np.ndarray],
    before_classification: Dict[str, object],
    after_classification: Dict[str, object],
) -> Tuple[bool, str]:
    initially_blocked = [int(index) for index in before_classification["blocked"]]
    finger_drop = [
        max(0, int(before["angle_act"][index]) - int(after["angle_act"][index]))
        for index in range(4)
    ]
    blocked_drop = max((finger_drop[index] for index in initially_blocked), default=0)
    force_change = np.abs(after["force"] - before["force"])
    current_change = np.abs(after["current"] - before["current"])
    print(f"verification_lift_finger_drop_A0_A3: {fmt(finger_drop)}")
    print(f"verification_lift_max_blocked_finger_drop: {blocked_drop}")
    print(f"verification_lift_force_change[6]: {fmt(force_change.tolist())}")
    print(f"verification_lift_current_change[6]: {fmt(current_change.tolist())}")

    if not before_classification["passed"]:
        return False, "insufficient_finger_obstruction_before_lift"
    if not after_classification["passed"]:
        return False, "finger_obstruction_not_retained_after_lift"
    if blocked_drop > args.verify_max_finger_drop:
        return False, f"finger_angle_drop={blocked_drop} exceeds {args.verify_max_finger_drop}"
    return True, "finger_obstruction_retained_after_lift"


def compute_vision_target_while_holding(
    args: argparse.Namespace,
    pre_target: np.ndarray,
    controller: OfficialStyleArmSdkController,
    q_left_hold: Dict[int, float],
    q_right_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
) -> np.ndarray:
    result: Dict[str, object] = {}

    def worker() -> None:
        try:
            result["target"] = compute_vision_grasp_target(args, pre_target)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker, name="retry-redetect", daemon=True)
    thread.start()
    next_tick = time.monotonic()
    while thread.is_alive():
        controller.write_group_command(
            q_left=q_left_hold,
            q_right=q_right_hold,
            q_waist=q_waist_hold,
            weight=args.weight,
            kp_right=args.kp_right,
            kd_right=args.kd_right,
            kp_left=args.kp_left,
            kd_left=args.kd_left,
            kp_waist=args.kp_waist,
            kd_waist=args.kd_waist,
        )
        next_tick += controller.control_dt
        time.sleep(max(0.0, next_tick - time.monotonic()))
    thread.join()
    if "error" in result:
        raise RuntimeError(f"retry redetection failed: {result['error']}") from result["error"]
    return np.asarray(result["target"], dtype=float)


class MotionTrackingError(RuntimeError):
    def __init__(self, message: str, q_actual: Dict[int, float]):
        super().__init__(message)
        self.q_actual = dict(q_actual)


def read_actual_right(
    controller: OfficialStyleArmSdkController,
    right_joints: Sequence[int],
) -> Dict[int, float]:
    state = controller.wait_for_low_state(2.0)
    return joint_positions(state, right_joints)


def hold_until_reached(
    controller: OfficialStyleArmSdkController,
    args: argparse.Namespace,
    q_left_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
    right_joints: Sequence[int],
    goal: Dict[int, float],
    phase_label: str,
) -> Dict[int, float]:
    deadline = time.monotonic() + max(0.0, args.clearance_settle_duration)
    next_tick = time.monotonic()
    consecutive_reached = 0
    actual = read_actual_right(controller, right_joints)
    while True:
        controller.write_group_command(
            q_left=q_left_hold,
            q_right=goal,
            q_waist=q_waist_hold,
            weight=args.weight,
            kp_right=args.kp_right,
            kd_right=args.kd_right,
            kp_left=args.kp_left,
            kd_left=args.kd_left,
            kp_waist=args.kp_waist,
            kd_waist=args.kd_waist,
        )
        actual = read_actual_right(controller, right_joints)
        errors = {joint: abs(float(actual[joint]) - float(goal[joint])) for joint in right_joints}
        max_joint = max(errors, key=errors.get)
        max_error = float(errors[max_joint])
        if max_error <= args.clearance_joint_tolerance:
            consecutive_reached += 1
            if consecutive_reached >= 3:
                print(f"{phase_label}_joint_tracking_error_max_rad: {max_error:.4f}")
                return actual
        else:
            consecutive_reached = 0
        if time.monotonic() >= deadline:
            print(f"{phase_label}_joint_tracking_error_max_rad: {max_error:.4f}")
            raise MotionTrackingError(
                f"{phase_label} blocked or not reached: joint={max_joint} "
                f"error={max_error:.3f} exceeds {args.clearance_joint_tolerance:.3f}",
                actual,
            )
        next_tick += controller.control_dt
        time.sleep(max(0.0, next_tick - time.monotonic()))


def run_tracked_segment(
    controller: OfficialStyleArmSdkController,
    args: argparse.Namespace,
    q_left_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
    right_joints: Sequence[int],
    goal: Dict[int, float],
    minimum_duration: float,
    phase_label: str,
) -> Dict[int, float]:
    actual_start = read_actual_right(controller, right_joints)
    actual_values = np.asarray([actual_start[joint] for joint in right_joints], dtype=float)
    goal_values = np.asarray([goal[joint] for joint in right_joints], dtype=float)
    duration = segment_duration(minimum_duration, actual_values, goal_values, args.max_joint_speed)
    print(f"{phase_label}: tracked motion from measured state for {duration:.2f} s.")
    start_time = time.monotonic()
    next_tick = start_time
    next_tracking_check = start_time
    excessive_error_since = None
    while True:
        now = time.monotonic()
        phase = min(max((now - start_time) / max(duration, 1e-6), 0.0), 1.0)
        ratio = phase * phase * phase * (phase * (phase * 6.0 - 15.0) + 10.0)
        command = interpolate_positions(actual_start, goal, ratio)
        controller.write_group_command(
            q_left=q_left_hold,
            q_right=command,
            q_waist=q_waist_hold,
            weight=args.weight,
            kp_right=args.kp_right,
            kd_right=args.kd_right,
            kp_left=args.kp_left,
            kd_left=args.kd_left,
            kp_waist=args.kp_waist,
            kd_waist=args.kd_waist,
        )
        if now >= next_tracking_check:
            actual = read_actual_right(controller, right_joints)
            errors = {joint: abs(float(actual[joint]) - float(command[joint])) for joint in right_joints}
            max_joint = max(errors, key=errors.get)
            max_error = float(errors[max_joint])
            if max_error > args.motion_tracking_error_limit:
                if excessive_error_since is None:
                    excessive_error_since = now
                elif now - excessive_error_since >= args.motion_tracking_error_duration:
                    raise MotionTrackingError(
                        f"{phase_label} obstructed during motion: joint={max_joint} "
                        f"tracking_error={max_error:.3f} exceeds {args.motion_tracking_error_limit:.3f}",
                        actual,
                    )
            else:
                excessive_error_since = None
            next_tracking_check = now + 0.02
        if phase >= 1.0:
            break
        next_tick += controller.control_dt
        time.sleep(max(0.0, next_tick - time.monotonic()))
    return hold_until_reached(
        controller,
        args,
        q_left_hold,
        q_waist_hold,
        right_joints,
        goal,
        phase_label,
    )


def operator_safety_hold(
    controller: OfficialStyleArmSdkController,
    args: argparse.Namespace,
    q_left_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
    right_joints: Sequence[int],
    q_fallback: Dict[int, float],
    reason: str,
) -> Dict[int, float]:
    try:
        q_hold = read_actual_right(controller, right_joints)
    except Exception:
        q_hold = dict(q_fallback)
    interactive = sys.stdin.isatty()
    print("", file=sys.stderr)
    print(f"SAFETY HOLD: {reason}", file=sys.stderr)
    print("The arm will remain actively held; the hand will not be opened.", file=sys.stderr)
    if interactive:
        print("Support the arm and bottle, clear the table, then type RELEASE and press Enter.", file=sys.stderr)
    else:
        print("Non-interactive terminal: keep this process alive and stop it only after physically supporting the arm.", file=sys.stderr)
    next_tick = time.monotonic()
    while True:
        controller.write_group_command(
            q_left=q_left_hold,
            q_right=q_hold,
            q_waist=q_waist_hold,
            weight=args.weight,
            kp_right=args.kp_right,
            kd_right=args.kd_right,
            kp_left=args.kp_left,
            kd_left=args.kd_left,
            kp_waist=args.kp_waist,
            kd_waist=args.kd_waist,
        )
        if interactive:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if readable:
                response = sys.stdin.readline()
                if response.strip().upper() == "RELEASE":
                    print("Operator confirmed physical support; releasing arm_sdk.", file=sys.stderr)
                    return q_hold
                print("Safety hold remains active. Type RELEASE only after supporting the arm.", file=sys.stderr)
        next_tick += controller.control_dt
        try:
            time.sleep(max(0.0, next_tick - time.monotonic()))
        except KeyboardInterrupt:
            print("Safety hold remains active after Ctrl-C. Type RELEASE after supporting the arm.", file=sys.stderr)


def run_clearance_retreat(
    controller: OfficialStyleArmSdkController,
    args: argparse.Namespace,
    q_left_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
    right_joints: Sequence[int],
    q_start: Dict[int, float],
    q_retrace: Dict[int, float],
    q_retrace_values: np.ndarray,
    q_clearance_side: Dict[int, float],
    q_clearance_side_values: np.ndarray,
    q_destination: Dict[int, float],
    q_destination_values: np.ndarray,
    label: str,
) -> Dict[int, float]:
    del q_start, q_retrace_values, q_clearance_side_values, q_destination_values
    print(f"{label}: reverse the proven approach path to the pregrasp waypoint.")
    run_tracked_segment(
        controller,
        args,
        q_left_hold,
        q_waist_hold,
        right_joints,
        q_retrace,
        args.lateral_duration,
        f"{label}_retrace",
    )
    print(f"{label}: lift and move laterally only after reaching the table-side pregrasp waypoint.")
    run_tracked_segment(
        controller,
        args,
        q_left_hold,
        q_waist_hold,
        right_joints,
        q_clearance_side,
        args.clearance_side_duration,
        f"{label}_side",
    )
    print(f"{label}: lower only after clearing the table edge.")
    return run_tracked_segment(
        controller,
        args,
        q_left_hold,
        q_waist_hold,
        right_joints,
        q_destination,
        args.clearance_lift_duration,
        f"{label}_lower",
    )


def solve_right_q(
    target: np.ndarray,
    waist_q: np.ndarray,
    right_joints: Sequence[int],
    seed_motor_q: np.ndarray,
    tolerance: float,
) -> Tuple[object, np.ndarray, Dict[int, float]]:
    signs = np.ones(7)
    seed_model_q = motor_to_model_q(seed_motor_q, signs)
    ik = solve_position_ik(
        target=target,
        waist_q=waist_q,
        joint_indices=right_joints,
        signs=signs,
        seed=seed_model_q,
        tolerance=tolerance,
    )
    q_values = model_to_motor_q(ik.q_model, signs)
    return ik, q_values, {joint: float(value) for joint, value in zip(right_joints, q_values)}


def solve_right_q_fixed_wrist(
    target: np.ndarray,
    waist_q: np.ndarray,
    right_joints: Sequence[int],
    seed_motor_q: np.ndarray,
    tolerance: float,
    wrist_roll_target: float,
    wrist_pitch_target: float,
    wrist_yaw_target: float,
) -> Tuple[object, np.ndarray, Dict[int, float]]:
    signs = np.ones(7)
    fixed_targets = {
        G1JointIndex.RightWristRoll: float(wrist_roll_target),
        G1JointIndex.RightWristPitch: float(wrist_pitch_target),
        G1JointIndex.RightWristYaw: float(wrist_yaw_target),
    }

    def fixed_wrist_fk(q_model, waist_values, joint_indices):
        q_fixed = np.asarray(q_model, dtype=float).copy()
        for joint, value in fixed_targets.items():
            if joint in joint_indices:
                q_fixed[joint_indices.index(joint)] = value
        return right_hand_pose(q_fixed, waist_values, joint_indices)[:3, 3]

    seed_model_q = motor_to_model_q(seed_motor_q, signs)
    for joint, value in fixed_targets.items():
        if joint in right_joints:
            seed_model_q[right_joints.index(joint)] = value
    ik = solve_position_ik_for(
        target=target,
        waist_q=waist_q,
        joint_indices=right_joints,
        signs=signs,
        seed=seed_model_q,
        tolerance=tolerance,
        fk_func=fixed_wrist_fk,
    )
    q_model = np.asarray(ik.q_model, dtype=float).copy()
    for joint, value in fixed_targets.items():
        if joint in right_joints:
            q_model[right_joints.index(joint)] = value
    reached = fixed_wrist_fk(q_model, waist_q, right_joints)
    residual = float(np.linalg.norm(np.asarray(target, dtype=float) - reached))
    ik.q_model = q_model
    ik.reached = reached
    ik.residual = residual
    ik.success = residual <= tolerance
    ik.message = "fixed-wrist IK converged" if ik.success else "fixed-wrist IK did not reach tolerance"
    q_values = model_to_motor_q(q_model, signs)
    return ik, q_values, {joint: float(value) for joint, value in zip(right_joints, q_values)}


def command_position(q_values: np.ndarray, waist_q: np.ndarray, right_joints: Sequence[int]) -> np.ndarray:
    model_q = motor_to_model_q(q_values, np.ones(7))
    return right_hand_pose(model_q, waist_q, right_joints)[:3, 3]


def validate_wrist_target(joint: int, value: float) -> None:
    low, high = WRIST_LIMITS[joint]
    if not low <= value <= high:
        raise ValueError(f"{WRIST_NAMES[joint]} must be within [{low:.4f}, {high:.4f}] rad")


def apply_wrist_targets(
    q_values: np.ndarray,
    q_targets: Dict[int, float],
    right_joints: Sequence[int],
    wrist_roll_target: float,
    wrist_pitch_target: float,
    wrist_yaw_target: float,
) -> Tuple[np.ndarray, Dict[int, float]]:
    adjusted_values = np.asarray(q_values, dtype=float).copy()
    adjusted_targets = dict(q_targets)
    wrist_targets = {
        G1JointIndex.RightWristRoll: wrist_roll_target,
        G1JointIndex.RightWristPitch: wrist_pitch_target,
        G1JointIndex.RightWristYaw: wrist_yaw_target,
    }
    for joint, value in wrist_targets.items():
        if joint in right_joints:
            index = right_joints.index(joint)
            adjusted_values[index] = float(value)
            adjusted_targets[joint] = float(value)
    return adjusted_values, adjusted_targets


def segment_duration(default_duration: float, q_from: np.ndarray, q_to: np.ndarray, max_speed: float) -> float:
    delta = float(np.max(np.abs(q_to - q_from)))
    return max(default_duration, delta / max(max_speed, 1e-6))


def print_plan_item(name: str, target: np.ndarray, ik, q_values: np.ndarray, q_prev: np.ndarray, right_joints) -> None:
    print(f"{name}_target_pelvis_xyz:", np.array2string(target, precision=4))
    print(f"{name}_ik_success:", ik.success)
    print(f"{name}_ik_residual_m: {ik.residual:.4f}")
    print(f"{name}_ik_reached_xyz:", np.array2string(ik.reached, precision=4))
    print(f"{name}_right_goal:")
    print(format_joint_table(right_joints, q_values))
    print(f"{name}_right_delta:")
    print(format_joint_delta_table(right_joints, q_prev, q_values))


def print_command_position(name: str, target: np.ndarray, q_values: np.ndarray, waist_q: np.ndarray, right_joints) -> None:
    reached = command_position(q_values, waist_q, right_joints)
    residual = float(np.linalg.norm(target - reached))
    print(f"{name}_command_reached_xyz:", np.array2string(reached, precision=4))
    print(f"{name}_command_residual_m: {residual:.4f}")


def validate_hand_args(args: argparse.Namespace) -> int:
    if args.hand_speed < 0 or args.hand_speed > 1000 or args.hand_force < 0 or args.hand_force > 3000:
        print("Refusing hand command: speed must be in 0..1000 and force must be in 0..3000.", file=sys.stderr)
        return 2
    if any(value < -1 or value > 1000 for value in args.hand_open_angle + args.hand_grasp_angle + args.hand_safe_close_angle):
        print("Refusing hand command: all hand angles must be -1 or in 0..1000.", file=sys.stderr)
        return 2
    if args.thumb_rotate_open is not None and (args.thumb_rotate_open < -1 or args.thumb_rotate_open > 1000):
        print("Refusing hand command: --thumb-rotate-open must be -1 or in 0..1000.", file=sys.stderr)
        return 2
    if args.thumb_rotate_close is not None and (args.thumb_rotate_close < -1 or args.thumb_rotate_close > 1000):
        print("Refusing hand command: --thumb-rotate-close must be -1 or in 0..1000.", file=sys.stderr)
        return 2
    if args.feedback_timeout < 0.0 or args.feedback_poll <= 0.0:
        print("Refusing feedback command: timeout must be non-negative and poll must be positive.", file=sys.stderr)
        return 2
    if args.feedback_min_phase < 0.0 or args.feedback_min_phase > 1.0:
        print("Refusing feedback command: --feedback-min-phase must be in 0..1.", file=sys.stderr)
        return 2
    if args.thumb_inward_delay < 0.0:
        print("Refusing feedback command: --thumb-inward-delay must be non-negative.", file=sys.stderr)
        return 2
    if args.thumb_bend_delay < 0.0:
        print("Refusing feedback command: --thumb-bend-delay must be non-negative.", file=sys.stderr)
        return 2
    if args.thumb_outward_delay < 0.0:
        print("Refusing feedback command: --thumb-outward-delay must be non-negative.", file=sys.stderr)
        return 2
    if args.thumb_stage_channel < 0 or args.thumb_stage_channel > 5:
        print("Refusing hand command: --thumb-stage-channel must be in 0..5.", file=sys.stderr)
        return 2
    if args.thumb_rotate_speed < 0 or args.thumb_rotate_speed > 1000:
        print("Refusing hand command: --thumb-rotate-speed must be in 0..1000.", file=sys.stderr)
        return 2
    if args.thumb_rotate_force < 0 or args.thumb_rotate_force > 3000:
        print("Refusing hand command: --thumb-rotate-force must be in 0..3000.", file=sys.stderr)
        return 2
    if args.thumb_rotate_hold < 0.0:
        print("Refusing hand command: --thumb-rotate-hold must be non-negative.", file=sys.stderr)
        return 2
    if args.thumb_rotate_step <= 0 or args.thumb_rotate_step > 1000:
        print("Refusing hand command: --thumb-rotate-step must be in 1..1000.", file=sys.stderr)
        return 2
    if args.thumb_close_tolerance < 0:
        print("Refusing hand command: --thumb-close-tolerance must be non-negative.", file=sys.stderr)
        return 2
    if args.feedback_force_delta < 0 or args.feedback_current_delta < 0:
        print("Refusing feedback command: thresholds must be non-negative.", file=sys.stderr)
        return 2
    if args.side_standoff_y < 0.0:
        print("Refusing vision command: --side-standoff-y must be non-negative.", file=sys.stderr)
        return 2
    if args.detector_timeout <= 0.0 or args.detector_retries < 0 or args.detector_retry_delay < 0.0:
        print(
            "Refusing vision command: detector timeout must be positive; retries and retry delay must be non-negative.",
            file=sys.stderr,
        )
        return 2
    if args.verify_lift_distance <= 0.0 or args.verify_lift_distance > 0.10:
        print("Refusing verification command: --verify-lift-distance must be in (0, 0.10].", file=sys.stderr)
        return 2
    if args.verify_lift_duration <= 0.0 or args.verify_hold < 0.0:
        print("Refusing verification command: lift duration must be positive and hold must be non-negative.", file=sys.stderr)
        return 2
    if args.verify_finger_gap < 0 or args.verify_finger_gap > 1000:
        print("Refusing verification command: --verify-finger-gap must be in 0..1000.", file=sys.stderr)
        return 2
    if args.verify_min_blocked_fingers < 1 or args.verify_min_blocked_fingers > 4:
        print("Refusing verification command: --verify-min-blocked-fingers must be in 1..4.", file=sys.stderr)
        return 2
    if args.verify_empty_max_gap < 0 or args.verify_empty_max_gap >= args.verify_finger_gap:
        print(
            "Refusing verification command: --verify-empty-max-gap must be non-negative and smaller than "
            "--verify-finger-gap.",
            file=sys.stderr,
        )
        return 2
    if args.verify_max_finger_drop < 0 or args.verify_max_finger_drop > 1000:
        print("Refusing verification command: --verify-max-finger-drop must be in 0..1000.", file=sys.stderr)
        return 2
    if args.retry_approach_extra < 0.0 or args.retry_approach_extra > 0.03:
        print("Refusing retry command: --retry-approach-extra must be in 0..0.03 m.", file=sys.stderr)
        return 2
    if args.test_first_attempt_shortfall < 0.0 or args.test_first_attempt_shortfall > 0.08:
        print("Refusing retry test: --test-first-attempt-shortfall must be in 0..0.08 m.", file=sys.stderr)
        return 2
    if args.clearance_lift_distance <= 0.0 or args.clearance_lift_distance > 0.25:
        print("Refusing clearance command: --clearance-lift-distance must be in (0, 0.25].", file=sys.stderr)
        return 2
    if args.clearance_lift_duration <= 0.0 or args.clearance_side_duration <= 0.0:
        print("Refusing clearance command: clearance durations must be positive.", file=sys.stderr)
        return 2
    if args.clearance_settle_duration <= 0.0 or args.clearance_settle_duration > 10.0:
        print("Refusing clearance command: --clearance-settle-duration must be in (0, 10].", file=sys.stderr)
        return 2
    if args.clearance_joint_tolerance <= 0.0 or args.clearance_joint_tolerance > 0.5:
        print("Refusing clearance command: --clearance-joint-tolerance must be in (0, 0.5].", file=sys.stderr)
        return 2
    if args.motion_tracking_error_limit <= 0.0 or args.motion_tracking_error_limit > 0.8:
        print("Refusing motion command: --motion-tracking-error-limit must be in (0, 0.8].", file=sys.stderr)
        return 2
    if args.motion_tracking_error_duration <= 0.0 or args.motion_tracking_error_duration > 2.0:
        print("Refusing motion command: --motion-tracking-error-duration must be in (0, 2].", file=sys.stderr)
        return 2
    return 0


def hand_tcp_preflight(args: argparse.Namespace) -> bool:
    try:
        with socket.create_connection((args.hand_ip, args.hand_port), timeout=args.hand_timeout):
            return True
    except OSError as exc:
        print(f"hand_preflight_connect: {args.hand_ip}:{args.hand_port} failed: {exc}", file=sys.stderr)
        print(HAND_NETWORK_HINT, file=sys.stderr)
        return False


def announce_text(args: argparse.Namespace, text: str, label: str) -> None:
    if args.no_announce or not text:
        return
    try:
        audio_client = AudioClient()
        audio_client.SetTimeout(10.0)
        audio_client.Init()
        code = audio_client.TtsMaker(text, args.announce_speaker_id)
        if code != 0:
            print(f"Warning: {label} TTS announcement failed with code {code}.", file=sys.stderr)
        else:
            print(f"TTS {label} announced: {text}")
    except Exception as exc:
        print(f"Warning: {label} TTS announcement failed: {exc}", file=sys.stderr)


def announce_start(args: argparse.Namespace) -> None:
    announce_text(args, args.announce_start_text, "start")


def announce_success(args: argparse.Namespace) -> None:
    announce_text(args, args.announce_text, "success")


def read_mjpg_rgb_frame(url: str) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required to read the MJPEG stream") from exc

    capture = cv2.VideoCapture(url)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open MJPEG stream: {url}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read one MJPEG frame: {url}")
        return frame
    finally:
        capture.release()


def read_http_rgbd_npz(url: str) -> Dict[str, object]:
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            payload = response.read()
    except Exception as exc:
        raise RuntimeError(f"failed to read HTTP RGB-D npz: {url}: {exc}") from exc

    try:
        data = np.load(io.BytesIO(payload))
        return {
            "source": "http_rgbd_npz",
            "color_bgr": np.asarray(data["color_bgr"]).copy(),
            "depth_raw": np.asarray(data["depth_raw"]).copy(),
            "depth_scale": float(data["depth_scale"]),
            "fx": float(data["fx"]),
            "fy": float(data["fy"]),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
            "rgbd_npz_url": url,
            "stamp": float(data["stamp"]) if "stamp" in data.files else None,
            "alignment_note": "RGB and depth are read from the same second-RealSense HTTP rgbd.npz buffer.",
        }
    except Exception as exc:
        raise RuntimeError(f"failed to decode HTTP RGB-D npz: {url}: {exc}") from exc


def scaled_mjpg_intrinsics(args: argparse.Namespace, image_shape: Sequence[int]) -> Tuple[float, float, float, float]:
    height, width = image_shape[:2]
    sx = float(width) / max(float(args.mjpg_reference_width), 1.0)
    sy = float(height) / max(float(args.mjpg_reference_height), 1.0)
    return (
        float(args.mjpg_fx) * sx,
        float(args.mjpg_fy) * sy,
        float(args.mjpg_cx) * sx,
        float(args.mjpg_cy) * sy,
    )


def read_realsense_depth_only(args: argparse.Namespace) -> Dict[str, object]:
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyrealsense2 is required for --mjpg-depth-source realsense") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if args.depth_serial:
        config.enable_device(args.depth_serial)
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.depth_fps)

    profile = pipeline.start(config)
    try:
        frames = None
        for _ in range(max(1, args.depth_warmup_frames)):
            frames = pipeline.wait_for_frames(5000)
        depth_frame = frames.get_depth_frame() if frames is not None else None
        if not depth_frame:
            raise RuntimeError("missing RealSense depth frame")
        intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        return {
            "source": "realsense_depth_only",
            "depth_raw": np.asanyarray(depth_frame.get_data()).copy(),
            "depth_scale": float(depth_sensor.get_depth_scale()),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "depth_serial": args.depth_serial,
        }
    finally:
        pipeline.stop()


def image_msg_to_depth_array(msg) -> Tuple[np.ndarray, float]:
    encoding = msg.encoding.lower()
    if encoding in ("16uc1", "mono16"):
        dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
        scale = 0.001
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
        scale = 1.0
    else:
        raise RuntimeError(f"unsupported ROS depth encoding: {msg.encoding}")

    array = np.frombuffer(msg.data, dtype=dtype)
    expected = int(msg.height) * int(msg.width)
    if array.size < expected:
        raise RuntimeError(f"ROS depth payload too small: got {array.size}, expected {expected}")
    array = array[:expected].reshape(int(msg.height), int(msg.width)).copy()
    return array, scale


def read_ros_aligned_depth(args: argparse.Namespace) -> Dict[str, object]:
    try:
        import rospy
        from sensor_msgs.msg import CameraInfo, Image
    except ModuleNotFoundError:
        ros_python = Path("/opt/ros/noetic/lib/python3/dist-packages")
        if ros_python.exists() and str(ros_python) not in sys.path:
            sys.path.insert(0, str(ros_python))
        try:
            import rospy
            from sensor_msgs.msg import CameraInfo, Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("rospy and sensor_msgs are required for --mjpg-depth-source ros") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("g1_mjpg_grasp_depth_reader", anonymous=True, disable_signals=True)

    depth_msg = rospy.wait_for_message(args.ros_depth_topic, Image, timeout=args.ros_timeout)
    info_msg = rospy.wait_for_message(args.ros_camera_info_topic, CameraInfo, timeout=args.ros_timeout)
    depth, scale = image_msg_to_depth_array(depth_msg)
    if len(info_msg.K) >= 6 and info_msg.K[0] > 0.0 and info_msg.K[4] > 0.0:
        fx = float(info_msg.K[0])
        fy = float(info_msg.K[4])
        cx = float(info_msg.K[2])
        cy = float(info_msg.K[5])
    else:
        raise RuntimeError("ROS CameraInfo has no usable K matrix")

    return {
        "source": "ros_aligned_depth",
        "depth_raw": depth,
        "depth_scale": scale,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "ros_depth_topic": args.ros_depth_topic,
        "ros_camera_info_topic": args.ros_camera_info_topic,
        "ros_depth_encoding": depth_msg.encoding,
        "alignment_note": "Depth is read from ROS aligned_depth_to_color topic and paired with MJPEG RGB.",
    }


def depth_point_for_mjpg_detection(
    depth_capture: Dict[str, object],
    det: Dict[str, object],
    rgb_shape: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    depth = depth_capture["depth_raw"]
    scale = float(depth_capture["depth_scale"])
    rgb_h, rgb_w = rgb_shape[:2]
    depth_h, depth_w = depth.shape[:2]
    sx = float(depth_w) / max(float(rgb_w), 1.0)
    sy = float(depth_h) / max(float(rgb_h), 1.0)

    x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
    cx_rgb, cy_rgb = [float(v) for v in det["center_xy"]]
    cx_depth = cx_rgb * sx
    cy_depth = cy_rgb * sy

    w = max(6, int((x2 - x1) * sx * 0.35))
    h = max(6, int((y2 - y1) * sy * 0.35))
    x0 = max(0, int(round(cx_depth - w * 0.5)))
    x3 = min(depth_w, int(round(cx_depth + w * 0.5)))
    y0 = max(0, int(round(cy_depth - h * 0.5)))
    y3 = min(depth_h, int(round(cy_depth + h * 0.5)))

    crop = depth[y0:y3, x0:x3].astype(np.float32) * scale
    valid = crop[np.isfinite(crop) & (crop > 0.05)]
    if valid.size == 0:
        raise RuntimeError("no valid depth samples in mapped detection crop")

    z = float(np.median(valid))
    x = (cx_depth - float(depth_capture["cx"])) * z / float(depth_capture["fx"])
    y = (cy_depth - float(depth_capture["cy"])) * z / float(depth_capture["fy"])
    return np.asarray([x, y, z], dtype=float), {
        "rgb_source": "mjpg",
        "depth_source": depth_capture.get("source", "depth_capture"),
        "rgb_size_wh": [int(rgb_w), int(rgb_h)],
        "depth_size_wh": [int(depth_w), int(depth_h)],
        "rgb_to_depth_scale_xy": [sx, sy],
        "rgb_center_xy": [cx_rgb, cy_rgb],
        "depth_center_xy": [cx_depth, cy_depth],
        "depth_crop_xyxy": [x0, y0, x3, y3],
        "valid_depth_samples": int(valid.size),
        "depth_median_m": z,
        "depth_intrinsics": {
            "fx": float(depth_capture["fx"]),
            "fy": float(depth_capture["fy"]),
            "cx": float(depth_capture["cx"]),
            "cy": float(depth_capture["cy"]),
        },
        "rgbd_npz_url": depth_capture.get("rgbd_npz_url"),
        "rgbd_stamp": depth_capture.get("stamp"),
        "ros_depth_topic": depth_capture.get("ros_depth_topic"),
        "ros_camera_info_topic": depth_capture.get("ros_camera_info_topic"),
        "ros_depth_encoding": depth_capture.get("ros_depth_encoding"),
        "alignment_note": depth_capture.get(
            "alignment_note",
            "RGB/depth pixels are scaled by image size; verify stream alignment if targets drift.",
        ),
    }


def fixed_depth_point_for_detection(args: argparse.Namespace, image_bgr: np.ndarray, det: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, object]]:
    if args.mjpg_assumed_depth <= 0.0:
        raise RuntimeError("--mjpg-assumed-depth must be positive")
    fx, fy, cx, cy = scaled_mjpg_intrinsics(args, image_bgr.shape)
    u, v = (float(det["center_xy"][0]), float(det["center_xy"][1]))
    z = float(args.mjpg_assumed_depth)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    height, width = image_bgr.shape[:2]
    return np.asarray([x, y, z], dtype=float), {
        "rgb_source": "mjpg",
        "rgb_url": args.mjpg_url,
        "rgb_size_wh": [int(width), int(height)],
        "reference_size_wh": [float(args.mjpg_reference_width), float(args.mjpg_reference_height)],
        "intrinsics_used": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "assumed_depth_m": z,
        "note": "RGB-only MJPEG fallback; point_camera uses --mjpg-assumed-depth.",
    }


def read_detector_server(args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object]]:
    payload = json.dumps({"object": args.object}).encode("utf-8")
    attempts = max(1, int(args.detector_retries) + 1)
    retry_delay = max(0.0, float(args.detector_retry_delay))
    last_error: Optional[str] = None
    result = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            args.detector_server_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=args.detector_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            if result.get("ok"):
                break
            last_error = f"detector server returned error: {result}"
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"detector server HTTP {exc.code}: {body}"
        except Exception as exc:
            last_error = f"failed to query detector server: {args.detector_server_url}: {exc}"
        if attempt < attempts:
            print(
                f"detector_server_retry: attempt={attempt}/{attempts} error={last_error}",
                file=sys.stderr,
            )
            time.sleep(retry_delay)

    if result is None or not result.get("ok"):
        raise RuntimeError(last_error or "detector server failed without an error message")
    return (
        np.asarray(result["point_camera_m"], dtype=float),
        result["detection"],
        {
            **result.get("depth_stats", {}),
            "detector_server_url": args.detector_server_url,
            "detector_request_count": result.get("request_count"),
            "detector_timing_s": result.get("timing_s"),
            "detector_source": result.get("source"),
            "rgbd_npz_url": result.get("rgbd_npz_url"),
            "stamp": result.get("stamp"),
        },
    )


def compute_realsense_detection(args: argparse.Namespace, fallback_reason: Optional[str] = None) -> Tuple[str, np.ndarray, Dict[str, object], Dict[str, object]]:
    model, model_path = load_model(args.object)
    capture_args = Namespace(
        host=args.vision_host,
        password=args.vision_password,
        interface=args.network_interface,
    )
    capture_info = run_capture(capture_args)
    det = detect_object(model, capture_info["color_bgr"], args.object)
    point_camera, depth_stats = depth_point_for_detection(capture_info, det)
    depth_stats = {
        **depth_stats,
        "fallback_source": "realsense",
        "fallback_reason": fallback_reason,
        "capture_source": capture_info.get("source", "realsense"),
    }
    return model_path, point_camera, det, depth_stats


def fallback_realsense_or_raise(args: argparse.Namespace, reason: str) -> Tuple[str, np.ndarray, Dict[str, object], Dict[str, object]]:
    if args.vision_fallback_source != "realsense":
        raise RuntimeError(reason)
    print(f"vision_primary_failed: {reason}", file=sys.stderr)
    print("vision_fallback_source: realsense", file=sys.stderr)
    return compute_realsense_detection(args, reason)


def detector_error_allows_local_fallback(reason: str) -> bool:
    # If the detector server is alive but returns 5xx, it may already own the
    # RealSense device. Retrying the server is safe; opening the camera here is not.
    blocked_fragments = (
        "detector server HTTP 5",
        "detector server returned error",
    )
    return not any(fragment in reason for fragment in blocked_fragments)


def compute_vision_grasp_target(args: argparse.Namespace, pre_target: np.ndarray) -> np.ndarray:
    if args.detector_server_url:
        try:
            model_path = "persistent_detector_server"
            point_camera, det, depth_stats = read_detector_server(args)
        except Exception as exc:
            reason = f"failed to query detector server {args.detector_server_url}: {exc}"
            if not detector_error_allows_local_fallback(reason):
                raise RuntimeError(
                    f"{reason}. Detector server is reachable but failed; refusing local RealSense fallback "
                    "because the server may already be using the camera."
                ) from exc
            model_path, point_camera, det, depth_stats = fallback_realsense_or_raise(
                args,
                reason,
            )
    elif args.vision_source == "mjpg":
        try:
            model, model_path = load_model(args.object)
            if args.mjpg_depth_source == "http-rgbd":
                capture_info = read_http_rgbd_npz(args.rgbd_npz_url)
                color_bgr = capture_info["color_bgr"]
            else:
                color_bgr = read_mjpg_rgb_frame(args.mjpg_url)
                capture_info = {
                    "source": "mjpg",
                    "color_bgr": color_bgr,
                }
            det = detect_object(model, color_bgr, args.object)
            if args.mjpg_depth_source == "http-rgbd":
                point_camera, depth_stats = depth_point_for_mjpg_detection(capture_info, det, color_bgr.shape)
            elif args.mjpg_depth_source == "ros":
                depth_capture = read_ros_aligned_depth(args)
                point_camera, depth_stats = depth_point_for_mjpg_detection(depth_capture, det, color_bgr.shape)
            elif args.mjpg_depth_source == "realsense":
                depth_capture = read_realsense_depth_only(args)
                point_camera, depth_stats = depth_point_for_mjpg_detection(depth_capture, det, color_bgr.shape)
            else:
                point_camera, depth_stats = fixed_depth_point_for_detection(args, color_bgr, det)
        except Exception as exc:
            model_path, point_camera, det, depth_stats = fallback_realsense_or_raise(
                args,
                f"failed to use MJPEG/RGB-D vision source: {exc}",
            )
    else:
        model_path, point_camera, det, depth_stats = compute_realsense_detection(args)

    point_pelvis = camera_to_pelvis(point_camera, args.camera_pitch)

    target = pre_target.copy()
    target[0] = point_pelvis[0] + args.vision_xy_offset[0]
    object_y_with_offset = point_pelvis[1] + args.vision_xy_offset[1]
    approach_side = -1.0 if pre_target[1] <= object_y_with_offset else 1.0
    target[1] = object_y_with_offset + approach_side * args.side_standoff_y
    target[2] = args.fixed_grasp_z

    print("vision_model:", model_path)
    print("vision_source:", args.vision_source)
    print("vision_detection:", det)
    print("vision_depth_stats:", depth_stats)
    print("vision_point_camera_m:", np.array2string(point_camera, precision=4))
    print("vision_point_pelvis_m:", np.array2string(point_pelvis, precision=4))
    print("vision_xy_offset_m:", np.array2string(np.asarray(args.vision_xy_offset, dtype=float), precision=4))
    print("vision_side_standoff_y_m:", f"{args.side_standoff_y:.4f}")
    print("vision_approach_side:", f"{approach_side:+.0f}")
    print("vision_fixed_grasp_z:", f"{args.fixed_grasp_z:.4f}")
    return target


def main() -> int:
    args = parse_args()
    if args.execute and not args.yes:
        print("Refusing to move without --yes.", file=sys.stderr)
        return 1
    if args.weight <= 0.0 or args.weight > 1.0:
        print("--weight must be in (0, 1].", file=sys.stderr)
        return 2
    if args.weight > 1.0 and not args.allow_full_weight:
        print("Refusing --weight above 1.0 without --allow-full-weight.", file=sys.stderr)
        return 2
    if args.left_distance < 0.0 and not args.use_vision:
        print("--left-distance must be non-negative when --use-vision is not set.", file=sys.stderr)
        return 2
    if not args.no_vertical_wrist:
        try:
            validate_wrist_target(G1JointIndex.RightWristRoll, args.wrist_roll_target)
            validate_wrist_target(G1JointIndex.RightWristPitch, args.wrist_pitch_target)
            validate_wrist_target(G1JointIndex.RightWristYaw, args.wrist_yaw_target)
        except ValueError as exc:
            print(f"Refusing wrist target: {exc}", file=sys.stderr)
            return 2
    if min(args.pre_duration, args.lateral_duration, args.return_duration, args.return_side_duration, args.release_duration) < 0.0:
        print("Durations must be non-negative.", file=sys.stderr)
        return 2
    hand_error = validate_hand_args(args)
    if hand_error:
        return hand_error
    if args.execute and not args.skip_hand and not hand_tcp_preflight(args):
        return 6

    pre_target = np.asarray(args.pregrasp_target, dtype=float)
    if args.use_vision:
        grasp_target = compute_vision_grasp_target(args, pre_target)
        if args.test_first_attempt_shortfall > 0.0:
            approach_direction = 1.0 if grasp_target[1] >= pre_target[1] else -1.0
            grasp_target[1] -= approach_direction * args.test_first_attempt_shortfall
            print("retry_test_mode: enabled")
            print(f"first_attempt_shortfall_m: {args.test_first_attempt_shortfall:.4f}")
            print("first_attempt_test_target:", np.array2string(grasp_target, precision=4))
        args.left_distance = float(grasp_target[1] - pre_target[1])
    else:
        grasp_target = pre_target.copy()
        grasp_target[1] += args.left_distance
    return_side_enabled = abs(args.return_side_offset) > 1e-6
    return_side_sign = -1.0 if pre_target[1] <= 0.0 else 1.0
    return_side_target = pre_target.copy()
    return_side_target[1] = pre_target[1] + return_side_sign * args.return_side_offset
    target_checks = [("pregrasp", pre_target), ("grasp", grasp_target)]
    if return_side_enabled:
        target_checks.append(("return_side", return_side_target))
    for label, target in target_checks:
        try:
            validate_target(target)
        except ValueError as exc:
            print(f"Refusing {label} target: {exc}", file=sys.stderr)
            return 2

    print("saved_pregrasp_target:", np.array2string(pre_target, precision=4))
    print("final_grasp_target:", np.array2string(grasp_target, precision=4))
    if args.approach_only:
        print("approach_only: true; hand close and feedback-close monitor are disabled.")
    print(f"computed_x_delta: {float(grasp_target[0] - pre_target[0]):.4f}")
    print(f"computed_left_distance: {args.left_distance:.4f}")

    network_interface = detect_interface(args.network_interface)
    print(f"network_interface: {network_interface}")
    controller = OfficialStyleArmSdkController(network_interface, control_dt=0.02)
    controller.init()
    state = controller.wait_for_low_state(5.0)

    right_joints = right_arm_joints(7)
    arm_dq_max = joint_velocity_abs_max(state, LEFT_ARM_7DOF + right_joints)
    waist_dq_max = joint_velocity_abs_max(state, WAIST_JOINTS)
    print(f"start_arm_dq_abs_max: {arm_dq_max:.4f}")
    print(f"start_waist_dq_abs_max: {waist_dq_max:.4f}")
    if arm_dq_max > args.max_start_arm_dq:
        print("Refusing: arms are already moving too fast.", file=sys.stderr)
        return 3
    if waist_dq_max > args.max_start_waist_dq:
        print("Refusing: waist is already moving too fast.", file=sys.stderr)
        return 3

    q_left_hold = joint_positions(state, LEFT_ARM_7DOF)
    q_waist_hold = joint_positions(state, WAIST_JOINTS)
    q_right_start = joint_positions(state, right_joints)
    q_start_values = joint_vector(state, right_joints)
    waist_q = joint_vector(state, WAIST_JOINTS)

    pre_ik, q_pre_values, q_pre = solve_right_q(pre_target, waist_q, right_joints, q_start_values, args.tolerance)
    if not args.no_vertical_wrist:
        q_pre_values, q_pre = apply_wrist_targets(
            q_pre_values,
            q_pre,
            right_joints,
            args.wrist_roll_target,
            args.wrist_pitch_target,
            args.wrist_yaw_target,
        )
    grasp_ik, q_grasp_values, q_grasp = solve_right_q(grasp_target, waist_q, right_joints, q_pre_values, args.tolerance)
    if not args.no_vertical_wrist:
        q_grasp_values, q_grasp = apply_wrist_targets(
            q_grasp_values,
            q_grasp,
            right_joints,
            args.wrist_roll_target,
            args.wrist_pitch_target,
            args.wrist_yaw_target,
        )
        print("wrist_override_enabled:")
        print(f"  RightWristRoll -> {args.wrist_roll_target:.4f} rad")
        print(f"  RightWristPitch -> {args.wrist_pitch_target:.4f} rad")
        print(f"  RightWristYaw -> {args.wrist_yaw_target:.4f} rad")
    else:
        print("wrist_override_enabled: false")

    grasp_command_xyz = command_position(q_grasp_values, waist_q, right_joints)
    verify_lift_target = grasp_command_xyz.copy()
    verify_lift_target[2] += args.verify_lift_distance
    verify_lift_ik = None
    q_verify_lift_values = q_grasp_values.copy()
    q_verify_lift = dict(q_grasp)
    if args.verify_grasp and not args.approach_only and not args.skip_hand:
        try:
            validate_target(verify_lift_target)
        except ValueError as exc:
            print(f"Refusing verification lift target: {exc}", file=sys.stderr)
            return 2
        if args.no_vertical_wrist:
            verify_lift_ik, q_verify_lift_values, q_verify_lift = solve_right_q(
                verify_lift_target, waist_q, right_joints, q_grasp_values, min(args.tolerance, 0.025)
            )
        else:
            verify_lift_ik, q_verify_lift_values, q_verify_lift = solve_right_q_fixed_wrist(
                verify_lift_target,
                waist_q,
                right_joints,
                q_grasp_values,
                min(args.tolerance, 0.025),
                args.wrist_roll_target,
                args.wrist_pitch_target,
                args.wrist_yaw_target,
            )

    return_side_ik = None
    q_return_side_values = q_pre_values.copy()
    q_return_side = dict(q_pre)
    if return_side_enabled:
        return_side_ik, q_return_side_values, q_return_side = solve_right_q(
            return_side_target, waist_q, right_joints, q_pre_values, args.tolerance
        )
        if not args.no_vertical_wrist:
            q_return_side_values, q_return_side = apply_wrist_targets(
                q_return_side_values,
                q_return_side,
                right_joints,
                args.wrist_roll_target,
                args.wrist_pitch_target,
                args.wrist_yaw_target,
            )
        if not return_side_ik.success:
            print(
                "return_side_ik_success: false; disabling side-return waypoint and falling back to original return path.",
                file=sys.stderr,
            )
            return_side_enabled = False
            q_return_side_values = q_pre_values.copy()
            q_return_side = dict(q_pre)

    clearance_enabled = not args.no_clearance_retreat
    clearance_vertical_target = grasp_command_xyz.copy()
    clearance_vertical_target[2] += args.clearance_lift_distance
    q_clearance_vertical_values = q_grasp_values.copy()
    q_clearance_vertical = dict(q_grasp)
    clearance_vertical_ik = None
    q_clearance_side_values = q_return_side_values.copy()
    q_clearance_side = dict(q_return_side)
    clearance_side_ik = None
    clearance_side_target = command_position(q_return_side_values, waist_q, right_joints)
    clearance_side_target[2] = clearance_vertical_target[2]
    if clearance_enabled:
        for label, target in (
            ("clearance_vertical", clearance_vertical_target),
            ("clearance_side", clearance_side_target),
        ):
            try:
                validate_target(target)
            except ValueError as exc:
                print(f"Refusing {label} target: {exc}", file=sys.stderr)
                return 2
        clearance_tolerance = min(args.tolerance, 0.015)
        if args.no_vertical_wrist:
            clearance_vertical_ik, q_clearance_vertical_values, q_clearance_vertical = solve_right_q(
                clearance_vertical_target,
                waist_q,
                right_joints,
                q_grasp_values,
                clearance_tolerance,
            )
            clearance_side_ik, q_clearance_side_values, q_clearance_side = solve_right_q(
                clearance_side_target,
                waist_q,
                right_joints,
                q_pre_values,
                clearance_tolerance,
            )
        else:
            clearance_vertical_ik, q_clearance_vertical_values, q_clearance_vertical = solve_right_q_fixed_wrist(
                clearance_vertical_target,
                waist_q,
                right_joints,
                q_grasp_values,
                clearance_tolerance,
                args.wrist_roll_target,
                args.wrist_pitch_target,
                args.wrist_yaw_target,
            )
            clearance_side_ik, q_clearance_side_values, q_clearance_side = solve_right_q_fixed_wrist(
                clearance_side_target,
                waist_q,
                right_joints,
                q_pre_values,
                clearance_tolerance,
                args.wrist_roll_target,
                args.wrist_pitch_target,
                args.wrist_yaw_target,
            )

    print_plan_item("pregrasp", pre_target, pre_ik, q_pre_values, q_start_values, right_joints)
    print_command_position("pregrasp", pre_target, q_pre_values, waist_q, right_joints)
    print_plan_item("grasp", grasp_target, grasp_ik, q_grasp_values, q_pre_values, right_joints)
    print_command_position("grasp", grasp_target, q_grasp_values, waist_q, right_joints)
    if verify_lift_ik is not None:
        print_plan_item(
            "verify_lift",
            verify_lift_target,
            verify_lift_ik,
            q_verify_lift_values,
            q_grasp_values,
            right_joints,
        )
        print_command_position("verify_lift", verify_lift_target, q_verify_lift_values, waist_q, right_joints)
    if clearance_enabled and clearance_vertical_ik is not None and clearance_side_ik is not None:
        print_plan_item(
            "clearance_vertical",
            clearance_vertical_target,
            clearance_vertical_ik,
            q_clearance_vertical_values,
            q_grasp_values,
            right_joints,
        )
        print_command_position(
            "clearance_vertical",
            clearance_vertical_target,
            q_clearance_vertical_values,
            waist_q,
            right_joints,
        )
        print_plan_item(
            "clearance_side",
            clearance_side_target,
            clearance_side_ik,
            q_clearance_side_values,
            q_pre_values,
            right_joints,
        )
        print_command_position(
            "clearance_side",
            clearance_side_target,
            q_clearance_side_values,
            waist_q,
            right_joints,
        )
    else:
        print("clearance_retreat_enabled: false")
    if return_side_enabled and return_side_ik is not None:
        print_plan_item("return_side", return_side_target, return_side_ik, q_return_side_values, q_pre_values, right_joints)
        print_command_position("return_side", return_side_target, q_return_side_values, waist_q, right_joints)
    else:
        print("return_side_enabled: false")
    print("left_hold:")
    print(format_joint_table(LEFT_ARM_7DOF, [q_left_hold[joint] for joint in LEFT_ARM_7DOF]))
    print("waist_hold:")
    print(format_joint_table(WAIST_JOINTS, [q_waist_hold[joint] for joint in WAIST_JOINTS]))

    if (
        not pre_ik.success
        or not grasp_ik.success
        or (verify_lift_ik is not None and not verify_lift_ik.success)
        or (clearance_vertical_ik is not None and not clearance_vertical_ik.success)
        or (clearance_side_ik is not None and not clearance_side_ik.success)
    ):
        print("Refusing execution because IK did not meet tolerance.", file=sys.stderr)
        return 4

    max_delta_pre = float(np.max(np.abs(q_pre_values - q_start_values)))
    max_delta_lateral = float(np.max(np.abs(q_grasp_values - q_pre_values)))
    max_delta_verify_lift = (
        float(np.max(np.abs(q_verify_lift_values - q_grasp_values))) if verify_lift_ik is not None else 0.0
    )
    max_delta_clearance_vertical = (
        float(np.max(np.abs(q_clearance_vertical_values - q_grasp_values))) if clearance_enabled else 0.0
    )
    max_delta_clearance_side = (
        float(np.max(np.abs(q_clearance_side_values - q_pre_values))) if clearance_enabled else 0.0
    )
    max_delta_return_side = float(np.max(np.abs(q_return_side_values - q_pre_values))) if return_side_enabled else 0.0
    max_delta_return_start = float(np.max(np.abs(q_start_values - q_return_side_values)))
    max_delta = max(
        max_delta_pre,
        max_delta_lateral,
        max_delta_verify_lift,
        max_delta_clearance_vertical,
        max_delta_clearance_side,
        max_delta_return_side,
        max_delta_return_start,
    )
    print(f"max_pre_delta_rad: {max_delta_pre:.4f}")
    print(f"max_lateral_delta_rad: {max_delta_lateral:.4f}")
    print(f"max_verify_lift_delta_rad: {max_delta_verify_lift:.4f}")
    print(f"max_clearance_vertical_delta_rad: {max_delta_clearance_vertical:.4f}")
    print(f"max_clearance_side_delta_rad: {max_delta_clearance_side:.4f}")
    print(f"max_return_side_delta_rad: {max_delta_return_side:.4f}")
    print(f"max_return_start_delta_rad: {max_delta_return_start:.4f}")
    if max_delta > args.max_joint_delta:
        print(f"Refusing execution: max joint delta {max_delta:.3f} exceeds {args.max_joint_delta:.3f}.", file=sys.stderr)
        return 5

    pre_duration = segment_duration(args.pre_duration, q_start_values, q_pre_values, args.max_joint_speed)
    lateral_duration = segment_duration(args.lateral_duration, q_pre_values, q_grasp_values, args.max_joint_speed)
    return_side_duration = (
        segment_duration(args.return_side_duration, q_pre_values, q_return_side_values, args.max_joint_speed)
        if return_side_enabled
        else 0.0
    )
    return_duration = segment_duration(args.return_duration, q_return_side_values, q_start_values, args.max_joint_speed)
    print(f"pre_duration_s: {pre_duration:.2f}")
    print(f"lateral_duration_s: {lateral_duration:.2f}")
    print(f"return_side_duration_s: {return_side_duration:.2f}")
    print(f"return_duration_s: {return_duration:.2f}")

    if not args.execute:
        print("Dry run only. Add --execute --yes to move the robot.")
        return 0

    announce_start(args)

    q_right_last = dict(q_right_start)
    exit_code = 0
    feedback_stop = None
    feedback_thread = None
    hand_closed_before_return = False
    grasp_verified = not args.verify_grasp
    verification_reason = "verification_disabled"
    unsafe_failure = None
    try:
        open_angles = with_thumb_rotation(args.hand_open_angle, args.thumb_rotate_open)
        close_angles = with_thumb_rotation(args.hand_grasp_angle, args.thumb_rotate_close)
        safe_close_angles = with_thumb_rotation(args.hand_safe_close_angle, None)
        thumb_outer_angles = thumb_outward_pose(open_angles, close_angles, args.thumb_stage_channel)
        thumb_inner_angles = thumb_inward_pose(open_angles, close_angles, args.thumb_stage_channel)
        print(f"hand_thumb_stage_channel: A{args.thumb_stage_channel}")
        print("hand_safe_close_angles:", fmt(safe_close_angles))
        print("hand_thumb_outer_angles:", fmt(thumb_outer_angles))
        print("hand_open_angles:", fmt(open_angles))
        print("hand_thumb_inner_angles:", fmt(thumb_inner_angles))
        print("hand_close_angles:", fmt(close_angles))

        if not args.skip_hand:
            print("Stage 0: close hand with thumb at maximum rotation before arm motion.")
            write_safe_close_once(args, safe_close_angles, "startup_safe_close")
            hand_closed_before_return = True

        print("Stage 1: move to saved pregrasp target.")
        run_timed_phase(controller, pre_duration, q_left_hold, q_waist_hold, q_right_start, q_pre, 0.0, args.weight, args)
        q_right_last = dict(q_pre)
        hold_group(controller, args.hold_before_open, q_left_hold, q_pre, q_waist_hold, args.weight, args)

        if not args.skip_hand:
            print("Stage 2: open fingers; thumb is already at maximum rotation.")
            write_hand_angles_once(args, open_angles, "pregrasp_open")
            hand_closed_before_return = False

        if not args.skip_hand and args.hold_after_open > 0.0:
            print("Stage 2 hold: hold pregrasp with hand open.")
            hold_group(controller, args.hold_after_open, q_left_hold, q_pre, q_waist_hold, args.weight, args)

        attempt_target = grasp_target.copy()
        attempt_q_values = q_grasp_values.copy()
        attempt_q = dict(q_grasp)
        attempt_duration = lateral_duration
        attempt_lift_target = verify_lift_target.copy()
        attempt_lift_q_values = q_verify_lift_values.copy()
        attempt_lift_q = dict(q_verify_lift)
        attempt_clearance_vertical_target = clearance_vertical_target.copy()
        attempt_clearance_vertical_q_values = q_clearance_vertical_values.copy()
        attempt_clearance_vertical_q = dict(q_clearance_vertical)
        attempt_clearance_side_target = clearance_side_target.copy()
        attempt_clearance_side_q_values = q_clearance_side_values.copy()
        attempt_clearance_side_q = dict(q_clearance_side)
        max_attempts = 1 + (args.max_grasp_retries if args.verify_grasp else 0)
        attempt_index = 0
        q_return_from = dict(q_pre)
        attempt_start_q = read_actual_right(controller, right_joints)

        while attempt_index < max_attempts:
            attempt_number = attempt_index + 1
            retry_allowed = False
            attempt_contact_detected = False
            print(f"Stage 3 attempt {attempt_number}/{max_attempts}: move whole hand to grasp target.")
            print("attempt_target:", np.array2string(attempt_target, precision=4))
            feedback_state = None
            q_stage3_last = dict(attempt_q)
            if not args.skip_hand and not args.no_force_feedback_grasp and not args.approach_only:
                print("Stage 3 feedback-limited approach: force/current feedback stops arm advance, then closes hand.")
                q_stage3_last, feedback_state = run_feedback_limited_phase(
                    controller,
                    attempt_duration,
                    q_left_hold,
                    q_waist_hold,
                    attempt_start_q,
                    attempt_q,
                    args.weight,
                    args.weight,
                    args,
                    thumb_inner_angles,
                    close_angles,
                )
            else:
                run_timed_phase(
                    controller,
                    attempt_duration,
                    q_left_hold,
                    q_waist_hold,
                    attempt_start_q,
                    attempt_q,
                    args.weight,
                    args.weight,
                    args,
                )
                q_stage3_last = dict(attempt_q)
            q_right_last = dict(q_stage3_last)
            q_return_from = dict(q_stage3_last)
            hold_group(controller, args.hold_before_grasp, q_left_hold, q_stage3_last, q_waist_hold, args.weight, args)

            if args.approach_only:
                print("Stage 4: approach-only mode; closing hand with thumb at maximum rotation before return.")
                if not args.skip_hand:
                    write_safe_close_once(args, safe_close_angles, "approach_only_return_safe_close")
                    hand_closed_before_return = True
                break
            if not args.skip_hand:
                if args.no_force_feedback_grasp:
                    print("Stage 4: grasp with original one-shot hand command.")
                    write_hand_pose(hand_args(args), close_angles, "grasp")
                    hand_closed_before_return = True
                else:
                    if feedback_stop is not None and feedback_thread is not None:
                        stop_feedback_monitor(feedback_stop, feedback_thread)
                    if feedback_state and feedback_state.get("error"):
                        raise RuntimeError(f"hand feedback monitor failed: {feedback_state['error']}")
                    print(f"Stage 4: feedback_triggered={bool(feedback_state and feedback_state.get('triggered'))}")
                    print(f"Stage 4: feedback_thumb_rotated={bool(feedback_state and feedback_state.get('thumb_rotated'))}")
                    print(f"Stage 4: feedback_closed={bool(feedback_state and feedback_state.get('closed'))}")
                    print(f"Stage 4: feedback_reason={feedback_state.get('reason') if feedback_state else 'disabled'}")
                    attempt_contact_detected = bool(
                        feedback_state
                        and feedback_state.get("triggered")
                        and feedback_state.get("reason") not in ("approach_end_fallback", "feedback_timeout", "not_triggered")
                    )
                    print(f"Stage 4: physical_contact_detected={attempt_contact_detected}")
                    hand_closed_before_return = bool(feedback_state and feedback_state.get("closed"))
                    if not hand_closed_before_return:
                        if args.close_on_feedback_timeout:
                            print("Stage 4: no feedback-triggered close; timeout policy rotates thumb inward, then closes fingers.")
                            write_thumb_then_close_once(args, thumb_inner_angles, close_angles, "timeout")
                            hand_closed_before_return = True
                            monitor_hand_feedback_window(args, "close", min(max(args.hand_hold, 0.5), 2.0))
                        else:
                            print("Stage 4: no feedback-triggered close; hand close skipped.")
                hold_group(controller, args.hold_after_grasp, q_left_hold, q_stage3_last, q_waist_hold, args.weight, args)

            if not args.verify_grasp or args.skip_hand:
                break

            if not hand_closed_before_return:
                verification_reason = "hand_did_not_close"
                print(f"grasp_verification_passed: false reason={verification_reason}")
            else:
                before_lift = read_hand_feedback_once(args, f"attempt_{attempt_number}_before_lift")
                before_classification = classify_finger_obstruction(
                    args,
                    before_lift,
                    close_angles,
                    f"attempt_{attempt_number}_before_lift",
                )
                if before_classification["passed"]:
                    print(
                        f"Stage 4 verification: lift {args.verify_lift_distance:.3f} m "
                        "using measured joint state."
                    )
                    q_stage3_last = run_tracked_segment(
                        controller,
                        args,
                        q_left_hold,
                        q_waist_hold,
                        right_joints,
                        attempt_lift_q,
                        args.verify_lift_duration,
                        f"attempt_{attempt_number}_verify_lift",
                    )
                    q_right_last = dict(q_stage3_last)
                    q_return_from = dict(q_stage3_last)
                    hold_group(
                        controller,
                        args.verify_hold,
                        q_left_hold,
                        q_stage3_last,
                        q_waist_hold,
                        args.weight,
                        args,
                    )
                    after_lift = read_hand_feedback_once(args, f"attempt_{attempt_number}_after_lift")
                    after_classification = classify_finger_obstruction(
                        args,
                        after_lift,
                        close_angles,
                        f"attempt_{attempt_number}_after_lift",
                    )
                    grasp_verified, verification_reason = evaluate_lift_retention(
                        args,
                        before_lift,
                        after_lift,
                        before_classification,
                        after_classification,
                    )
                    print(
                        f"grasp_verification_passed: {str(grasp_verified).lower()} "
                        f"reason={verification_reason}"
                    )
                    if grasp_verified:
                        break
                    retry_allowed = (
                        classification_is_confidently_empty(args, after_classification)
                        and not attempt_contact_detected
                    )
                    print("Verification failed after lift; keep the lifted pose and do not descend toward the table.")
                else:
                    verification_reason = "insufficient_finger_obstruction_before_lift"
                    retry_allowed = (
                        classification_is_confidently_empty(args, before_classification)
                        and not attempt_contact_detected
                    )
                    print(f"grasp_verification_passed: false reason={verification_reason}")

            if attempt_number >= max_attempts:
                break

            if not retry_allowed:
                verification_reason = f"{verification_reason}; retry_suppressed_ambiguous_grasp"
                print(
                    "Retry suppressed: the hand is not confidently empty, so it may already contain the bottle. "
                    "Keep the current hand command and retreat.",
                    file=sys.stderr,
                )
                break

            print("Retry preparation: reverse the approach path to pregrasp; no downward reset motion is allowed.")
            q_right_last = run_tracked_segment(
                controller,
                args,
                q_left_hold,
                q_waist_hold,
                right_joints,
                q_pre,
                args.lateral_duration,
                f"retry_{attempt_number}_retrace",
            )
            q_return_from = dict(q_right_last)
            attempt_start_q = dict(q_right_last)
            write_thumb_then_open_once(args, thumb_outer_angles, open_angles, f"retry_{attempt_number}_open")
            hand_closed_before_return = False

            try:
                if args.use_vision:
                    retry_target = compute_vision_target_while_holding(
                        args,
                        pre_target,
                        controller,
                        q_left_hold,
                        attempt_start_q,
                        q_waist_hold,
                    )
                else:
                    retry_target = attempt_target.copy()
                approach_direction = 1.0 if retry_target[1] >= pre_target[1] else -1.0
                retry_target[1] += approach_direction * args.retry_approach_extra
                print("retry_redetected_target:", np.array2string(retry_target, precision=4))
                print(f"retry_approach_extra_m: {args.retry_approach_extra:.4f}")
                validate_target(retry_target)
                retry_ik, retry_q_values, retry_q = solve_right_q(
                    retry_target,
                    waist_q,
                    right_joints,
                    q_pre_values,
                    args.tolerance,
                )
                if not args.no_vertical_wrist:
                    retry_q_values, retry_q = apply_wrist_targets(
                        retry_q_values,
                        retry_q,
                        right_joints,
                        args.wrist_roll_target,
                        args.wrist_pitch_target,
                        args.wrist_yaw_target,
                    )
                print_plan_item("retry_grasp", retry_target, retry_ik, retry_q_values, q_pre_values, right_joints)
                print_command_position("retry_grasp", retry_target, retry_q_values, waist_q, right_joints)
                if not retry_ik.success:
                    raise RuntimeError("retry grasp IK did not meet tolerance")
                retry_delta = float(np.max(np.abs(retry_q_values - q_pre_values)))
                if retry_delta > args.max_joint_delta:
                    raise RuntimeError(
                        f"retry grasp joint delta {retry_delta:.3f} exceeds {args.max_joint_delta:.3f}"
                    )

                retry_command_xyz = command_position(retry_q_values, waist_q, right_joints)
                retry_lift_target = retry_command_xyz.copy()
                retry_lift_target[2] += args.verify_lift_distance
                validate_target(retry_lift_target)
                if args.no_vertical_wrist:
                    retry_lift_ik, retry_lift_q_values, retry_lift_q = solve_right_q(
                        retry_lift_target,
                        waist_q,
                        right_joints,
                        retry_q_values,
                        min(args.tolerance, 0.025),
                    )
                else:
                    retry_lift_ik, retry_lift_q_values, retry_lift_q = solve_right_q_fixed_wrist(
                        retry_lift_target,
                        waist_q,
                        right_joints,
                        retry_q_values,
                        min(args.tolerance, 0.025),
                        args.wrist_roll_target,
                        args.wrist_pitch_target,
                        args.wrist_yaw_target,
                    )
                print_plan_item(
                    "retry_verify_lift",
                    retry_lift_target,
                    retry_lift_ik,
                    retry_lift_q_values,
                    retry_q_values,
                    right_joints,
                )
                if not retry_lift_ik.success:
                    raise RuntimeError("retry verification lift IK did not meet tolerance")
                retry_lift_delta = float(np.max(np.abs(retry_lift_q_values - retry_q_values)))
                if retry_lift_delta > args.max_joint_delta:
                    raise RuntimeError(
                        f"retry lift joint delta {retry_lift_delta:.3f} exceeds {args.max_joint_delta:.3f}"
                    )

                retry_clearance_vertical_target = retry_command_xyz.copy()
                retry_clearance_vertical_target[2] += args.clearance_lift_distance
                retry_clearance_side_target = command_position(q_return_side_values, waist_q, right_joints)
                retry_clearance_side_target[2] = retry_clearance_vertical_target[2]
                for label, target in (
                    ("retry_clearance_vertical", retry_clearance_vertical_target),
                    ("retry_clearance_side", retry_clearance_side_target),
                ):
                    validate_target(target)
                clearance_tolerance = min(args.tolerance, 0.015)
                if args.no_vertical_wrist:
                    retry_clearance_vertical_ik, retry_clearance_vertical_q_values, retry_clearance_vertical_q = solve_right_q(
                        retry_clearance_vertical_target,
                        waist_q,
                        right_joints,
                        retry_q_values,
                        clearance_tolerance,
                    )
                    retry_clearance_side_ik, retry_clearance_side_q_values, retry_clearance_side_q = solve_right_q(
                        retry_clearance_side_target,
                        waist_q,
                        right_joints,
                        q_pre_values,
                        clearance_tolerance,
                    )
                else:
                    retry_clearance_vertical_ik, retry_clearance_vertical_q_values, retry_clearance_vertical_q = solve_right_q_fixed_wrist(
                        retry_clearance_vertical_target,
                        waist_q,
                        right_joints,
                        retry_q_values,
                        clearance_tolerance,
                        args.wrist_roll_target,
                        args.wrist_pitch_target,
                        args.wrist_yaw_target,
                    )
                    retry_clearance_side_ik, retry_clearance_side_q_values, retry_clearance_side_q = solve_right_q_fixed_wrist(
                        retry_clearance_side_target,
                        waist_q,
                        right_joints,
                        q_pre_values,
                        clearance_tolerance,
                        args.wrist_roll_target,
                        args.wrist_pitch_target,
                        args.wrist_yaw_target,
                    )
                if not retry_clearance_vertical_ik.success or not retry_clearance_side_ik.success:
                    raise RuntimeError("retry clearance IK did not meet tolerance")
                print_plan_item(
                    "retry_clearance_vertical",
                    retry_clearance_vertical_target,
                    retry_clearance_vertical_ik,
                    retry_clearance_vertical_q_values,
                    retry_q_values,
                    right_joints,
                )
                print_plan_item(
                    "retry_clearance_side",
                    retry_clearance_side_target,
                    retry_clearance_side_ik,
                    retry_clearance_side_q_values,
                    q_pre_values,
                    right_joints,
                )

                attempt_target = retry_target
                attempt_q_values = retry_q_values
                attempt_q = retry_q
                attempt_duration = segment_duration(
                    args.lateral_duration,
                    q_pre_values,
                    retry_q_values,
                    args.max_joint_speed,
                )
                attempt_lift_target = retry_lift_target
                attempt_lift_q_values = retry_lift_q_values
                attempt_lift_q = retry_lift_q
                attempt_clearance_vertical_target = retry_clearance_vertical_target
                attempt_clearance_vertical_q_values = retry_clearance_vertical_q_values
                attempt_clearance_vertical_q = retry_clearance_vertical_q
                attempt_clearance_side_target = retry_clearance_side_target
                attempt_clearance_side_q_values = retry_clearance_side_q_values
                attempt_clearance_side_q = retry_clearance_side_q
                attempt_index += 1
            except Exception as exc:
                verification_reason = f"retry_planning_failed: {exc}"
                print(verification_reason, file=sys.stderr)
                break

        if not args.skip_hand and not hand_closed_before_return:
            print("Stage 4 return prep: close hand with thumb at maximum rotation before moving back.")
            write_safe_close_once(args, safe_close_angles, "return_prep_safe_close")
            hand_closed_before_return = True

        if args.verify_grasp and not grasp_verified and not args.skip_hand:
            print(
                "Verification was inconclusive: preserve the current grasp command during retreat; "
                "do not rotate the thumb outward or reopen the hand."
            )

        if clearance_enabled:
            print("Stage 5: execute high table-clearance retreat.")
            q_right_last = run_clearance_retreat(
                controller,
                args,
                q_left_hold,
                q_waist_hold,
                right_joints,
                q_return_from,
                q_pre,
                q_pre_values,
                attempt_clearance_side_q,
                attempt_clearance_side_q_values,
                q_return_side,
                q_return_side_values,
                "final_clearance",
            )
            print("Stage 6: retract arm to start posture after clearing the table.")
            q_right_last = run_tracked_segment(
                controller,
                args,
                q_left_hold,
                q_waist_hold,
                right_joints,
                q_right_start,
                return_duration,
                "final_retract",
            )
        else:
            print("Stage 5: move back to saved pregrasp target.")
            q_return_values = np.asarray([q_return_from[joint] for joint in right_joints], dtype=float)
            final_retreat_duration = segment_duration(
                args.lateral_duration,
                q_return_values,
                q_pre_values,
                args.max_joint_speed,
            )
            run_timed_phase(
                controller,
                final_retreat_duration,
                q_left_hold,
                q_waist_hold,
                q_return_from,
                q_pre,
                args.weight,
                args.weight,
                args,
            )
            q_right_last = dict(q_pre)
            if return_side_enabled:
                print("Stage 6: move laterally farther outside the table before retracting.")
                run_timed_phase(
                    controller,
                    return_side_duration,
                    q_left_hold,
                    q_waist_hold,
                    q_pre,
                    q_return_side,
                    args.weight,
                    args.weight,
                    args,
                )
                q_right_last = dict(q_return_side)
                print("Stage 7: retract arm to start posture.")
                run_timed_phase(
                    controller,
                    return_duration,
                    q_left_hold,
                    q_waist_hold,
                    q_return_side,
                    q_right_start,
                    args.weight,
                    args.weight,
                    args,
                )
            else:
                print("Stage 6: retract arm to start posture.")
                run_timed_phase(
                    controller,
                    return_duration,
                    q_left_hold,
                    q_waist_hold,
                    q_pre,
                    q_right_start,
                    args.weight,
                    args.weight,
                    args,
                )
        q_right_last = dict(q_right_start)
    except MotionTrackingError as exc:
        q_right_last = dict(exc.q_actual)
        unsafe_failure = str(exc)
        exit_code = 8
        print(f"Motion safety stop: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        unsafe_failure = "motion interrupted by operator before a verified safe return"
        try:
            q_right_last = read_actual_right(controller, right_joints)
        except Exception:
            pass
        print("Interrupted by user; entering safety hold.", file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        unsafe_failure = f"unexpected execution failure: {exc}"
        try:
            q_right_last = read_actual_right(controller, right_joints)
        except Exception:
            pass
        print(f"Execution failed; entering safety hold: {exc}", file=sys.stderr)
        exit_code = 9
    finally:
        if feedback_stop is not None and feedback_thread is not None and feedback_thread.is_alive():
            stop_feedback_monitor(feedback_stop, feedback_thread)
        if unsafe_failure is not None:
            q_right_last = operator_safety_hold(
                controller,
                args,
                q_left_hold,
                q_waist_hold,
                right_joints,
                q_right_last,
                unsafe_failure,
            )
        print("Stage 8: release arm_sdk.")
        release_group(controller, q_left_hold, q_right_last, q_waist_hold, args.weight, args)
        print("arm_sdk released.")

    if (
        exit_code == 0
        and args.verify_grasp
        and not args.approach_only
        and not args.skip_hand
        and not grasp_verified
    ):
        exit_code = 7
        print(f"grasp_result: failed verification ({verification_reason})", file=sys.stderr)
    elif exit_code == 0 and args.verify_grasp and grasp_verified:
        print(f"grasp_result: verified ({verification_reason})")

    if exit_code == 0:
        announce_success(args)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
