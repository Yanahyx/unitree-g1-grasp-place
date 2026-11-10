#!/usr/bin/env python3
"""V2 visual lateral grasp sequence with board-referenced grasp height.

Sequence:
1. Close RH56E2 hand before lifting.
2. Move right hand to PREGRASP_TARGET in pelvis frame.
3. Open RH56E2 hand and rotate the thumb outward.
4. Move the whole right hand to the final grasp target.
5. On force/current feedback, rotate the thumb inward, then close the fingers.
6. Move back to PREGRASP_TARGET.
7. Return the right arm to its start posture and release arm_sdk.

This V2 is an isolated copy. The original entry point remains unchanged.
"""

import argparse
import os
import io
import json
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
    right_hand_fk,
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
    optical_to_pelvis_rotation,
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
# Legacy visual grasp posture: rotate the thumb inward to the original A5=300
# target before the fingers close. The separate safe-close travel posture uses
# A5=1000.
DEFAULT_HAND_GRASP_ANGLE = [0, 0, 0, 0, 400, 300]
DEFAULT_HAND_SAFE_CLOSE_ANGLE = [0, 0, 0, 0, 400, 1000]
DEFAULT_WRIST_ROLL = 0.30
DEFAULT_WRIST_PITCH = -0.80
DEFAULT_WRIST_YAW = 0.50
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
    parser.add_argument(
        "--vision-height-fraction",
        type=float,
        default=0.55,
        help="Grasp height as a fraction of object height measured up from the fitted tabletop plane.",
    )
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
    parser.add_argument(
        "--move-to-pregrasp",
        action="store_true",
        help=(
            "Move the arm from its current pose to the saved pregrasp target before opening the hand. "
            "Default leaves the arm at its current pose."
        ),
    )
    parser.add_argument(
        "--move-during-grasp",
        action="store_true",
        help=(
            "Legacy mode: move to pregrasp and then move the arm toward the final grasp target while waiting for feedback. "
            "Default keeps the arm stationary at its current pose."
        ),
    )
    parser.add_argument(
        "--return-after-grasp",
        action="store_true",
        help=(
            "Legacy mode: retract to pregrasp and then the start posture after closing. "
            "Default leaves the arm at the grasp posture for the next place command."
        ),
    )
    parser.add_argument("--weight", type=float, default=0.75)
    parser.add_argument("--allow-full-weight", action="store_true")
    parser.add_argument("--pre-duration", type=float, default=6.0)
    parser.add_argument("--lateral-duration", type=float, default=3.0)
    parser.add_argument("--return-duration", type=float, default=5.0)
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
    parser.add_argument(
        "--feedback-start-delay",
        type=float,
        default=5.0,
        help="Seconds to wait after the empty-hand baseline before monitoring contact feedback.",
    )
    parser.add_argument(
        "--hand-contact-force-delta",
        type=int,
        default=120,
        help="Hand-only contact threshold on finger forceAct channels A0..A3.",
    )
    parser.add_argument(
        "--hand-contact-current-delta",
        type=int,
        default=80,
        help="Hand-only corroborating current threshold on the same finger channel.",
    )
    parser.add_argument(
        "--feedback-confirm-polls",
        type=int,
        default=4,
        help="Require this many consecutive contact polls before closing the hand.",
    )
    parser.add_argument("--feedback-poll", type=float, default=0.05, help="Seconds between hand feedback reads.")
    parser.add_argument(
        "--thumb-inward-delay",
        type=float,
        default=0.45,
        help="Seconds to wait after rotating the thumb inward before closing the fingers.",
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


def force_feedback_grasp(args: argparse.Namespace) -> bool:
    open_angles = with_thumb_rotation(args.hand_open_angle, args.thumb_rotate_open)
    close_angles = with_thumb_rotation(args.hand_grasp_angle, args.thumb_rotate_close)
    thumb_outer_angles = thumb_outward_pose(open_angles, close_angles, args.thumb_stage_channel)
    thumb_inner_angles = thumb_inward_pose(open_angles, close_angles, args.thumb_stage_channel)
    print("force_feedback_grasp_thumb_outer_angles:", fmt(thumb_outer_angles))
    print("force_feedback_grasp_open_angles:", fmt(open_angles))
    print("force_feedback_grasp_thumb_inner_angles:", fmt(thumb_inner_angles))
    print("force_feedback_grasp_close_angles:", fmt(close_angles))

    closed = False
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print(f"hand_connect: {args.hand_ip}:{args.hand_port}")
        print("TCP connected")
        angle_before_outer = read_hand_angle_act(client)
        start_thumb_rotation = int(angle_before_outer[args.thumb_stage_channel])
        print(f"hand_angleAct_before_thumb_outer: {fmt(angle_before_outer)}")
        write_staged_thumb_rotation_client(
            client,
            args,
            thumb_outer_angles,
            start_thumb_rotation,
            open_angles[args.thumb_stage_channel],
            "thumb_outer",
        )
        time.sleep(max(0.0, args.thumb_outward_delay))
        write_hand_angles_open_client(client, args, open_angles, "open")
        time.sleep(max(0.0, args.hand_hold))

        baseline = hand_feedback_snapshot(client)
        print(f"feedback_baseline_forceAct_signed[6]: {fmt(baseline['force'].tolist())}")
        print(f"feedback_baseline_current[6]: {fmt(baseline['current'].tolist())}")

        if args.feedback_start_delay > 0.0:
            print(f"feedback_start_delay_s: {args.feedback_start_delay:.1f}")
            time.sleep(args.feedback_start_delay)

        triggered = False
        trigger_reason = "timeout"
        trigger_snapshot = baseline
        consecutive = 0
        deadline = (
            time.monotonic() + args.feedback_timeout
            if args.feedback_timeout > 0.0
            else None
        )
        poll = max(0.01, args.feedback_poll)
        while deadline is None or time.monotonic() < deadline:
            snapshot = hand_feedback_snapshot(client)
            reason = hand_contact_feedback_reason(snapshot, baseline, args)
            if reason is not None:
                consecutive += 1
                if consecutive >= max(1, args.feedback_confirm_polls):
                    triggered = True
                    trigger_reason = reason
                    trigger_snapshot = snapshot
                    break
            else:
                consecutive = 0
            time.sleep(poll)

        print(f"feedback_triggered: {triggered}")
        print(f"feedback_trigger_reason: {trigger_reason}")
        print(f"feedback_trigger_forceAct_signed[6]: {fmt(trigger_snapshot['force'].tolist())}")
        print(f"feedback_trigger_current[6]: {fmt(trigger_snapshot['current'].tolist())}")
        print(f"feedback_trigger_angleAct[6]: {fmt(trigger_snapshot['angle_act'].tolist())}")
        print_feedback_delta_detail("feedback_trigger", trigger_snapshot, baseline)

        if triggered or args.close_on_feedback_timeout:
            write_thumb_inward_then_fingers_close_client(client, args, thumb_inner_angles, close_angles, "feedback_grasp")
            closed = True
            final = hand_feedback_snapshot(client)
            print(f"feedback_final_forceAct_signed[6]: {fmt(final['force'].tolist())}")
            print(f"feedback_final_current[6]: {fmt(final['current'].tolist())}")
            print(f"feedback_final_angleAct[6]: {fmt(final['angle_act'].tolist())}")
        else:
            print("feedback_close_skipped: no feedback before timeout")
    return closed


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
    write_hand_angles_open_client(client, args, close_angles, f"{label}_fingers_close")
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


def start_arm_hold_thread(
    controller: OfficialStyleArmSdkController,
    q_left_hold: Dict[int, float],
    q_right_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
    weight: float,
    args: argparse.Namespace,
) -> Tuple[threading.Event, threading.Thread, Dict[str, object]]:
    """Keep the arm at one pose while the separate hand waits for contact."""

    stop_event = threading.Event()
    state: Dict[str, object] = {"error": None}

    def worker() -> None:
        next_tick = time.monotonic()
        try:
            while not stop_event.is_set():
                controller.write_group_command(
                    q_left=q_left_hold,
                    q_right=q_right_hold,
                    q_waist=q_waist_hold,
                    weight=weight,
                    kp_right=args.kp_right,
                    kd_right=args.kd_right,
                    kp_left=args.kp_left,
                    kd_left=args.kd_left,
                    kp_waist=args.kp_waist,
                    kd_waist=args.kd_waist,
                )
                next_tick += controller.control_dt
                stop_event.wait(max(0.0, next_tick - time.monotonic()))
        except Exception as exc:
            state["error"] = str(exc)

    thread = threading.Thread(target=worker, name="static_grasp_arm_hold", daemon=True)
    thread.start()
    return stop_event, thread, state


def stop_arm_hold_thread(
    stop_event: threading.Event,
    thread: threading.Thread,
    state: Dict[str, object],
) -> None:
    stop_event.set()
    thread.join(timeout=2.0)
    if thread.is_alive():
        raise RuntimeError("static grasp arm-hold thread did not stop")
    if state.get("error"):
        raise RuntimeError(f"static grasp arm-hold failed: {state['error']}")


def feedback_trigger_reason(snapshot: Dict[str, np.ndarray], baseline: Dict[str, np.ndarray], args: argparse.Namespace) -> Optional[str]:
    force_delta, current_delta = feedback_deltas(snapshot, baseline)
    # A5 is the thumb rotation motor and can report force while it is moving.
    # Only the four finger channels may stop the approach.
    finger_force = force_delta[:4]
    finger_current = current_delta[:4]
    force_index, max_force_delta = max_delta_channel(finger_force)
    current_index, max_current_delta = max_delta_channel(finger_current)
    if max_force_delta >= args.feedback_force_delta:
        return f"force_delta={max_force_delta} channel={hand_channel_label(force_index)} index={force_index}"
    if max_current_delta >= args.feedback_current_delta:
        return f"current_delta={max_current_delta} channel={hand_channel_label(current_index)} index={current_index}"
    return None


def hand_contact_feedback_reason(
    snapshot: Dict[str, np.ndarray],
    baseline: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Optional[str]:
    """Use contact evidence suitable for the hand-only waiting phase.

    The full arm sequence keeps the original force-or-current trigger.  During
    a hand-only wait, however, the current register often changes when an
    empty hand finishes settling, so a current-only event is not sufficient to
    close.  Prefer forceAct on the four finger channels and accept current only
    as corroboration on that same finger.
    """
    force_delta, current_delta = feedback_deltas(snapshot, baseline)
    finger_force = force_delta[:4]
    finger_current = current_delta[:4]
    force_index = int(np.argmax(finger_force))
    max_force = int(finger_force[force_index])
    force_threshold = max(1, int(getattr(args, "hand_contact_force_delta", 120)))
    if max_force >= force_threshold:
        return (
            f"force_delta={max_force} channel={hand_channel_label(force_index)} "
            f"index={force_index}"
        )

    current_index = int(np.argmax(finger_current))
    max_current = int(finger_current[current_index])
    current_threshold = max(1, int(getattr(args, "hand_contact_current_delta", 80)))
    corroborating_force = int(finger_force[current_index])
    if max_current >= current_threshold and corroborating_force >= force_threshold // 2:
        return (
            f"current_delta={max_current} channel={hand_channel_label(current_index)} "
            f"index={current_index} force_delta={corroborating_force}"
        )
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
                        finger_contacts = int(np.count_nonzero(force_delta[:4] >= args.feedback_force_delta))
                        state["finger_contact_channels"] = finger_contacts
                        if finger_contacts < 2:
                            state["reason"] = f"single_finger_contact_rejected:{reason}"
                            print(
                                "feedback_limited_close_skipped: "
                                f"only {finger_contacts} finger channel(s) showed force contact; arm stopped for safe return."
                            )
                            return q_right_last, state
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


def solve_grasp_waypoint(
    target: np.ndarray,
    waist_q: np.ndarray,
    right_joints: Sequence[int],
    seed_motor_q: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[object, np.ndarray, Dict[int, float]]:
    """Solve arm position with the tested wrist posture held inside IK."""

    if args.no_vertical_wrist:
        return solve_right_q(target, waist_q, right_joints, seed_motor_q, args.tolerance)

    active_joints = list(right_joints[:4])
    fixed_wrist = np.asarray(
        [args.wrist_roll_target, args.wrist_pitch_target, args.wrist_yaw_target], dtype=float
    )

    def fixed_wrist_fk(active_q, current_waist_q, _active_joint_indices):
        full_q = np.concatenate((np.asarray(active_q, dtype=float), fixed_wrist))
        return right_hand_fk(full_q, current_waist_q, right_joints)

    seed_model_q = motor_to_model_q(np.asarray(seed_motor_q, dtype=float), np.ones(len(right_joints)))
    ik = solve_position_ik_for(
        target=target,
        waist_q=waist_q,
        joint_indices=active_joints,
        signs=np.ones(len(active_joints)),
        seed=seed_model_q[: len(active_joints)],
        tolerance=args.tolerance,
        fk_func=fixed_wrist_fk,
    )
    q_values = np.concatenate((ik.q_model, fixed_wrist))
    q_targets = {joint: float(value) for joint, value in zip(right_joints, q_values)}
    return ik, q_values, q_targets


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
    model_q = motor_to_model_q(q_values, np.ones(7))
    pose = right_hand_pose(model_q, waist_q, right_joints)
    reached = pose[:3, 3]
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
    validate_detection_bbox(det, capture_info["color_bgr"].shape)
    point_camera, depth_stats = depth_point_for_detection(capture_info, det)
    depth_stats = {
        **depth_stats,
        "fallback_source": "realsense",
        "fallback_reason": fallback_reason,
        "capture_source": capture_info.get("source", "realsense"),
        "_capture_info": capture_info,
    }
    return model_path, point_camera, det, depth_stats


def validate_detection_bbox(det: Dict[str, object], image_shape: Sequence[int]) -> None:
    """Reject clipped or implausibly broad boxes before any arm plan is built."""

    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [float(value) for value in det["bbox_xyxy"]]
    box_width = x2 - x1
    box_height = y2 - y1
    area_fraction = box_width * box_height / max(width * height, 1)
    if min(box_width, box_height) <= 0.0:
        raise RuntimeError(f"invalid vision detection box: {det!r}")
    if x1 <= 2.0 or y1 <= 2.0 or x2 >= width - 2.0 or y2 >= height - 2.0:
        raise RuntimeError(
            f"vision detection is clipped by the image edge; refusing arm motion: bbox={det['bbox_xyxy']}"
        )
    if area_fraction > 0.25:
        raise RuntimeError(
            f"vision detection covers {area_fraction:.1%} of the image; refusing arm motion: bbox={det['bbox_xyxy']}"
        )
    if box_width < 18.0 or box_height < 30.0:
        raise RuntimeError(
            f"vision detection is too small for stable grasp targeting: bbox={det['bbox_xyxy']}"
        )


def estimate_board_referenced_object_center(
    capture: Dict[str, object],
    det: Dict[str, object],
    camera_pitch: float,
    height_fraction: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Estimate object height from a local plane fit beneath its detector box.

    The strip immediately below the detected object is expected to contain the
    calibration board/table surface. The object base is the ray-plane
    intersection at the bottom-center of the detector box; the top is measured
    from RGB-D just inside the top of the object.
    """

    if not 0.35 <= float(height_fraction) <= 0.75:
        raise ValueError("--vision-height-fraction must be between 0.35 and 0.75")

    depth_raw = np.asarray(capture["depth_raw"])
    if depth_raw.ndim != 2:
        raise RuntimeError(f"expected a 2-D aligned depth image, got {depth_raw.shape}")
    depth_scale = float(capture["depth_scale"])
    fx, fy = float(capture["fx"]), float(capture["fy"])
    cx, cy = float(capture["cx"]), float(capture["cy"])
    if depth_scale <= 0.0 or min(fx, fy) <= 0.0:
        raise RuntimeError("invalid depth scale or RGB-D intrinsics")

    image_h, image_w = depth_raw.shape
    x1, y1, x2, y2 = [float(value) for value in det["bbox_xyxy"]]
    box_w, box_h = x2 - x1, y2 - y1
    roi_x0 = max(0, int(np.floor(x1 - 0.25 * box_w)))
    roi_x1 = min(image_w, int(np.ceil(x2 + 0.30 * box_w)))
    roi_y0 = max(0, int(np.floor(y2 + 3.0)))
    roi_y1 = min(image_h, int(np.ceil(y2 + 0.48 * box_h)))
    if roi_x1 - roi_x0 < 20 or roi_y1 - roi_y0 < 12:
        raise RuntimeError("not enough visible calibration-board/table area below the object")

    local_depth = depth_raw[roi_y0:roi_y1:2, roi_x0:roi_x1:2].astype(np.float64) * depth_scale
    local_v, local_u = np.indices(local_depth.shape)
    valid = np.isfinite(local_depth) & (local_depth > 0.10) & (local_depth < 2.0)
    z_values = local_depth[valid]
    u_values = local_u[valid] * 2 + roi_x0
    v_values = local_v[valid] * 2 + roi_y0
    if z_values.size < 200:
        raise RuntimeError(f"too few valid board depth samples below object: {z_values.size}")

    plane_points = np.column_stack(
        (
            (u_values - cx) * z_values / fx,
            (v_values - cy) * z_values / fy,
            z_values,
        )
    )
    inliers = np.ones(len(plane_points), dtype=bool)
    for _ in range(3):
        points = plane_points[inliers]
        center = np.median(points, axis=0)
        _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
        normal_camera = vh[-1]
        residuals = (plane_points - center) @ normal_camera
        median_residual = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median_residual)))
        threshold = max(0.006, 3.0 * 1.4826 * mad)
        inliers = np.abs(residuals - median_residual) <= threshold
    points = plane_points[inliers]
    if len(points) < 200:
        raise RuntimeError(f"board plane fit retained too few samples: {len(points)}")
    plane_origin_camera = np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(points - plane_origin_camera, full_matrices=False)
    normal_camera = vh[-1]
    residuals = (points - plane_origin_camera) @ normal_camera
    plane_rms = float(np.sqrt(np.mean(residuals * residuals)))
    if plane_rms > 0.010:
        raise RuntimeError(f"calibration-board/table plane is noisy: RMS={plane_rms:.4f} m")

    rotation = optical_to_pelvis_rotation(camera_pitch)
    normal_pelvis = rotation @ normal_camera
    if normal_pelvis[2] < 0.0:
        normal_camera = -normal_camera
        normal_pelvis = -normal_pelvis
    if normal_pelvis[2] < 0.80:
        raise RuntimeError(f"fitted board plane is not level in pelvis frame: normal={normal_pelvis.tolist()}")

    center_u = 0.5 * (x1 + x2)
    base_ray = np.asarray([(center_u - cx) / fx, (y2 - cy) / fy, 1.0], dtype=float)
    denominator = float(normal_camera @ base_ray)
    if abs(denominator) < 1e-5:
        raise RuntimeError("camera ray is parallel to the fitted board plane")
    ray_scale = float(normal_camera @ plane_origin_camera) / denominator
    if not np.isfinite(ray_scale) or ray_scale <= 0.0:
        raise RuntimeError("invalid ray-plane intersection for the object base")
    base_camera = base_ray * ray_scale

    top_v = y1 + 0.10 * box_h
    top_u_px = int(round(center_u))
    top_v_px = int(round(top_v))
    top_crop = depth_raw[max(0, top_v_px - 4):min(image_h, top_v_px + 5),
                         max(0, top_u_px - 5):min(image_w, top_u_px + 6)].astype(np.float64) * depth_scale
    top_valid = top_crop[np.isfinite(top_crop) & (top_crop > 0.10) & (top_crop < 2.0)]
    if top_valid.size < 12:
        raise RuntimeError("too few valid depth samples at the top of the detected object")
    top_depth = float(np.median(top_valid))
    top_camera = np.asarray(
        [(center_u - cx) * top_depth / fx, (top_v - cy) * top_depth / fy, top_depth],
        dtype=float,
    )

    base_pelvis = camera_to_pelvis(base_camera, camera_pitch)
    top_pelvis = camera_to_pelvis(top_camera, camera_pitch)
    object_height = float((top_pelvis - base_pelvis) @ normal_pelvis)
    if not 0.045 <= object_height <= 0.35:
        raise RuntimeError(f"board-referenced object height is implausible: {object_height:.3f} m")

    grasp_center_pelvis = base_pelvis + float(height_fraction) * object_height * normal_pelvis
    return grasp_center_pelvis, {
        "plane_roi_xyxy": [roi_x0, roi_y0, roi_x1, roi_y1],
        "plane_sample_count": int(len(points)),
        "plane_rms_m": plane_rms,
        "plane_normal_pelvis": normal_pelvis.tolist(),
        "base_point_pelvis_m": base_pelvis.tolist(),
        "top_point_pelvis_m": top_pelvis.tolist(),
        "object_height_above_board_m": object_height,
        "grasp_height_fraction": float(height_fraction),
        "grasp_center_pelvis_m": grasp_center_pelvis.tolist(),
    }


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
            validate_detection_bbox(det, color_bgr.shape)
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

    board_reference = None
    capture_info = depth_stats.pop("_capture_info", None)
    if args.use_vision and args.vision_source == "realsense":
        if capture_info is None:
            raise RuntimeError("RGB-D capture metadata is unavailable for board-referenced targeting")
        point_pelvis, board_reference = estimate_board_referenced_object_center(
            capture_info,
            det,
            args.camera_pitch,
            args.vision_height_fraction,
        )
        depth_stats["board_reference"] = board_reference
    else:
        point_pelvis = camera_to_pelvis(point_camera, args.camera_pitch)

    target = pre_target.copy()
    target[0] = point_pelvis[0] + args.vision_xy_offset[0]
    object_y_with_offset = point_pelvis[1] + args.vision_xy_offset[1]
    approach_side = -1.0 if pre_target[1] <= object_y_with_offset else 1.0
    target[1] = object_y_with_offset + approach_side * args.side_standoff_y
    if board_reference is not None:
        target[2] = float(point_pelvis[2])
    else:
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
    if board_reference is not None:
        print("vision_target_z_source: calibration-board object height")
        print("vision_target_z_m:", f"{target[2]:.4f}")
    else:
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
    if min(args.pre_duration, args.lateral_duration, args.return_duration, args.release_duration) < 0.0:
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
        args.left_distance = float(grasp_target[1] - pre_target[1])
    else:
        grasp_target = pre_target.copy()
        grasp_target[1] += args.left_distance
    for label, target in (("pregrasp", pre_target), ("grasp", grasp_target)):
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

    pre_ik, q_pre_values, q_pre = solve_grasp_waypoint(
        pre_target, waist_q, right_joints, q_start_values, args
    )
    grasp_ik, q_grasp_values, q_grasp = solve_grasp_waypoint(
        grasp_target, waist_q, right_joints, q_pre_values, args
    )
    if not args.no_vertical_wrist:
        print("wrist_fixed_during_ik:")
        print(f"  RightWristRoll -> {args.wrist_roll_target:.4f} rad")
        print(f"  RightWristPitch -> {args.wrist_pitch_target:.4f} rad")
        print(f"  RightWristYaw -> {args.wrist_yaw_target:.4f} rad")
    else:
        print("wrist_override_enabled: false")

    print_plan_item("pregrasp", pre_target, pre_ik, q_pre_values, q_start_values, right_joints)
    print_command_position("pregrasp", pre_target, q_pre_values, waist_q, right_joints)
    print_plan_item("grasp", grasp_target, grasp_ik, q_grasp_values, q_pre_values, right_joints)
    print_command_position("grasp", grasp_target, q_grasp_values, waist_q, right_joints)
    print("left_hold:")
    print(format_joint_table(LEFT_ARM_7DOF, [q_left_hold[joint] for joint in LEFT_ARM_7DOF]))
    print("waist_hold:")
    print(format_joint_table(WAIST_JOINTS, [q_waist_hold[joint] for joint in WAIST_JOINTS]))

    if not pre_ik.success or not grasp_ik.success:
        print("Refusing execution because IK did not meet tolerance.", file=sys.stderr)
        return 4

    max_delta_pre = float(np.max(np.abs(q_pre_values - q_start_values)))
    max_delta_lateral = float(np.max(np.abs(q_grasp_values - q_pre_values)))
    max_delta = max(max_delta_pre, max_delta_lateral)
    print(f"max_pre_delta_rad: {max_delta_pre:.4f}")
    print(f"max_lateral_delta_rad: {max_delta_lateral:.4f}")
    if max_delta > args.max_joint_delta:
        print(f"Refusing execution: max joint delta {max_delta:.3f} exceeds {args.max_joint_delta:.3f}.", file=sys.stderr)
        return 5

    pre_duration = segment_duration(args.pre_duration, q_start_values, q_pre_values, args.max_joint_speed)
    lateral_duration = segment_duration(args.lateral_duration, q_pre_values, q_grasp_values, args.max_joint_speed)
    return_duration = segment_duration(args.return_duration, q_pre_values, q_start_values, args.max_joint_speed)
    print(f"pre_duration_s: {pre_duration:.2f}")
    print(f"lateral_duration_s: {lateral_duration:.2f}")
    print(f"return_duration_s: {return_duration:.2f}")

    if not args.execute:
        print("Dry run only. Add --execute --yes to move the robot.")
        return 0

    announce_start(args)

    q_right_last = dict(q_right_start)
    exit_code = 0
    feedback_stop = None
    feedback_thread = None
    arm_hold_stop = None
    arm_hold_thread = None
    arm_hold_state = None
    hand_closed_before_return = False
    grasp_contact_confirmed = False
    # This entry point is the legacy visual version, so arm motion is the
    # default even when the historical command does not include the newer
    # --move-* switches. The hand-only version lives in *_static.py.
    move_arm_for_grasp = True
    static_hand_grasp = (
        not args.skip_hand
        and not args.no_force_feedback_grasp
        and not args.approach_only
        and not move_arm_for_grasp
    )
    return_after_grasp = True
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

        if move_arm_for_grasp:
            print("Stage 1: move to saved pregrasp target.")
            run_timed_phase(
                controller,
                pre_duration,
                q_left_hold,
                q_waist_hold,
                q_right_start,
                q_pre,
                0.0,
                args.weight,
                args,
            )
            arm_grasp_hold = dict(q_pre)
        else:
            print("Stage 1: hand-only grasp; keep the arm at its current pose.")
            arm_grasp_hold = dict(q_right_start)
        q_right_last = dict(arm_grasp_hold)
        hold_group(controller, args.hold_before_open, q_left_hold, arm_grasp_hold, q_waist_hold, args.weight, args)

        if not args.skip_hand:
            print("Stage 2: open fingers; thumb is already at maximum rotation.")
            write_hand_angles_once(args, open_angles, "pregrasp_open")
            hand_closed_before_return = False

        if not args.skip_hand and args.hold_after_open > 0.0:
            print("Stage 2 hold: hold arm pose with hand open.")
            hold_group(
                controller,
                args.hold_after_open,
                q_left_hold,
                arm_grasp_hold,
                q_waist_hold,
                args.weight,
                args,
            )

        print("Stage 3: grasp phase.")
        feedback_state = None
        q_stage3_last = dict(q_grasp)
        if static_hand_grasp:
            print("Stage 3: static hand-only grasp; arm stays at its current pose while waiting for contact.")
            q_stage3_last = dict(arm_grasp_hold)
            arm_hold_stop, arm_hold_thread, arm_hold_state = start_arm_hold_thread(
                controller,
                q_left_hold,
                arm_grasp_hold,
                q_waist_hold,
                args.weight,
                args,
            )
            try:
                hand_closed_before_return = force_feedback_grasp(args)
            finally:
                stop_arm_hold_thread(arm_hold_stop, arm_hold_thread, arm_hold_state)
                arm_hold_stop = None
                arm_hold_thread = None
                arm_hold_state = None
            feedback_state = {
                "triggered": hand_closed_before_return,
                "thumb_rotated": hand_closed_before_return,
                "closed": hand_closed_before_return,
                "reason": "static_hand_force_feedback",
                "error": None,
            }
            grasp_contact_confirmed = bool(hand_closed_before_return)
        elif not args.skip_hand and not args.no_force_feedback_grasp and not args.approach_only:
            print("Stage 3 feedback-limited approach: force/current feedback stops arm advance, then closes hand.")
            q_stage3_last, feedback_state = run_feedback_limited_phase(
                controller,
                lateral_duration,
                q_left_hold,
                q_waist_hold,
                q_pre,
                q_grasp,
                args.weight,
                args.weight,
                args,
                thumb_inner_angles,
                close_angles,
            )
        elif move_arm_for_grasp or args.approach_only or args.skip_hand:
            run_timed_phase(controller, lateral_duration, q_left_hold, q_waist_hold, q_pre, q_grasp, args.weight, args.weight, args)
            q_stage3_last = dict(q_grasp)
        else:
            print("Stage 3: arm remains at its current pose; no grasp approach motion.")
            q_stage3_last = dict(arm_grasp_hold)
        q_right_last = dict(q_stage3_last)
        hold_group(controller, args.hold_before_grasp, q_left_hold, q_stage3_last, q_waist_hold, args.weight, args)

        if args.approach_only:
            print("Stage 4: approach-only mode; closing hand with thumb at maximum rotation before return.")
            if not args.skip_hand:
                write_safe_close_once(args, safe_close_angles, "approach_only_return_safe_close")
                hand_closed_before_return = True
        elif not args.skip_hand and not static_hand_grasp:
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
                hand_closed_before_return = bool(feedback_state and feedback_state.get("closed"))
                grasp_contact_confirmed = bool(
                    feedback_state
                    and feedback_state.get("closed")
                    and int(feedback_state.get("finger_contact_channels", 0)) >= 2
                )
                if not (feedback_state and feedback_state.get("closed")):
                    if args.close_on_feedback_timeout:
                        print("Stage 4: no feedback-triggered close; timeout policy rotates thumb inward, then closes fingers.")
                        write_thumb_then_close_once(args, thumb_inner_angles, close_angles, "timeout")
                        hand_closed_before_return = True
                        monitor_hand_feedback_window(args, "close", min(max(args.hand_hold, 0.5), 2.0))
                    else:
                        print("Stage 4: no feedback-triggered close; hand close skipped.")
            hold_group(controller, args.hold_after_grasp, q_left_hold, q_stage3_last, q_waist_hold, args.weight, args)

        if return_after_grasp and not args.skip_hand and not hand_closed_before_return:
            print("Stage 4 return prep: close hand with thumb at maximum rotation before moving back.")
            write_safe_close_once(args, safe_close_angles, "return_prep_safe_close")
            hand_closed_before_return = True

        if return_after_grasp:
            print("Stage 5: move back to saved pregrasp target.")
            run_timed_phase(controller, lateral_duration, q_left_hold, q_waist_hold, q_stage3_last, q_pre, args.weight, args.weight, args)
            q_right_last = dict(q_pre)

            print("Stage 6: retract arm to start posture.")
            run_timed_phase(controller, return_duration, q_left_hold, q_waist_hold, q_pre, q_right_start, args.weight, args.weight, args)
            q_right_last = dict(q_right_start)
        else:
            print("Stage 5: hold grasp posture; no arm retraction.")
            q_right_last = dict(q_stage3_last)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        exit_code = 130
    finally:
        if arm_hold_stop is not None and arm_hold_thread is not None and arm_hold_state is not None:
            try:
                stop_arm_hold_thread(arm_hold_stop, arm_hold_thread, arm_hold_state)
            except Exception as exc:
                print(f"Warning: failed to stop static grasp arm hold: {exc}", file=sys.stderr)
        if feedback_stop is not None and feedback_thread is not None and feedback_thread.is_alive():
            stop_feedback_monitor(feedback_stop, feedback_thread)
        print("Stage 7: release arm_sdk.")
        release_group(controller, q_left_hold, q_right_last, q_waist_hold, args.weight, args)
        print("arm_sdk released.")

    if exit_code == 0 and grasp_contact_confirmed:
        print("grasp_result: multi-finger contact detected and hand closed; object retention still needs visual confirmation.")
        announce_success(args)
    elif exit_code == 0 and args.use_vision and not args.approach_only:
        print("grasp_result: no confirmed multi-finger contact; object grasp not confirmed.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
