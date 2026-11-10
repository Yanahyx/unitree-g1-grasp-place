#!/usr/bin/env python3
"""Open the RH56E2 hand, move the arm, grasp, then optionally deliver the object."""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

from g1_right_arm_body_ik_example import (
    ArmSdkController,
    G1JointIndex,
    JOINT_LIMITS,
    JOINT_NAMES,
    DEFAULT_BASE_LINK_IN_PELVIS_RPY,
    DEFAULT_BASE_LINK_IN_PELVIS_XYZ,
    format_joint_delta_table,
    format_joint_table,
    matrix_to_rpy,
    model_to_motor_q,
    motor_to_model_q,
    right_arm_joints,
    right_hand_pose,
    solve_position_ik,
    target_to_pelvis_frame,
    validate_target,
)

HANDS_DIR = Path(__file__).resolve().parents[1] / "hands"
if str(HANDS_DIR) not in sys.path:
    sys.path.insert(0, str(HANDS_DIR))
HIGH_LEVEL_DIR = Path(__file__).resolve().parents[1] / "high_level"
if str(HIGH_LEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(HIGH_LEVEL_DIR))

from rh56e2_hand_only import write_hand_pose  # noqa: E402
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient  # noqa: E402


def waist_joints() -> list:
    return [
        G1JointIndex.WaistYaw,
        G1JointIndex.WaistRoll,
        G1JointIndex.WaistPitch,
    ]


def smoothstep(ratio: float) -> float:
    ratio = min(max(ratio, 0.0), 1.0)
    return ratio * ratio * (3.0 - 2.0 * ratio)


def smootherstep(ratio: float) -> float:
    ratio = min(max(ratio, 0.0), 1.0)
    return ratio * ratio * ratio * (ratio * (ratio * 6.0 - 15.0) + 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move right arm with g1_right_arm_body_ik_example logic, then command RH56E2 hand."
    )
    parser.add_argument("network_interface", nargs="?", help="DDS network interface, e.g. eth0")
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.30, -0.20, 0.10),
        help="Right hand target in --target-frame, meters.",
    )
    parser.add_argument(
        "--target-frame",
        choices=("base_link", "body", "pelvis", "map"),
        default="base_link",
        help="Frame of --target.",
    )
    parser.add_argument(
        "--base-link-in-pelvis-xyz",
        nargs=3,
        type=float,
        default=DEFAULT_BASE_LINK_IN_PELVIS_XYZ,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--base-link-in-pelvis-rpy",
        nargs=3,
        type=float,
        default=DEFAULT_BASE_LINK_IN_PELVIS_RPY,
        metavar=("R", "P", "Y"),
    )
    parser.add_argument(
        "--joint-signs",
        nargs=7,
        type=float,
        default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        metavar=("SP", "SR", "SY", "EL", "WR", "WP", "WY"),
    )
    parser.add_argument("--arm-dof", type=int, choices=(5, 7), default=7)
    parser.add_argument("--tolerance", type=float, default=0.025)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--max-joint-speed", type=float, default=0.45)
    parser.add_argument("--max-joint-delta", type=float, default=2.80)
    parser.add_argument("--allow-large-motion", action="store_true")
    parser.add_argument("--arm-sdk-weight", type=float, default=0.55, help="Max arm_sdk blend weight in (0, 1].")
    parser.add_argument("--allow-full-weight", action="store_true", help="Allow arm_sdk weight above 0.70.")
    parser.add_argument(
        "--arm-sdk-ramp-in-ratio",
        type=float,
        default=0.25,
        help="Portion of move duration used to ramp arm_sdk weight to target.",
    )
    parser.add_argument("--release-duration", type=float, default=1.0, help="Seconds to ramp arm_sdk weight down to 0.")
    parser.add_argument("--kp", type=float, default=40.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument("--wrist-roll-offset", type=float, default=0.0, help="Additional right wrist roll command in rad.")
    parser.add_argument("--wrist-pitch-offset", type=float, default=0.0, help="Additional right wrist pitch command in rad.")
    parser.add_argument("--wrist-yaw-offset", type=float, default=0.0, help="Additional right wrist yaw command in rad.")
    parser.add_argument("--hold-after-hand", type=float, default=2.0)
    parser.add_argument("--return-duration", type=float, default=6.0,
                        help="Seconds to return the arm to its initial pose before releasing arm_sdk.")
    parser.add_argument("--drop-distance", type=float, default=0.0, help="Optionally lower the grasped hand by this many meters.")
    parser.add_argument("--drop-duration", type=float, default=4.0, help="Seconds to lower the grasped hand if enabled.")
    parser.add_argument("--hold-after-drop", type=float, default=1.0, help="Seconds to hold after lowering.")
    parser.add_argument(
        "--post-grasp-outward",
        type=float,
        default=0.0,
        help="After grasping, move the right hand outward by this many meters before lowering.",
    )
    parser.add_argument(
        "--post-grasp-down",
        type=float,
        default=0.0,
        help="After the outward move, lower the grasped hand by this many meters.",
    )
    parser.add_argument("--post-grasp-outward-duration", type=float, default=2.0)
    parser.add_argument("--post-grasp-down-duration", type=float, default=3.0)
    parser.add_argument(
        "--deliver-target",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional final delivery target for the grasped object.",
    )
    parser.add_argument(
        "--deliver-target-frame",
        choices=("base_link", "body", "pelvis", "map"),
        default=None,
        help="Frame of --deliver-target. Defaults to --target-frame.",
    )
    parser.add_argument("--deliver-duration", type=float, default=4.0, help="Seconds to move to --deliver-target.")
    parser.add_argument("--hold-after-deliver", type=float, default=1.0, help="Seconds to hold at the delivery target.")
    parser.add_argument(
        "--no-return-after-drop",
        action="store_true",
        help="Do not return the arm to the initial pose after drop/delivery; release at the final pose.",
    )
    parser.add_argument(
        "--no-release-on-exit",
        action="store_true",
        help="Keep arm_sdk enabled at final pose; skip release ramp-down on exit.",
    )
    parser.add_argument(
        "--keep-enabled",
        action="store_true",
        help="Deprecated alias for --no-release-on-exit.",
    )
    parser.add_argument("--no-prompt", action="store_true")

    parser.add_argument("--hand-ip", default="192.168.124.210")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--hand-unit-id", type=int, default=1)
    parser.add_argument("--hand-timeout", type=float, default=2.0)
    parser.add_argument("--hand-speed", type=int, default=250)
    parser.add_argument("--hand-force", type=int, default=250)
    parser.add_argument("--hand-hold", type=float, default=1.0)
    parser.add_argument("--hand-read-before", action="store_true")
    parser.add_argument("--hand-read-after", action="store_true")
    parser.add_argument("--skip-initial-hand-open", action="store_true", help="Do not open the hand before arm motion.")
    parser.add_argument(
        "--hand-open-angle",
        type=int,
        nargs=6,
        default=[700, 700, 700, 700, 700, 700],
        metavar=("A0", "A1", "A2", "A3", "A4", "A5"),
        help="RH56E2 open angle command.",
    )
    parser.add_argument(
        "--hand-grasp-angle",
        type=int,
        nargs=6,
        default=[0, 0, 0, 0, 0, 0],
        metavar=("A0", "A1", "A2", "A3", "A4", "A5"),
        help="RH56E2 grasp/close angle command.",
    )
    parser.add_argument(
        "--hand-sequence",
        choices=("open", "grasp", "open-grasp", "grasp-open"),
        default="grasp",
        help="Hand command sequence after the arm reaches the target. Default is grasp.",
    )
    parser.add_argument(
        "--announce-text",
        default="抓取完成",
        help="TTS text spoken after the grasp routine completes successfully.",
    )
    parser.add_argument("--announce-speaker-id", type=int, default=0)
    parser.add_argument(
        "--no-announce",
        action="store_true",
        help="Disable the success TTS announcement after grasp completion.",
    )
    parser.add_argument("--yes", action="store_true", help="Confirm the arm and hand have clear space.")
    return parser.parse_args()


def hand_args(args: argparse.Namespace, read_after: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        ip=args.hand_ip,
        port=args.hand_port,
        unit_id=args.hand_unit_id,
        timeout=args.hand_timeout,
        speed=args.hand_speed,
        force=args.hand_force,
        hold=args.hand_hold,
        read_before=args.hand_read_before,
        read_after=read_after,
    )


def write_hand_sequence(args: argparse.Namespace) -> None:
    commands = {
        "open": [("open", args.hand_open_angle)],
        "grasp": [("grasp", args.hand_grasp_angle)],
        "open-grasp": [
            ("open", args.hand_open_angle),
            ("grasp", args.hand_grasp_angle),
        ],
        "grasp-open": [
            ("grasp", args.hand_grasp_angle),
            ("open", args.hand_open_angle),
        ],
    }[args.hand_sequence]

    for index, (label, angle) in enumerate(commands):
        command_args = hand_args(args, read_after=args.hand_read_after and index == len(commands) - 1)
        write_hand_pose(command_args, angle, label)


def hold_arm_until_stopped(
    controller: ArmSdkController,
    hold_positions: dict,
    kp: float,
    kd: float,
    stop_event: threading.Event,
    arm_sdk_weight: float,
) -> None:
    while not stop_event.is_set():
        controller.write_arm_command(hold_positions, kp, kd, arm_sdk_weight)
        time.sleep(controller.control_dt)


def move_right_arm_to(
    controller: ArmSdkController,
    controlled_joints: list,
    goal_positions: dict,
    duration: float,
    hold_time: float,
    kp: float,
    kd: float,
    arm_sdk_weight: float,
    arm_sdk_ramp_in_ratio: float,
    ramp_in_weight: bool,
) -> None:
    start_positions = controller.current_positions(controlled_joints)
    all_goals = dict(start_positions)
    all_goals.update(goal_positions)

    start_time = time.monotonic()
    next_tick = start_time
    while True:
        now = time.monotonic()
        elapsed = now - start_time
        ratio = smootherstep(elapsed / duration)
        targets = {
            joint: start_positions[joint] + ratio * (all_goals[joint] - start_positions[joint])
            for joint in controlled_joints
        }
        if ramp_in_weight:
            ramp_denominator = max(arm_sdk_ramp_in_ratio, 1e-6)
            ramp_phase = min(1.0, (elapsed / max(duration, 1e-6)) / ramp_denominator)
            weight = arm_sdk_weight * smootherstep(ramp_phase)
        else:
            weight = arm_sdk_weight
        controller.write_arm_command(targets, kp, kd, weight)
        if elapsed >= duration:
            break
        next_tick += controller.control_dt
        time.sleep(max(0.0, next_tick - time.monotonic()))

    hold_deadline = time.monotonic() + hold_time
    while time.monotonic() < hold_deadline:
        controller.write_arm_command(all_goals, kp, kd, arm_sdk_weight)
        time.sleep(controller.control_dt)


def release_arm_sdk_safely(
    controller: ArmSdkController,
    hold_positions: dict,
    kp: float,
    kd: float,
    arm_sdk_weight: float,
    release_duration: float,
) -> None:
    try:
        start_time = time.monotonic()
        while True:
            elapsed = time.monotonic() - start_time
            ratio = min(max(elapsed / max(release_duration, 1e-6), 0.0), 1.0)
            remaining = 1.0 - ratio
            weight = arm_sdk_weight * remaining
            controller.write_arm_command(
                hold_positions,
                kp * remaining,
                kd * remaining,
                weight,
            )
            if ratio >= 1.0:
                break
            time.sleep(controller.control_dt)
        print("arm_sdk released.")
    except Exception as exc:
        print(f"Failed to release arm_sdk cleanly: {exc}", file=sys.stderr)


def announce_success(args: argparse.Namespace) -> None:
    if args.no_announce or not args.announce_text:
        return

    try:
        audio_client = AudioClient()
        audio_client.SetTimeout(10.0)
        audio_client.Init()
        code = audio_client.TtsMaker(args.announce_text, args.announce_speaker_id)
        if code != 0:
            print(f"Warning: TTS announcement failed with code {code}.", file=sys.stderr)
        else:
            print(f"TTS announced: {args.announce_text}")
    except Exception as exc:
        print(f"Warning: TTS announcement failed: {exc}", file=sys.stderr)


def solve_right_arm_goal(
    target: np.ndarray,
    waist_q: np.ndarray,
    right_goal_joints: list,
    signs: np.ndarray,
    seed_model_q: np.ndarray,
    tolerance: float,
):
    ik = solve_position_ik(
        target=target,
        waist_q=waist_q,
        joint_indices=right_goal_joints,
        signs=signs,
        seed=seed_model_q,
        tolerance=tolerance,
    )
    motor_q = model_to_motor_q(ik.q_model, signs)
    pose = right_hand_pose(ik.q_model, waist_q, right_goal_joints)
    reached_rpy = matrix_to_rpy(pose[:3, :3])
    return ik, motor_q, reached_rpy


def apply_wrist_offsets(
    joints: list,
    motor_q: np.ndarray,
    wrist_roll_offset: float,
    wrist_pitch_offset: float,
    wrist_yaw_offset: float,
) -> np.ndarray:
    adjusted = np.asarray(motor_q, dtype=float).copy()
    for idx, joint in enumerate(joints):
        if joint == G1JointIndex.RightWristRoll:
            adjusted[idx] += wrist_roll_offset
        elif joint == G1JointIndex.RightWristPitch:
            adjusted[idx] += wrist_pitch_offset
        elif joint == G1JointIndex.RightWristYaw:
            adjusted[idx] += wrist_yaw_offset
    return adjusted


def command_pose_metrics(
    target: np.ndarray,
    waist_q: np.ndarray,
    joints: list,
    motor_q: np.ndarray,
    signs: np.ndarray,
):
    model_q = motor_to_model_q(motor_q, signs)
    pose = right_hand_pose(model_q, waist_q, joints)
    reached = pose[:3, 3]
    reached_rpy = matrix_to_rpy(pose[:3, :3])
    residual = float(np.linalg.norm(target - reached))
    return reached, reached_rpy, residual


def validate_arm_goal(joints: list, q_values: np.ndarray, label: str) -> int:
    for joint, value in zip(joints, q_values):
        if joint not in JOINT_LIMITS:
            continue
        lo, hi = JOINT_LIMITS[joint]
        if value < lo or value > hi:
            print(
                f"Refusing {label}: {JOINT_NAMES[joint]}={value:.3f} outside [{lo:.3f}, {hi:.3f}]",
                file=sys.stderr,
            )
            return 4
    return 0


def main() -> int:
    args = parse_args()
    skip_release_on_exit = args.no_release_on_exit or args.keep_enabled
    if args.keep_enabled:
        print("Warning: --keep-enabled is deprecated; use --no-release-on-exit.", file=sys.stderr)
    if not args.yes:
        print("Refusing to move without --yes.", file=sys.stderr)
        return 1
    if args.hand_speed < 0 or args.hand_speed > 1000 or args.hand_force < 0 or args.hand_force > 1000:
        print("Refusing hand command: hand speed/force must be in 0..1000.", file=sys.stderr)
        return 2
    if any(value < 0 or value > 1000 for value in args.hand_open_angle + args.hand_grasp_angle):
        print("Refusing hand command: all hand angles must be in 0..1000.", file=sys.stderr)
        return 2
    if args.drop_distance < 0 or args.drop_duration <= 0 or args.hold_after_drop < 0:
        print("Refusing drop command: drop-distance must be non-negative; drop-duration must be positive.", file=sys.stderr)
        return 2
    if args.post_grasp_outward < 0.0 or args.post_grasp_down < 0.0:
        print("Refusing post-grasp command: outward/down distances must be non-negative.", file=sys.stderr)
        return 2
    if args.post_grasp_outward_duration <= 0.0 or args.post_grasp_down_duration <= 0.0:
        print("Refusing post-grasp command: durations must be positive.", file=sys.stderr)
        return 2
    if args.deliver_duration <= 0 or args.hold_after_deliver < 0:
        print("Refusing delivery command: deliver-duration must be positive.", file=sys.stderr)
        return 2
    if args.arm_sdk_weight <= 0.0 or args.arm_sdk_weight > 1.0:
        print("Refusing arm command: --arm-sdk-weight must be in (0, 1].", file=sys.stderr)
        return 2
    if args.arm_sdk_weight > 0.70 and not args.allow_full_weight:
        print("Refusing arm command: arm_sdk weight above 0.70 requires --allow-full-weight.", file=sys.stderr)
        return 2
    if args.arm_sdk_ramp_in_ratio <= 0.0:
        print("Refusing arm command: --arm-sdk-ramp-in-ratio must be positive.", file=sys.stderr)
        return 2
    if args.release_duration <= 0.0:
        print("Refusing arm command: --release-duration must be positive.", file=sys.stderr)
        return 2

    target = np.asarray(args.target, dtype=float)
    signs = np.asarray(args.joint_signs, dtype=float)
    signs = np.where(signs >= 0.0, 1.0, -1.0)
    right_goal_joints = right_arm_joints(args.arm_dof)
    wrist_offsets = (
        args.wrist_roll_offset,
        args.wrist_pitch_offset,
        args.wrist_yaw_offset,
    )
    if args.arm_dof == 5 and (abs(args.wrist_pitch_offset) > 1e-9 or abs(args.wrist_yaw_offset) > 1e-9):
        print(
            "Warning: wrist pitch/yaw offsets are ignored when --arm-dof=5.",
            file=sys.stderr,
        )

    try:
        pelvis_target, base_link_target = target_to_pelvis_frame(
            target=target,
            target_frame=args.target_frame,
            base_link_in_pelvis_xyz=args.base_link_in_pelvis_xyz,
            base_link_in_pelvis_rpy=args.base_link_in_pelvis_rpy,
        )
        validate_target(pelvis_target)
        deliver_pelvis_target = None
        deliver_input_target = None
        deliver_base_link_target = None
        if args.deliver_target is not None:
            deliver_input_target = np.asarray(args.deliver_target, dtype=float)
            deliver_frame = args.deliver_target_frame or args.target_frame
            deliver_pelvis_target, deliver_base_link_target = target_to_pelvis_frame(
                target=deliver_input_target,
                target_frame=deliver_frame,
                base_link_in_pelvis_xyz=args.base_link_in_pelvis_xyz,
                base_link_in_pelvis_rpy=args.base_link_in_pelvis_rpy,
            )
            validate_target(deliver_pelvis_target)
    except (RuntimeError, ValueError) as exc:
        print(f"Refusing target: {exc}", file=sys.stderr)
        return 2

    controller = ArmSdkController(args.network_interface, args.arm_dof, control_dt=0.02)
    controller.init()
    read_joints = right_goal_joints + waist_joints()
    current_positions = controller.current_positions(read_joints)
    current_motor_q = np.array([current_positions[joint] for joint in right_goal_joints])
    current_model_seed = motor_to_model_q(current_motor_q, signs)
    current_waist_q = np.array(
        [
            current_positions[G1JointIndex.WaistYaw],
            current_positions[G1JointIndex.WaistRoll],
            current_positions[G1JointIndex.WaistPitch],
        ]
    )

    ik, right_goal_q, reached_rpy = solve_right_arm_goal(
        target=pelvis_target,
        waist_q=current_waist_q,
        right_goal_joints=right_goal_joints,
        signs=signs,
        seed_model_q=current_model_seed,
        tolerance=args.tolerance,
    )
    right_goal_q = apply_wrist_offsets(right_goal_joints, right_goal_q, *wrist_offsets)
    command_reached, command_rpy, command_residual = command_pose_metrics(
        pelvis_target,
        current_waist_q,
        right_goal_joints,
        right_goal_q,
        signs,
    )

    planned_moves = [
        {
            "name": "grasp target",
            "target": pelvis_target,
            "ik": ik,
            "q": right_goal_q,
            "rpy": reached_rpy,
            "command_reached": command_reached,
            "command_rpy": command_rpy,
            "command_residual": command_residual,
            "duration": args.duration,
            "hold": 0.0,
        }
    ]

    seed_model_q = ik.q_model

    def append_planned_move(name: str, target_value: np.ndarray, duration: float, hold: float) -> None:
        nonlocal seed_model_q
        move_ik, move_q, move_reached_rpy = solve_right_arm_goal(
            target=target_value,
            waist_q=current_waist_q,
            right_goal_joints=right_goal_joints,
            signs=signs,
            seed_model_q=seed_model_q,
            tolerance=args.tolerance,
        )
        move_q = apply_wrist_offsets(right_goal_joints, move_q, *wrist_offsets)
        move_command_reached, move_command_rpy, move_command_residual = command_pose_metrics(
            target_value,
            current_waist_q,
            right_goal_joints,
            move_q,
            signs,
        )
        planned_moves.append(
            {
                "name": name,
                "target": target_value,
                "ik": move_ik,
                "q": move_q,
                "rpy": move_reached_rpy,
                "command_reached": move_command_reached,
                "command_rpy": move_command_rpy,
                "command_residual": move_command_residual,
                "duration": duration,
                "hold": hold,
            }
        )
        seed_model_q = move_ik.q_model

    if args.post_grasp_outward > 0.0:
        outward_pelvis_target = planned_moves[-1]["target"].copy()
        outward_pelvis_target[1] -= args.post_grasp_outward
        append_planned_move(
            "outward",
            outward_pelvis_target,
            args.post_grasp_outward_duration,
            0.0,
        )

    if args.post_grasp_down > 0.0:
        lower_pelvis_target = planned_moves[-1]["target"].copy()
        lower_pelvis_target[2] -= args.post_grasp_down
        append_planned_move(
            "lower",
            lower_pelvis_target,
            args.post_grasp_down_duration,
            args.hold_after_drop,
        )

    if args.drop_distance > 0.0:
        drop_pelvis_target = planned_moves[-1]["target"].copy()
        drop_pelvis_target[2] -= args.drop_distance
        drop_ik, right_drop_q, drop_reached_rpy = solve_right_arm_goal(
            target=drop_pelvis_target,
            waist_q=current_waist_q,
            right_goal_joints=right_goal_joints,
            signs=signs,
            seed_model_q=seed_model_q,
            tolerance=args.tolerance,
        )
        right_drop_q = apply_wrist_offsets(right_goal_joints, right_drop_q, *wrist_offsets)
        drop_command_reached, drop_command_rpy, drop_command_residual = command_pose_metrics(
            drop_pelvis_target,
            current_waist_q,
            right_goal_joints,
            right_drop_q,
            signs,
        )
        planned_moves.append(
            {
                "name": "drop",
                "target": drop_pelvis_target,
                "ik": drop_ik,
                "q": right_drop_q,
                "rpy": drop_reached_rpy,
                "command_reached": drop_command_reached,
                "command_rpy": drop_command_rpy,
                "command_residual": drop_command_residual,
                "duration": args.drop_duration,
                "hold": args.hold_after_drop,
            }
        )
        seed_model_q = drop_ik.q_model

    if deliver_pelvis_target is not None:
        deliver_ik, right_deliver_q, deliver_reached_rpy = solve_right_arm_goal(
            target=deliver_pelvis_target,
            waist_q=current_waist_q,
            right_goal_joints=right_goal_joints,
            signs=signs,
            seed_model_q=seed_model_q,
            tolerance=args.tolerance,
        )
        right_deliver_q = apply_wrist_offsets(right_goal_joints, right_deliver_q, *wrist_offsets)
        deliver_command_reached, deliver_command_rpy, deliver_command_residual = command_pose_metrics(
            deliver_pelvis_target,
            current_waist_q,
            right_goal_joints,
            right_deliver_q,
            signs,
        )
        planned_moves.append(
            {
                "name": "delivery",
                "target": deliver_pelvis_target,
                "ik": deliver_ik,
                "q": right_deliver_q,
                "rpy": deliver_reached_rpy,
                "command_reached": deliver_command_reached,
                "command_rpy": deliver_command_rpy,
                "command_residual": deliver_command_residual,
                "duration": args.deliver_duration,
                "hold": args.hold_after_deliver,
            }
        )

    print(f"Input target ({args.target_frame}):", np.array2string(target, precision=3))
    if base_link_target is not None:
        print("Base_link-frame target:", np.array2string(base_link_target, precision=3))
    print("Pelvis-frame IK target:", np.array2string(pelvis_target, precision=3))
    print("URDF reached position (IK):", np.array2string(ik.reached, precision=3))
    print("URDF reached RPY (IK):", np.array2string(reached_rpy, precision=3))
    print(
        "Command reached position (with wrist offsets):",
        np.array2string(planned_moves[0]["command_reached"], precision=3),
    )
    print(
        "Command reached RPY (with wrist offsets):",
        np.array2string(planned_moves[0]["command_rpy"], precision=3),
    )
    print(f"IK final residual: {ik.residual:.4f} m after {ik.iterations} iterations ({ik.message})")
    print(f"Command residual: {planned_moves[0]['command_residual']:.4f} m")
    print("Right arm command:")
    print(format_joint_table(right_goal_joints, right_goal_q))
    if deliver_input_target is not None:
        print(
            f"Delivery input target ({args.deliver_target_frame or args.target_frame}):",
            np.array2string(deliver_input_target, precision=3),
        )
        if deliver_base_link_target is not None:
            print("Delivery base_link-frame target:", np.array2string(deliver_base_link_target, precision=3))
    for move in planned_moves[1:]:
        move_ik = move["ik"]
        print(f"{move['name'].title()} pelvis-frame target:", np.array2string(move["target"], precision=3))
        print(f"{move['name'].title()} reached position (IK):", np.array2string(move_ik.reached, precision=3))
        print(f"{move['name'].title()} reached RPY (IK):", np.array2string(move["rpy"], precision=3))
        print(
            f"{move['name'].title()} command position:",
            np.array2string(move["command_reached"], precision=3),
        )
        print(
            f"{move['name'].title()} command RPY:",
            np.array2string(move["command_rpy"], precision=3),
        )
        print(f"{move['name'].title()} command residual: {move['command_residual']:.4f} m")
        print(
            f"{move['name'].title()} IK final residual: {move_ik.residual:.4f} m "
            f"after {move_ik.iterations} iterations ({move_ik.message})"
        )
        print(f"{move['name'].title()} arm command:")
        print(format_joint_table(right_goal_joints, move["q"]))

    if any(abs(value) > 1e-9 for value in wrist_offsets):
        for move in planned_moves:
            if move["command_residual"] > args.tolerance:
                print(
                    f"Warning: {move['name']} command residual {move['command_residual']:.4f} m exceeds "
                    f"tolerance {args.tolerance:.4f} m after applying wrist offsets.",
                    file=sys.stderr,
                )

    for move in planned_moves:
        if not move["ik"].success:
            print(f"Refusing execution because {move['name']} IK did not meet tolerance.", file=sys.stderr)
            return 3
        limit_error = validate_arm_goal(right_goal_joints, move["q"], f"{move['name']} execution")
        if limit_error:
            return limit_error

    previous_q = current_motor_q
    for move in planned_moves:
        move["delta"] = np.abs(move["q"][: len(previous_q)] - previous_q)
        move["max_delta"] = float(np.max(move["delta"])) if len(move["delta"]) else 0.0
        if move["max_delta"] > args.max_joint_delta and not args.allow_large_motion:
            print(
                f"Refusing execution: {move['name']} max joint delta {move['max_delta']:.3f} rad exceeds "
                f"--max-joint-delta {args.max_joint_delta:.3f}.",
                file=sys.stderr,
            )
            print(f"{move['name'].title()} joint deltas:", file=sys.stderr)
            print(format_joint_delta_table(right_goal_joints, previous_q, move["q"]), file=sys.stderr)
            return 5
        previous_q = move["q"]

    if not args.no_prompt:
        print("WARNING: hand will open, arm will move, hand will grasp, then optional delivery motion will run.")
        input("Press Enter to continue, or Ctrl-C to abort...")

    for move in planned_moves:
        move["runtime_duration"] = max(move["duration"], move["max_delta"] / max(args.max_joint_speed, 1e-3))

    right_goal_positions = {joint: float(value) for joint, value in zip(right_goal_joints, planned_moves[0]["q"])}
    hold_positions = {joint: current_positions[joint] for joint in right_goal_joints}
    stop_event = threading.Event()
    hold_thread = None
    exit_code = 0

    try:
        if not args.skip_initial_hand_open:
            print("Step 1: opening hand before arm motion.")
            open_args = hand_args(args)
            write_hand_pose(open_args, args.hand_open_angle, "initial_open")

        print("Step 2: moving right arm only to target with g1_right_arm_body_ik_example logic.")
        move_right_arm_to(
            controller=controller,
            controlled_joints=right_goal_joints,
            goal_positions=right_goal_positions,
            duration=planned_moves[0]["runtime_duration"],
            hold_time=0.0,
            kp=args.kp,
            kd=args.kd,
            arm_sdk_weight=args.arm_sdk_weight,
            arm_sdk_ramp_in_ratio=args.arm_sdk_ramp_in_ratio,
            ramp_in_weight=True,
        )
        hold_positions.update(right_goal_positions)

        hold_thread = threading.Thread(
            target=hold_arm_until_stopped,
            args=(controller, hold_positions, args.kp, args.kd, stop_event, args.arm_sdk_weight),
            name="hold_body_ik_arm_target",
            daemon=True,
        )
        hold_thread.start()

        print(f"Step 3: arm is held; running hand sequence: {args.hand_sequence}")
        write_hand_sequence(args)
        if args.hold_after_hand > 0.0:
            print(f"Holding arm after hand sequence for {args.hold_after_hand:.1f}s.")
            time.sleep(args.hold_after_hand)

        stop_event.set()
        if hold_thread is not None:
            hold_thread.join(timeout=1.0)
            hold_thread = None

        for step_index, move in enumerate(planned_moves[1:], start=4):
            print(f"Step {step_index}: moving grasped object to {move['name']} target.")
            move_positions = {joint: float(value) for joint, value in zip(right_goal_joints, move["q"])}
            move_right_arm_to(
                controller=controller,
                controlled_joints=right_goal_joints,
                goal_positions=move_positions,
                duration=move["runtime_duration"],
                hold_time=move["hold"],
                kp=args.kp,
                kd=args.kd,
                arm_sdk_weight=args.arm_sdk_weight,
                arm_sdk_ramp_in_ratio=args.arm_sdk_ramp_in_ratio,
                ramp_in_weight=False,
            )
            hold_positions.update(move_positions)

        if args.no_return_after_drop:
            print("Skipping return-to-initial pose because --no-return-after-drop is set.")
        else:
            return_step = 4 + len(planned_moves[1:])
            print(f"Step {return_step}: returning arm to initial pose before releasing arm_sdk.")
            return_positions = {joint: float(value) for joint, value in zip(right_goal_joints, current_motor_q)}
            return_duration = max(args.return_duration, 1.0)
            move_right_arm_to(
                controller=controller,
                controlled_joints=right_goal_joints,
                goal_positions=return_positions,
                duration=return_duration,
                hold_time=0.5,
                kp=args.kp,
                kd=args.kd,
                arm_sdk_weight=args.arm_sdk_weight,
                arm_sdk_ramp_in_ratio=args.arm_sdk_ramp_in_ratio,
                ramp_in_weight=False,
            )
            hold_positions.update(return_positions)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        exit_code = 130
    finally:
        stop_event.set()
        if hold_thread is not None:
            hold_thread.join(timeout=1.0)
        if skip_release_on_exit:
            print("Skipping arm_sdk release on exit; arm_sdk stays enabled at final pose.")
        else:
            release_arm_sdk_safely(
                controller,
                hold_positions,
                args.kp,
                args.kd,
                args.arm_sdk_weight,
                args.release_duration,
            )

    if exit_code == 0:
        announce_success(args)
        print("Done.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
