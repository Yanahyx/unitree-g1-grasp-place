#!/usr/bin/env python3
"""Table-contact place sequence for the real G1 right arm.

Sequence:
1. Move right hand to PREPLACE_TARGET in pelvis frame.
2. Move the whole right hand to the final place target above the table.
3. Lower the hand by LOWER_DISTANCE to set the object on the table.
4. Open the RH56E2 hand to release the object.
5. Lift back to the place target, then retreat to PREPLACE_TARGET.
6. Close the RH56E2 hand back to the contracted preset.
7. Return the right arm to its start posture and release arm_sdk.

This script is separate from g1_right_arm_lateral_place_sequence.py so the
existing place behavior is kept intact.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from g1_right_arm_lateral_grasp_sequence import (  # noqa: E402
    DEFAULT_HAND_GRASP_ANGLE,
    DEFAULT_HAND_OPEN_ANGLE,
    DEFAULT_WRIST_PITCH,
    DEFAULT_WRIST_ROLL,
    DEFAULT_WRIST_YAW,
    G1JointIndex,
    LEFT_ARM_7DOF,
    PREGRASP_TARGET,
    WAIST_JOINTS,
    OfficialStyleArmSdkController,
    announce_success,
    apply_wrist_targets,
    detect_interface,
    format_joint_table,
    hand_args,
    hand_tcp_preflight,
    hold_group,
    joint_positions,
    joint_vector,
    joint_velocity_abs_max,
    print_command_position,
    print_plan_item,
    release_group,
    right_arm_joints,
    run_timed_phase,
    segment_duration,
    solve_right_q,
    validate_hand_args,
    validate_target,
    validate_wrist_target,
    write_hand_pose,
)


DEFAULT_LEFT_DISTANCE = 0.10
DEFAULT_LOWER_DISTANCE = 0.04


def parse_place_target(values: Optional[List[float]], preplace_target: np.ndarray, left_distance: float) -> np.ndarray:
    if values is not None:
        return np.asarray(values, dtype=float)
    target = preplace_target.copy()
    target[1] += left_distance
    return target


def table_place_target(place_target: np.ndarray, lower_distance: float) -> np.ndarray:
    target = place_target.copy()
    target[2] -= lower_distance
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Right-arm table-contact place/release sequence.")
    parser.add_argument("network_interface", nargs="?", default=None, help="DDS interface, e.g. eth1.")
    parser.add_argument(
        "--preplace-target",
        nargs=3,
        type=float,
        default=PREGRASP_TARGET.tolist(),
        metavar=("X", "Y", "Z"),
        help="Approach/retract point in pelvis frame. Default matches the grasp pregrasp target.",
    )
    parser.add_argument(
        "--place-target",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Release point above the table in pelvis frame. Default: preplace target plus --left-distance in +Y.",
    )
    parser.add_argument(
        "--left-distance",
        type=float,
        default=DEFAULT_LEFT_DISTANCE,
        help="Pelvis-frame +Y lateral move from --preplace-target when --place-target is omitted.",
    )
    parser.add_argument(
        "--lower-distance",
        type=float,
        default=DEFAULT_LOWER_DISTANCE,
        help="Meters to lower from --place-target before opening the hand.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("preplace", "place", "table-place"),
        default=None,
        help="Debug only: stop after reaching one planned point, then release arm_sdk.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually move the arm and hand.")
    parser.add_argument("--yes", action="store_true", help="Confirm the robot, hand, table, and receiving area are clear.")
    parser.add_argument("--weight", type=float, default=0.75)
    parser.add_argument("--allow-full-weight", action="store_true")
    parser.add_argument("--pre-duration", type=float, default=6.0)
    parser.add_argument("--place-duration", type=float, default=3.0)
    parser.add_argument("--lower-duration", type=float, default=2.0)
    parser.add_argument("--return-duration", type=float, default=5.0)
    parser.add_argument("--hold-before-release", type=float, default=0.3)
    parser.add_argument("--hold-after-release", type=float, default=0.8)
    parser.add_argument("--hold-after-hand-close", type=float, default=0.3)
    parser.add_argument(
        "--manual-load",
        action="store_true",
        help="Before placing, open the hand so an object can be loaded manually, then close the hand.",
    )
    parser.add_argument(
        "--manual-load-open-hold",
        type=float,
        default=5.0,
        help="Seconds to hold the hand open for manual object loading.",
    )
    parser.add_argument(
        "--manual-load-close-hold",
        type=float,
        default=0.8,
        help="Seconds to wait after closing the hand before starting the place motion.",
    )
    parser.add_argument(
        "--wrist-roll-target",
        type=float,
        default=DEFAULT_WRIST_ROLL,
        help="RightWristRoll target used for the place wrist pose.",
    )
    parser.add_argument(
        "--wrist-pitch-target",
        type=float,
        default=DEFAULT_WRIST_PITCH,
        help="RightWristPitch target used for the place wrist pose.",
    )
    parser.add_argument(
        "--wrist-yaw-target",
        type=float,
        default=DEFAULT_WRIST_YAW,
        help="RightWristYaw target used for the place wrist pose.",
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
    parser.add_argument("--hand-force", type=int, default=1000)
    parser.add_argument("--hand-hold", type=float, default=0.8)
    parser.add_argument("--hand-open-angle", nargs=6, type=int, default=DEFAULT_HAND_OPEN_ANGLE)
    parser.add_argument("--hand-grasp-angle", nargs=6, type=int, default=DEFAULT_HAND_GRASP_ANGLE, help=argparse.SUPPRESS)
    parser.add_argument("--skip-hand", action="store_true", help="Run arm motion only; do not command RH56E2.")
    parser.add_argument("--announce-text", default="放置完成")
    parser.add_argument("--announce-speaker-id", type=int, default=0)
    parser.add_argument("--no-announce", action="store_true")
    return parser.parse_args()


def solve_target(name, target, waist_q, right_joints, seed_values, args):
    ik, q_values, q_target = solve_right_q(target, waist_q, right_joints, seed_values, args.tolerance)
    if not args.no_vertical_wrist:
        q_values, q_target = apply_wrist_targets(
            q_values,
            q_target,
            right_joints,
            args.wrist_roll_target,
            args.wrist_pitch_target,
            args.wrist_yaw_target,
        )
    return ik, q_values, q_target


def hand_args_with_hold(args: argparse.Namespace, hold: float) -> argparse.Namespace:
    hand = hand_args(args)
    hand.hold = hold
    return hand


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
    if args.place_target is None and args.left_distance < 0.0:
        print("--left-distance must be non-negative when --place-target is omitted.", file=sys.stderr)
        return 2
    if args.manual_load and args.skip_hand:
        print("Refusing --manual-load with --skip-hand.", file=sys.stderr)
        return 2
    if args.lower_distance < 0.0:
        print("--lower-distance must be non-negative.", file=sys.stderr)
        return 2
    if not args.no_vertical_wrist:
        try:
            validate_wrist_target(G1JointIndex.RightWristRoll, args.wrist_roll_target)
            validate_wrist_target(G1JointIndex.RightWristPitch, args.wrist_pitch_target)
            validate_wrist_target(G1JointIndex.RightWristYaw, args.wrist_yaw_target)
        except ValueError as exc:
            print(f"Refusing wrist target: {exc}", file=sys.stderr)
            return 2
    if (
        min(
            args.pre_duration,
            args.place_duration,
            args.lower_duration,
            args.return_duration,
            args.release_duration,
            args.hold_before_release,
            args.hold_after_release,
            args.hold_after_hand_close,
            args.manual_load_open_hold,
            args.manual_load_close_hold,
        )
        < 0.0
    ):
        print("Durations must be non-negative.", file=sys.stderr)
        return 2
    hand_error = validate_hand_args(args)
    if hand_error:
        return hand_error
    if args.execute and not args.skip_hand and not hand_tcp_preflight(args):
        return 6

    preplace_target = np.asarray(args.preplace_target, dtype=float)
    place_target = parse_place_target(args.place_target, preplace_target, args.left_distance)
    table_target = table_place_target(place_target, args.lower_distance)
    for label, target in (("preplace", preplace_target), ("place", place_target), ("table_place", table_target)):
        try:
            validate_target(target)
        except ValueError as exc:
            print(f"Refusing {label} target: {exc}", file=sys.stderr)
            return 2

    print("preplace_target:", np.array2string(preplace_target, precision=4))
    print("place_target_above_table:", np.array2string(place_target, precision=4))
    print("table_place_target:", np.array2string(table_target, precision=4))
    print(f"computed_x_delta: {float(place_target[0] - preplace_target[0]):.4f}")
    print(f"computed_left_distance: {float(place_target[1] - preplace_target[1]):.4f}")
    print(f"computed_lower_distance: {args.lower_distance:.4f}")

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

    pre_ik, q_pre_values, q_pre = solve_target("preplace", preplace_target, waist_q, right_joints, q_start_values, args)
    place_ik, q_place_values, q_place = solve_target("place", place_target, waist_q, right_joints, q_pre_values, args)
    table_ik, q_table_values, q_table = solve_target(
        "table_place", table_target, waist_q, right_joints, q_place_values, args
    )
    if not args.no_vertical_wrist:
        print("wrist_override_enabled:")
        print(f"  RightWristRoll -> {args.wrist_roll_target:.4f} rad")
        print(f"  RightWristPitch -> {args.wrist_pitch_target:.4f} rad")
        print(f"  RightWristYaw -> {args.wrist_yaw_target:.4f} rad")
    else:
        print("wrist_override_enabled: false")

    print_plan_item("preplace", preplace_target, pre_ik, q_pre_values, q_start_values, right_joints)
    print_command_position("preplace", preplace_target, q_pre_values, waist_q, right_joints)
    print_plan_item("place", place_target, place_ik, q_place_values, q_pre_values, right_joints)
    print_command_position("place", place_target, q_place_values, waist_q, right_joints)
    print_plan_item("table_place", table_target, table_ik, q_table_values, q_place_values, right_joints)
    print_command_position("table_place", table_target, q_table_values, waist_q, right_joints)
    print("left_hold:")
    print(format_joint_table(LEFT_ARM_7DOF, [q_left_hold[joint] for joint in LEFT_ARM_7DOF]))
    print("waist_hold:")
    print(format_joint_table(WAIST_JOINTS, [q_waist_hold[joint] for joint in WAIST_JOINTS]))

    if not pre_ik.success or not place_ik.success or not table_ik.success:
        print("Refusing execution because IK did not meet tolerance.", file=sys.stderr)
        return 4

    max_delta_pre = float(np.max(np.abs(q_pre_values - q_start_values)))
    max_delta_place = float(np.max(np.abs(q_place_values - q_pre_values)))
    max_delta_lower = float(np.max(np.abs(q_table_values - q_place_values)))
    max_delta = max(max_delta_pre, max_delta_place, max_delta_lower)
    print(f"max_pre_delta_rad: {max_delta_pre:.4f}")
    print(f"max_place_delta_rad: {max_delta_place:.4f}")
    print(f"max_lower_delta_rad: {max_delta_lower:.4f}")
    if max_delta > args.max_joint_delta:
        print(f"Refusing execution: max joint delta {max_delta:.3f} exceeds {args.max_joint_delta:.3f}.", file=sys.stderr)
        return 5

    pre_duration = segment_duration(args.pre_duration, q_start_values, q_pre_values, args.max_joint_speed)
    place_duration = segment_duration(args.place_duration, q_pre_values, q_place_values, args.max_joint_speed)
    lower_duration = segment_duration(args.lower_duration, q_place_values, q_table_values, args.max_joint_speed)
    return_duration = segment_duration(args.return_duration, q_pre_values, q_start_values, args.max_joint_speed)
    print(f"pre_duration_s: {pre_duration:.2f}")
    print(f"place_duration_s: {place_duration:.2f}")
    print(f"lower_duration_s: {lower_duration:.2f}")
    print(f"return_duration_s: {return_duration:.2f}")

    if not args.execute:
        print("Dry run only. Add --execute --yes to move the robot.")
        return 0

    q_right_last = dict(q_right_start)
    exit_code = 0
    try:
        if args.manual_load:
            print("Stage 0: open hand for manual object loading.")
            write_hand_pose(hand_args_with_hold(args, args.manual_load_open_hold), args.hand_open_angle, "manual_load_open")
            print("Stage 0: close hand after manual object loading.")
            write_hand_pose(hand_args_with_hold(args, args.manual_load_close_hold), args.hand_grasp_angle, "manual_load_close")

        print("Stage 1: move to preplace target.")
        run_timed_phase(controller, pre_duration, q_left_hold, q_waist_hold, q_right_start, q_pre, 0.0, args.weight, args)
        q_right_last = dict(q_pre)
        if args.stop_after == "preplace":
            print("Debug stop after preplace target.")
            return 0

        print("Stage 2: move to place target above table.")
        run_timed_phase(controller, place_duration, q_left_hold, q_waist_hold, q_pre, q_place, args.weight, args.weight, args)
        q_right_last = dict(q_place)
        if args.stop_after == "place":
            print("Debug stop after place target.")
            return 0

        print("Stage 3: lower to table place target.")
        run_timed_phase(controller, lower_duration, q_left_hold, q_waist_hold, q_place, q_table, args.weight, args.weight, args)
        q_right_last = dict(q_table)
        if args.stop_after == "table-place":
            print("Debug stop after table place target.")
            return 0
        hold_group(controller, args.hold_before_release, q_left_hold, q_table, q_waist_hold, args.weight, args)

        if not args.skip_hand:
            print("Stage 4: open hand to release object on table.")
            write_hand_pose(hand_args(args), args.hand_open_angle, "table_release_open")
            hold_group(controller, args.hold_after_release, q_left_hold, q_table, q_waist_hold, args.weight, args)

        print("Stage 5: lift back to place target above table.")
        run_timed_phase(controller, lower_duration, q_left_hold, q_waist_hold, q_table, q_place, args.weight, args.weight, args)
        q_right_last = dict(q_place)

        print("Stage 6: lateral safety retreat back to preplace target.")
        run_timed_phase(controller, place_duration, q_left_hold, q_waist_hold, q_place, q_pre, args.weight, args.weight, args)
        q_right_last = dict(q_pre)

        if not args.skip_hand:
            print("Stage 7: close hand back to contracted preset.")
            write_hand_pose(hand_args(args), args.hand_grasp_angle, "table_release_close")
            hold_group(controller, args.hold_after_hand_close, q_left_hold, q_pre, q_waist_hold, args.weight, args)

        print("Stage 8: retract arm to start posture.")
        run_timed_phase(controller, return_duration, q_left_hold, q_waist_hold, q_pre, q_right_start, args.weight, args.weight, args)
        q_right_last = dict(q_right_start)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        exit_code = 130
    finally:
        print("Stage 9: release arm_sdk.")
        release_group(controller, q_left_hold, q_right_last, q_waist_hold, args.weight, args)
        print("arm_sdk released.")

    if exit_code == 0:
        announce_success(args)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
