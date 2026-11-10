#!/usr/bin/env python3
"""Fixed-point lateral place sequence for the real G1 right arm.

Sequence:
1. Raise the right arm to a joint-space side-lift posture.
2. Move right hand to PREPLACE_TARGET in pelvis frame.
3. Move the whole right hand to the final place target.
4. Open the RH56E2 hand to release the object.
5. Close the RH56E2 hand back to the contracted preset.
6. Retreat through PREPLACE_TARGET and the side-lift waypoint.
7. Return the right arm to its start posture and release arm_sdk.

The motion and hand-control details intentionally mirror
g1_right_arm_lateral_grasp_sequence.py, but this script releases the held object
before retreating from the receiving box.
"""

import argparse
import sys
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
VISION_DIR = THIS_DIR.parents[0] / "vision"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))

from g1_right_arm_lateral_grasp_sequence import (  # noqa: E402
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
    format_joint_delta_table,
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
from g1_right_arm_lateral_grasp_sequence_mjpg_force_thumb import (  # noqa: E402
    DEFAULT_HAND_SAFE_CLOSE_ANGLE,
    write_safe_close_once,
)
from g1_right_arm_body_ik_example import JOINT_LIMITS, right_hand_fk  # noqa: E402
from g1_box_place_target import (  # noqa: E402
    add_box_vision_arguments,
    estimate_box_place_targets,
)


DEFAULT_PLACE_TARGET = PREGRASP_TARGET + np.asarray([0.0, 0.10, 0.0], dtype=float)
# A full side-lift is defined in shoulder/elbow joint space: no forward
# shoulder pitch, no shoulder yaw, 90 degrees of right-shoulder abduction,
# and an almost straight elbow.  The previous -1.35 rad pose still left the
# hand visibly below shoulder level; -pi/2 is the flat lateral raise target.
DEFAULT_SIDE_LIFT_JOINTS = np.asarray([0.0, -1.57, 0.0, 0.0], dtype=float)
# Leave the thumb rotation unchanged during release.  Its angle is set by the
# grasp sequence; release should only open the four fingers.
DEFAULT_RELEASE_OPEN_ANGLE = [1000, 1000, 1000, 1000, 1000, -1]


def parse_place_target(values: Optional[List[float]], preplace_target: np.ndarray, left_distance: float) -> np.ndarray:
    if values is not None:
        return np.asarray(values, dtype=float)
    target = preplace_target.copy()
    target[1] += left_distance
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-point lateral right-arm place/release sequence.")
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
        help="Final release point in pelvis frame. Default: preplace target plus --left-distance in +Y.",
    )
    parser.add_argument(
        "--left-distance",
        type=float,
        default=float(DEFAULT_PLACE_TARGET[1] - PREGRASP_TARGET[1]),
        help="Pelvis-frame +Y lateral move from --preplace-target when --place-target is omitted.",
    )
    parser.add_argument(
        "--side-lift-joints",
        nargs=4,
        type=float,
        default=DEFAULT_SIDE_LIFT_JOINTS.tolist(),
        metavar=("SHOULDER_PITCH", "SHOULDER_ROLL", "SHOULDER_YAW", "ELBOW"),
        help="Joint-space side-lift posture before approaching the box, in radians.",
    )
    add_box_vision_arguments(parser)
    parser.add_argument("--execute", action="store_true", help="Actually move the arm and hand.")
    parser.add_argument("--yes", action="store_true", help="Confirm the robot, hand, and receiving box have clear space.")
    parser.add_argument("--weight", type=float, default=0.75)
    parser.add_argument("--allow-full-weight", action="store_true")
    parser.add_argument("--side-lift-duration", type=float, default=3.0)
    parser.add_argument("--pre-duration", type=float, default=6.0)
    parser.add_argument("--place-duration", type=float, default=3.0)
    parser.add_argument("--return-duration", type=float, default=5.0)
    parser.add_argument("--side-lift-return-duration", type=float, default=3.0)
    parser.add_argument("--hold-after-side-lift", type=float, default=0.3)
    parser.add_argument("--hold-before-release", type=float, default=0.5)
    parser.add_argument("--hold-after-release", type=float, default=0.8)
    parser.add_argument("--hold-after-hand-close", type=float, default=0.3)
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
    parser.add_argument("--return-max-joint-speed", type=float, default=None,
                        help="Return-segment speed limit; defaults to --max-joint-speed.")
    parser.add_argument("--announce-on-arm-return", action="store_true",
                        help="Start completion TTS when arm return ends, during control release.")
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
    parser.add_argument("--hand-open-angle", nargs=6, type=int, default=DEFAULT_RELEASE_OPEN_ANGLE)
    parser.add_argument(
        "--hand-grasp-angle",
        nargs=6,
        type=int,
        default=[0, 0, 0, 0, 0, 400],
        help="Contracted hand preset after releasing the object.",
    )
    parser.add_argument("--thumb-stage-channel", type=int, default=5)
    parser.add_argument("--thumb-rotate-speed", type=int, default=500)
    parser.add_argument("--thumb-rotate-force", type=int, default=2000)
    parser.add_argument("--thumb-rotate-hold", type=float, default=0.03)
    parser.add_argument("--thumb-rotate-step", type=int, default=40)
    parser.add_argument("--skip-hand", action="store_true", help="Run arm motion only; do not command RH56E2.")
    parser.add_argument("--announce-text", default="放置完成")
    parser.add_argument("--announce-speaker-id", type=int, default=0)
    parser.add_argument("--no-announce", action="store_true")
    return parser.parse_args()


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
    if not args.box_vision and args.place_target is None and args.left_distance < 0.0:
        print("--left-distance must be non-negative when --place-target is omitted.", file=sys.stderr)
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
            args.side_lift_duration,
            args.pre_duration,
            args.place_duration,
            args.return_duration,
            args.side_lift_return_duration,
            args.release_duration,
            args.hold_after_side_lift,
            args.hold_after_hand_close,
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

    side_lift_joints = np.asarray(args.side_lift_joints, dtype=float)
    if side_lift_joints.shape != (4,) or not np.all(np.isfinite(side_lift_joints)):
        print("Refusing side-lift posture: expected four finite joint angles.", file=sys.stderr)
        return 2
    preplace_target = np.asarray(args.preplace_target, dtype=float)
    box_estimate = None
    if args.box_vision:
        if args.place_target is not None:
            print("Refusing --place-target together with --box-vision; use --box-place-offset instead.", file=sys.stderr)
            return 2
        try:
            box_estimate = estimate_box_place_targets(args)
        except Exception as exc:
            print(f"Refusing dynamic box placement: {exc}", file=sys.stderr)
            return 7
        preplace_target = np.asarray(box_estimate["approach_target_pelvis_m"], dtype=float)
        place_target = np.asarray(box_estimate["place_target_pelvis_m"], dtype=float)
        print("box_vision_enabled: true")
        print("box_vision_estimate:", box_estimate)
        if not box_estimate["place_workspace_ok"] or not box_estimate["approach_workspace_ok"]:
            print(
                "dynamic box target is outside the conservative right-arm workspace; "
                "reposition the robot/box before moving the arm.",
                file=sys.stderr,
            )
            if not args.allow_box_outside_arm_workspace:
                return 8
    else:
        place_target = parse_place_target(args.place_target, preplace_target, args.left_distance)
    for label, target in (
        ("preplace", preplace_target),
        ("place", place_target),
    ):
        try:
            validate_target(target)
        except ValueError as exc:
            print(f"Refusing {label} target: {exc}", file=sys.stderr)
            return 2

    print("side_lift_joint_target_rad:", np.array2string(side_lift_joints, precision=4))
    print("preplace_target:", np.array2string(preplace_target, precision=4))
    print("final_place_target:", np.array2string(place_target, precision=4))
    print(f"computed_x_delta: {float(place_target[0] - preplace_target[0]):.4f}")
    print(f"computed_left_distance: {float(place_target[1] - preplace_target[1]):.4f}")

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

    q_side_lift_values = q_start_values.copy()
    for index, target_value in enumerate(side_lift_joints):
        joint = right_joints[index]
        lower, upper = JOINT_LIMITS[joint]
        if not lower <= float(target_value) <= upper:
            print(
                f"Refusing side-lift posture: joint {joint} target {target_value:.4f} "
                f"outside [{lower:.4f}, {upper:.4f}].",
                file=sys.stderr,
            )
            return 2
        q_side_lift_values[index] = float(target_value)
    q_side_lift = {joint: float(value) for joint, value in zip(right_joints, q_side_lift_values)}
    if not args.no_vertical_wrist:
        q_side_lift_values, q_side_lift = apply_wrist_targets(
            q_side_lift_values,
            q_side_lift,
            right_joints,
            args.wrist_roll_target,
            args.wrist_pitch_target,
            args.wrist_yaw_target,
        )
    preplace_ik, q_preplace_values, q_preplace = solve_right_q(
        preplace_target, waist_q, right_joints, q_side_lift_values, args.tolerance
    )
    if not args.no_vertical_wrist:
        q_preplace_values, q_preplace = apply_wrist_targets(
            q_preplace_values,
            q_preplace,
            right_joints,
            args.wrist_roll_target,
            args.wrist_pitch_target,
            args.wrist_yaw_target,
        )
    place_ik, q_place_values, q_place = solve_right_q(
        place_target, waist_q, right_joints, q_preplace_values, args.tolerance
    )
    if not args.no_vertical_wrist:
        q_place_values, q_place = apply_wrist_targets(
            q_place_values,
            q_place,
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

    side_lift_hand_xyz = right_hand_fk(q_side_lift_values, waist_q, right_joints)
    print("side_lift_hand_fk_pelvis_xyz:", np.array2string(side_lift_hand_xyz, precision=4))
    print("side_lift_right_goal:")
    print(format_joint_table(right_joints, q_side_lift_values))
    print("side_lift_right_delta:")
    print(format_joint_delta_table(right_joints, q_start_values, q_side_lift_values))
    print_plan_item("preplace", preplace_target, preplace_ik, q_preplace_values, q_side_lift_values, right_joints)
    print_command_position("preplace", preplace_target, q_preplace_values, waist_q, right_joints)
    print_plan_item("place", place_target, place_ik, q_place_values, q_preplace_values, right_joints)
    print_command_position("place", place_target, q_place_values, waist_q, right_joints)
    print("left_hold:")
    print(format_joint_table(LEFT_ARM_7DOF, [q_left_hold[joint] for joint in LEFT_ARM_7DOF]))
    print("waist_hold:")
    print(format_joint_table(WAIST_JOINTS, [q_waist_hold[joint] for joint in WAIST_JOINTS]))

    if not preplace_ik.success or not place_ik.success:
        print("Refusing execution because IK did not meet tolerance.", file=sys.stderr)
        return 4

    max_delta_side_lift = float(np.max(np.abs(q_side_lift_values - q_start_values)))
    max_delta_pre = float(np.max(np.abs(q_preplace_values - q_side_lift_values)))
    max_delta_place = float(np.max(np.abs(q_place_values - q_preplace_values)))
    max_delta = max(max_delta_side_lift, max_delta_pre, max_delta_place)
    print(f"max_side_lift_delta_rad: {max_delta_side_lift:.4f}")
    print(f"max_pre_delta_rad: {max_delta_pre:.4f}")
    print(f"max_place_delta_rad: {max_delta_place:.4f}")
    if max_delta > args.max_joint_delta:
        print(f"Refusing execution: max joint delta {max_delta:.3f} exceeds {args.max_joint_delta:.3f}.", file=sys.stderr)
        return 5

    side_lift_duration = segment_duration(
        args.side_lift_duration, q_start_values, q_side_lift_values, args.max_joint_speed
    )
    pre_duration = segment_duration(args.pre_duration, q_side_lift_values, q_preplace_values, args.max_joint_speed)
    place_duration = segment_duration(args.place_duration, q_preplace_values, q_place_values, args.max_joint_speed)
    return_speed = args.max_joint_speed if args.return_max_joint_speed is None else args.return_max_joint_speed
    if not np.isfinite(return_speed) or return_speed <= 0.0:
        print("Return speed must be finite and positive.", file=sys.stderr)
        return 2
    retreat_duration = segment_duration(args.place_duration, q_place_values, q_preplace_values, return_speed)
    return_duration = segment_duration(args.return_duration, q_preplace_values, q_side_lift_values, return_speed)
    side_lift_return_duration = segment_duration(
        args.side_lift_return_duration, q_side_lift_values, q_start_values, return_speed
    )
    print(f"side_lift_duration_s: {side_lift_duration:.2f}")
    print(f"pre_duration_s: {pre_duration:.2f}")
    print(f"place_duration_s: {place_duration:.2f}")
    print(f"return_duration_s: {return_duration:.2f}")
    print(f"side_lift_return_duration_s: {side_lift_return_duration:.2f}")
    print(f"retreat_duration_s: {retreat_duration:.2f}")
    print(f"return_max_joint_speed: {return_speed:.3f}")

    if not args.execute:
        print("Dry run only. Add --execute --yes to move the robot.")
        return 0

    q_right_last = dict(q_right_start)
    exit_code = 0
    announce_thread = None
    try:
        print("Stage 1: side-lift the right arm.")
        run_timed_phase(
            controller,
            side_lift_duration,
            q_left_hold,
            q_waist_hold,
            q_right_start,
            q_side_lift,
            0.0,
            args.weight,
            args,
        )
        q_right_last = dict(q_side_lift)
        if args.hold_after_side_lift > 0.0:
            hold_group(
                controller,
                args.hold_after_side_lift,
                q_left_hold,
                q_side_lift,
                q_waist_hold,
                args.weight,
                args,
            )

        print("Stage 2: move to preplace target from the side-lift waypoint.")
        run_timed_phase(
            controller,
            pre_duration,
            q_left_hold,
            q_waist_hold,
            q_side_lift,
            q_preplace,
            args.weight,
            args.weight,
            args,
        )
        q_right_last = dict(q_preplace)

        print("Stage 3: move whole hand to final place target.")
        run_timed_phase(
            controller, place_duration, q_left_hold, q_waist_hold, q_preplace, q_place, args.weight, args.weight, args
        )
        q_right_last = dict(q_place)
        hold_group(controller, args.hold_before_release, q_left_hold, q_place, q_waist_hold, args.weight, args)

        if not args.skip_hand:
            print("Stage 4: open hand to release object.")
            write_hand_pose(hand_args(args), args.hand_open_angle, "release_open")
            hold_group(controller, args.hold_after_release, q_left_hold, q_place, q_waist_hold, args.weight, args)

            print("Stage 4: safe-close hand with thumb at maximum rotation.")
            write_safe_close_once(args, DEFAULT_HAND_SAFE_CLOSE_ANGLE, "release_safe_close")
        hold_group(controller, args.hold_after_hand_close, q_left_hold, q_place, q_waist_hold, args.weight, args)

        print("Stage 5: lateral safety retreat back to preplace target.")
        run_timed_phase(
            controller, retreat_duration, q_left_hold, q_waist_hold, q_place, q_preplace, args.weight, args.weight, args
        )
        q_right_last = dict(q_preplace)

        print("Stage 6: retreat to the side-lift waypoint.")
        run_timed_phase(
            controller,
            return_duration,
            q_left_hold,
            q_waist_hold,
            q_preplace,
            q_side_lift,
            args.weight,
            args.weight,
            args,
        )
        q_right_last = dict(q_side_lift)

        print("Stage 7: retract arm from the side-lift waypoint to start posture.")
        run_timed_phase(
            controller,
            side_lift_return_duration,
            q_left_hold,
            q_waist_hold,
            q_side_lift,
            q_right_start,
            args.weight,
            args.weight,
            args,
        )
        q_right_last = dict(q_right_start)
        if args.announce_on_arm_return and not args.skip_hand and not args.no_announce:
            # Keep audio RPC latency out of the control-release loop.
            announce_thread = threading.Thread(target=announce_success, args=(args,), daemon=True)
            announce_thread.start()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        exit_code = 130
    finally:
        print("Stage 8: release arm_sdk.")
        release_group(controller, q_left_hold, q_right_last, q_waist_hold, args.weight, args)
        print("arm_sdk released.")

    if exit_code == 0:
        if announce_thread is not None:
            announce_thread.join()
        else:
            announce_success(args)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
