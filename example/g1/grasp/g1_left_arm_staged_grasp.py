#!/usr/bin/env python3
"""Move the G1 left arm through staged grasp waypoints, then close the hand."""

import argparse
import sys
import threading
import time

import numpy as np

from g1_left_arm_body_ik_then_hand import (
    ArmSdkController,
    G1JointIndex,
    DEFAULT_BASE_LINK_IN_PELVIS_RPY,
    DEFAULT_BASE_LINK_IN_PELVIS_XYZ,
    format_joint_table,
    hand_args,
    left_arm_joints,
    move_left_arm_to,
    release_arm_sdk_safely,
    solve_left_arm_goal,
    target_to_pelvis_frame,
    validate_arm_goal,
    validate_target,
    waist_joints,
    write_hand_pose,
)
from g1_right_arm_body_ik_example import motor_to_model_q
from g1_right_arm_body_ik_example import left_hand_fk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged left-arm grasp: pregrasp, align, approach, grasp, lift.")
    parser.add_argument("network_interface", nargs="?", help="DDS network interface, e.g. eth0")
    parser.add_argument("--target", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--target-frame", choices=("base_link", "body", "pelvis", "map"), default="pelvis")
    parser.add_argument("--base-link-in-pelvis-xyz", nargs=3, type=float, default=DEFAULT_BASE_LINK_IN_PELVIS_XYZ)
    parser.add_argument("--base-link-in-pelvis-rpy", nargs=3, type=float, default=DEFAULT_BASE_LINK_IN_PELVIS_RPY)
    parser.add_argument("--joint-signs", nargs=7, type=float, default=[1, 1, 1, 1, 1, 1, 1])
    parser.add_argument("--arm-dof", type=int, choices=(5, 7), default=7)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--duration", type=float, default=2.0, help="Base duration for each staged motion.")
    parser.add_argument("--approach-duration", type=float, default=2.5)
    parser.add_argument("--lift-duration", type=float, default=2.0)
    parser.add_argument("--hold-after-hand", type=float, default=1.0)
    parser.add_argument("--max-joint-speed", type=float, default=0.12)
    parser.add_argument("--max-joint-delta", type=float, default=2.80)
    parser.add_argument("--allow-large-motion", action="store_true")
    parser.add_argument("--arm-sdk-weight", type=float, default=0.45)
    parser.add_argument("--arm-sdk-ramp-in-ratio", type=float, default=0.25)
    parser.add_argument("--release-duration", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=30.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--wrist-roll-offset", type=float, default=0.0)
    parser.add_argument("--wrist-pitch-offset", type=float, default=-0.5)
    parser.add_argument("--wrist-yaw-offset", type=float, default=0.0)
    parser.add_argument("--pregrasp-offset", nargs=3, type=float, default=[-0.10, 0.06, 0.12])
    parser.add_argument("--align-offset", nargs=3, type=float, default=[-0.05, 0.04, 0.05])
    parser.add_argument("--approach-offset", nargs=3, type=float, default=[0.0, 0.02, 0.02])
    parser.add_argument("--lift-offset", nargs=3, type=float, default=[0.0, 0.02, 0.18])
    parser.add_argument(
        "--feedback-control",
        action="store_true",
        help="After each staged open-loop move, print FK feedback diagnostics. Does not override the main motion.",
    )
    parser.add_argument("--feedback-pos-threshold", type=float, default=0.035)
    parser.add_argument("--feedback-step-m", type=float, default=0.025)
    parser.add_argument("--feedback-log-period", type=float, default=0.5)
    parser.add_argument("--feedback-timeout-scale", type=float, default=1.8)
    parser.add_argument("--hand-ip", default="192.168.124.211")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--hand-unit-id", type=int, default=1)
    parser.add_argument("--hand-timeout", type=float, default=2.0)
    parser.add_argument("--hand-speed", type=int, default=200)
    parser.add_argument("--hand-force", type=int, default=1000)
    parser.add_argument("--hand-hold", type=float, default=3.0)
    parser.add_argument("--hand-read-before", action="store_true")
    parser.add_argument("--hand-read-after", action="store_true")
    parser.add_argument("--hand-open-angle", nargs=6, type=int, default=[1000, 1000, 1000, 1000, 1000, 0])
    parser.add_argument("--hand-grasp-angle", nargs=6, type=int, default=[0, 0, 0, 0, 0, 400])
    parser.add_argument("--skip-initial-hand-open", action="store_true")
    parser.add_argument("--no-return-after-drop", action="store_true")
    parser.add_argument("--no-release-on-exit", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def make_waypoints(object_target: np.ndarray, args: argparse.Namespace) -> list:
    return [
        ("pregrasp", object_target + np.asarray(args.pregrasp_offset, dtype=float), args.duration, 0.0),
        ("align", object_target + np.asarray(args.align_offset, dtype=float), args.duration, 0.0),
        ("approach", object_target + np.asarray(args.approach_offset, dtype=float), args.approach_duration, 0.0),
        ("lift", object_target + np.asarray(args.lift_offset, dtype=float), args.lift_duration, 0.0),
    ]


def plan_waypoint(name, target, duration, hold, waist_q, joints, signs, seed, tolerance, wrist_offsets):
    ik, motor_q, reached_rpy = solve_left_arm_goal(
        target=target,
        waist_q=waist_q,
        left_goal_joints=joints,
        signs=signs,
        seed_model_q=seed,
        tolerance=tolerance,
    )
    motor_q = np.asarray(motor_q, dtype=float)
    for index, joint in enumerate(joints):
        if joint == G1JointIndex.LeftWristRoll:
            motor_q[index] += wrist_offsets[0]
        elif joint == G1JointIndex.LeftWristPitch:
            motor_q[index] += wrist_offsets[1]
        elif joint == G1JointIndex.LeftWristYaw:
            motor_q[index] += wrist_offsets[2]
    return {
        "name": name,
        "target": target,
        "ik": ik,
        "q": motor_q,
        "rpy": reached_rpy,
        "duration": duration,
        "hold": hold,
    }


def print_plan(planned_moves):
    for move in planned_moves:
        ik = move["ik"]
        print(f"{move['name']} target:", np.array2string(move["target"], precision=3))
        print(f"{move['name']} reached:", np.array2string(ik.reached, precision=3))
        print(f"{move['name']} residual: {ik.residual:.4f} m after {ik.iterations} iterations ({ik.message})")
        print(f"{move['name']} command:")
        print(format_joint_table(move["joints"], move["q"]))


def current_left_fk(controller, joints, signs):
    current_positions = controller.current_positions(joints + waist_joints())
    motor_q = np.asarray([current_positions[joint] for joint in joints], dtype=float)
    waist_q = np.asarray(
        [
            current_positions[G1JointIndex.WaistYaw],
            current_positions[G1JointIndex.WaistRoll],
            current_positions[G1JointIndex.WaistPitch],
        ],
        dtype=float,
    )
    model_q = motor_to_model_q(motor_q, signs)
    return left_hand_fk(model_q, waist_q, joints), motor_q, waist_q


def apply_wrist_offsets_to_q(joints, motor_q, wrist_offsets):
    adjusted = np.asarray(motor_q, dtype=float).copy()
    for index, joint in enumerate(joints):
        if joint == G1JointIndex.LeftWristRoll:
            adjusted[index] += wrist_offsets[0]
        elif joint == G1JointIndex.LeftWristPitch:
            adjusted[index] += wrist_offsets[1]
        elif joint == G1JointIndex.LeftWristYaw:
            adjusted[index] += wrist_offsets[2]
    return adjusted


def move_left_arm_to_feedback(
    controller,
    joints,
    signs,
    target,
    initial_goal_q,
    duration,
    kp,
    kd,
    arm_sdk_weight,
    arm_sdk_ramp_in_ratio,
    ramp_in_weight,
    tolerance,
    feedback_pos_threshold,
    feedback_step_m,
    feedback_log_period,
    timeout_scale,
    wrist_offsets,
):
    start_time = time.monotonic()
    deadline = start_time + max(duration * timeout_scale, duration + 1.0)
    next_tick = start_time
    next_log = start_time
    last_goal_q = np.asarray(initial_goal_q, dtype=float)
    commanded_positions = {joint: float(value) for joint, value in zip(joints, last_goal_q)}

    while True:
        now = time.monotonic()
        current_xyz, current_motor_q, waist_q = current_left_fk(controller, joints, signs)
        error = np.asarray(target, dtype=float) - current_xyz
        error_norm = float(np.linalg.norm(error))
        if now >= next_log:
            print(
                "feedback "
                f"target={np.array2string(np.asarray(target), precision=3)} "
                f"current={np.array2string(current_xyz, precision=3)} "
                f"err={error_norm:.4f}"
            )
            next_log = now + max(feedback_log_period, 0.1)
        if error_norm <= feedback_pos_threshold:
            break
        if now >= deadline:
            print(
                f"feedback timeout: final FK error {error_norm:.4f} m exceeds "
                f"{feedback_pos_threshold:.4f} m",
                file=sys.stderr,
            )
            break

        step_norm = min(error_norm, feedback_step_m)
        sub_target = current_xyz + error * (step_norm / max(error_norm, 1e-9))
        seed = motor_to_model_q(current_motor_q, signs)
        ik, motor_q, _ = solve_left_arm_goal(
            target=sub_target,
            waist_q=waist_q,
            left_goal_joints=joints,
            signs=signs,
            seed_model_q=seed,
            tolerance=min(tolerance, feedback_pos_threshold),
        )
        if not ik.success:
            print(f"feedback warning: sub-target IK residual {ik.residual:.4f} m", file=sys.stderr)
        motor_q = apply_wrist_offsets_to_q(joints, motor_q, wrist_offsets)
        commanded_positions = {joint: float(value) for joint, value in zip(joints, motor_q)}

        if ramp_in_weight:
            elapsed = now - start_time
            ramp_denominator = max(arm_sdk_ramp_in_ratio, 1e-6)
            ramp_phase = min(1.0, (elapsed / max(duration, 1e-6)) / ramp_denominator)
            weight = arm_sdk_weight * (ramp_phase * ramp_phase * (3.0 - 2.0 * ramp_phase))
        else:
            weight = arm_sdk_weight
        controller.write_arm_command(commanded_positions, kp, kd, weight)
        next_tick += controller.control_dt
        time.sleep(max(0.0, next_tick - time.monotonic()))

    hold_deadline = time.monotonic() + 0.2
    while time.monotonic() < hold_deadline:
        controller.write_arm_command(commanded_positions, kp, kd, arm_sdk_weight)
        time.sleep(controller.control_dt)
    return commanded_positions


def print_fk_feedback(controller, joints, signs, target, label):
    current_xyz, _, _ = current_left_fk(controller, joints, signs)
    error = np.asarray(target, dtype=float) - current_xyz
    print(
        "feedback "
        f"{label} target={np.array2string(np.asarray(target), precision=3)} "
        f"current={np.array2string(current_xyz, precision=3)} "
        f"err={float(np.linalg.norm(error)):.4f}"
    )


def main() -> int:
    args = parse_args()
    if not args.yes:
        print("Refusing to move without --yes.", file=sys.stderr)
        return 1
    if args.arm_sdk_weight <= 0.0 or args.arm_sdk_weight > 1.0:
        print("Refusing arm command: --arm-sdk-weight must be in (0, 1].", file=sys.stderr)
        return 2
    if args.release_duration <= 0.0:
        print("Refusing arm command: --release-duration must be positive.", file=sys.stderr)
        return 2
    if any(value < 0 or value > 1000 for value in args.hand_open_angle + args.hand_grasp_angle):
        print("Refusing hand command: all hand angles must be in 0..1000.", file=sys.stderr)
        return 2

    try:
        object_target, base_link_target = target_to_pelvis_frame(
            target=np.asarray(args.target, dtype=float),
            target_frame=args.target_frame,
            base_link_in_pelvis_xyz=args.base_link_in_pelvis_xyz,
            base_link_in_pelvis_rpy=args.base_link_in_pelvis_rpy,
        )
        validate_target(object_target)
        for _, waypoint, _, _ in make_waypoints(object_target, args):
            validate_target(waypoint)
    except (RuntimeError, ValueError) as exc:
        print(f"Refusing target: {exc}", file=sys.stderr)
        return 2

    signs = np.asarray(args.joint_signs, dtype=float)
    signs = np.where(signs >= 0.0, 1.0, -1.0)
    joints = left_arm_joints(args.arm_dof)
    controller = ArmSdkController(args.network_interface, args.arm_dof, control_dt=0.02)
    controller.init()
    read_joints = joints + waist_joints()
    current_positions = controller.current_positions(read_joints)
    current_motor_q = np.asarray([current_positions[joint] for joint in joints], dtype=float)
    seed = motor_to_model_q(current_motor_q, signs)
    waist_q = np.asarray(
        [
            current_positions[G1JointIndex.WaistYaw],
            current_positions[G1JointIndex.WaistRoll],
            current_positions[G1JointIndex.WaistPitch],
        ],
        dtype=float,
    )
    wrist_offsets = (args.wrist_roll_offset, args.wrist_pitch_offset, args.wrist_yaw_offset)

    planned_moves = []
    for name, target, duration, hold in make_waypoints(object_target, args):
        move = plan_waypoint(name, target, duration, hold, waist_q, joints, signs, seed, args.tolerance, wrist_offsets)
        move["joints"] = joints
        planned_moves.append(move)
        seed = move["ik"].q_model

    print("Object target (pelvis):", np.array2string(object_target, precision=3))
    if base_link_target is not None:
        print("Base_link-frame object target:", np.array2string(base_link_target, precision=3))
    print_plan(planned_moves)

    for move in planned_moves:
        if not move["ik"].success:
            print(f"Refusing execution because {move['name']} IK did not meet tolerance.", file=sys.stderr)
            return 3
        limit_error = validate_arm_goal(joints, move["q"], f"{move['name']} execution")
        if limit_error:
            return limit_error

    previous_q = current_motor_q
    for move in planned_moves:
        delta = np.abs(move["q"][: len(previous_q)] - previous_q)
        move["max_delta"] = float(np.max(delta)) if len(delta) else 0.0
        if move["max_delta"] > args.max_joint_delta and not args.allow_large_motion:
            print(
                f"Refusing execution: {move['name']} max joint delta {move['max_delta']:.3f} rad exceeds "
                f"--max-joint-delta {args.max_joint_delta:.3f}.",
                file=sys.stderr,
            )
            return 5
        previous_q = move["q"]

    if not args.no_prompt:
        print("WARNING: left hand will open, arm will run staged grasp, then hand will close and lift.")
        input("Press Enter to continue, or Ctrl-C to abort...")

    for move in planned_moves:
        move["runtime_duration"] = max(move["duration"], move["max_delta"] / max(args.max_joint_speed, 1e-3))

    hold_positions = {joint: current_positions[joint] for joint in joints}
    exit_code = 0
    try:
        if not args.skip_initial_hand_open:
            print("Step 1: opening left hand.")
            write_hand_pose(hand_args(args), args.hand_open_angle, "initial_open")

        for step_index, move in enumerate(planned_moves[:3], start=2):
            print(f"Step {step_index}: moving to {move['name']}.")
            positions = {joint: float(value) for joint, value in zip(joints, move["q"])}
            move_left_arm_to(
                controller=controller,
                controlled_joints=joints,
                goal_positions=positions,
                duration=move["runtime_duration"],
                hold_time=move["hold"],
                kp=args.kp,
                kd=args.kd,
                arm_sdk_weight=args.arm_sdk_weight,
                arm_sdk_ramp_in_ratio=args.arm_sdk_ramp_in_ratio,
                ramp_in_weight=step_index == 2,
            )
            if args.feedback_control:
                print_fk_feedback(controller, joints, signs, move["target"], move["name"])
            hold_positions.update(positions)

        print("Step 5: closing left hand.")
        write_hand_pose(hand_args(args), args.hand_grasp_angle, "grasp")
        if args.hold_after_hand > 0.0:
            print(f"Holding after grasp for {args.hold_after_hand:.1f}s.")
            time.sleep(args.hold_after_hand)

        lift = planned_moves[3]
        print("Step 6: lifting object.")
        lift_positions = {joint: float(value) for joint, value in zip(joints, lift["q"])}
        move_left_arm_to(
            controller=controller,
            controlled_joints=joints,
            goal_positions=lift_positions,
            duration=lift["runtime_duration"],
            hold_time=0.5,
            kp=args.kp,
            kd=args.kd,
            arm_sdk_weight=args.arm_sdk_weight,
            arm_sdk_ramp_in_ratio=args.arm_sdk_ramp_in_ratio,
            ramp_in_weight=False,
        )
        if args.feedback_control:
            print_fk_feedback(controller, joints, signs, lift["target"], "lift")
        hold_positions.update(lift_positions)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        exit_code = 130
    finally:
        if args.no_release_on_exit:
            print("Skipping arm_sdk release on exit.")
        else:
            release_arm_sdk_safely(controller, hold_positions, args.kp, args.kd, args.arm_sdk_weight, args.release_duration)

    if exit_code == 0:
        print("Done.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
