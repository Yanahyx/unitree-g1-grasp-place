#!/usr/bin/env python3
"""Conservative left-arm joint response probe for G1 arm_sdk."""

import argparse
import sys
import time

import numpy as np

from g1_right_arm_body_ik_example import ArmSdkController, G1JointIndex, JOINT_NAMES


LEFT_JOINTS = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
]

JOINT_ALIASES = {
    "sp": G1JointIndex.LeftShoulderPitch,
    "shoulder_pitch": G1JointIndex.LeftShoulderPitch,
    "sr": G1JointIndex.LeftShoulderRoll,
    "shoulder_roll": G1JointIndex.LeftShoulderRoll,
    "sy": G1JointIndex.LeftShoulderYaw,
    "shoulder_yaw": G1JointIndex.LeftShoulderYaw,
    "elbow": G1JointIndex.LeftElbow,
    "wr": G1JointIndex.LeftWristRoll,
    "wrist_roll": G1JointIndex.LeftWristRoll,
    "wp": G1JointIndex.LeftWristPitch,
    "wrist_pitch": G1JointIndex.LeftWristPitch,
    "wy": G1JointIndex.LeftWristYaw,
    "wrist_yaw": G1JointIndex.LeftWristYaw,
}


def smoothstep(ratio: float) -> float:
    ratio = min(max(ratio, 0.0), 1.0)
    return ratio * ratio * (3.0 - 2.0 * ratio)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe one left-arm joint with low-weight arm_sdk commands.")
    parser.add_argument("interface", help="DDS network interface, e.g. enx68da73a54ce1")
    parser.add_argument("--joint", choices=sorted(JOINT_ALIASES), default="sp")
    parser.add_argument("--delta", type=float, default=-0.05, help="Commanded joint delta in rad.")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--hold", type=float, default=3.0)
    parser.add_argument("--return-duration", type=float, default=3.0)
    parser.add_argument("--weight", type=float, default=0.45)
    parser.add_argument("--kp", type=float, default=30.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--period", type=float, default=0.25)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--no-return", action="store_true")
    return parser.parse_args()


def read_q(controller):
    positions = controller.current_positions(LEFT_JOINTS)
    return np.asarray([positions[joint] for joint in LEFT_JOINTS], dtype=float)


def print_feedback(label, q_current, q_target, probe_joint):
    index = LEFT_JOINTS.index(probe_joint)
    diff = q_current - q_target
    print(
        f"{label}: {JOINT_NAMES[probe_joint]} "
        f"lowstate={q_current[index]:+.4f} target={q_target[index]:+.4f} "
        f"diff={diff[index]:+.4f} max_abs_diff={float(np.max(np.abs(diff))):.4f}",
        flush=True,
    )


def command_trajectory(controller, q_start, q_goal, duration, weight, kp, kd, period, label, probe_joint):
    start_time = time.monotonic()
    next_log = start_time
    while True:
        now = time.monotonic()
        elapsed = now - start_time
        ratio = smoothstep(elapsed / max(duration, 1e-6))
        q_cmd = (1.0 - ratio) * q_start + ratio * q_goal
        controller.write_arm_command(
            {joint: float(value) for joint, value in zip(LEFT_JOINTS, q_cmd)},
            kp,
            kd,
            weight * smoothstep(min(1.0, elapsed / max(duration * 0.25, 1e-6))),
        )
        if now >= next_log:
            print_feedback(label, read_q(controller), q_cmd, probe_joint)
            next_log = now + period
        if elapsed >= duration:
            break
        time.sleep(controller.control_dt)


def hold_position(controller, q_target, hold, weight, kp, kd, period, label, probe_joint):
    deadline = time.monotonic() + max(hold, 0.0)
    next_log = time.monotonic()
    command = {joint: float(value) for joint, value in zip(LEFT_JOINTS, q_target)}
    while time.monotonic() < deadline:
        controller.write_arm_command(command, kp, kd, weight)
        now = time.monotonic()
        if now >= next_log:
            print_feedback(label, read_q(controller), q_target, probe_joint)
            next_log = now + period
        time.sleep(controller.control_dt)


def release(controller, q_hold, weight, kp, kd):
    command = {joint: float(value) for joint, value in zip(LEFT_JOINTS, q_hold)}
    for step in range(50):
        ratio = 1.0 - (step + 1) / 50.0
        controller.write_arm_command(command, kp * ratio, kd * ratio, weight * ratio)
        time.sleep(controller.control_dt)


def main():
    args = parse_args()
    if not args.yes:
        print("Refusing to move without --yes.", file=sys.stderr)
        return 1
    if args.weight <= 0.0 or args.weight > 0.70:
        print("Refusing probe: --weight must be in (0, 0.70].", file=sys.stderr)
        return 2
    if abs(args.delta) > 0.12:
        print("Refusing probe: abs(--delta) must be <= 0.12 rad.", file=sys.stderr)
        return 2

    probe_joint = JOINT_ALIASES[args.joint]
    controller = ArmSdkController(args.interface, 7, control_dt=0.02)
    controller.init()
    time.sleep(0.3)

    q_start = read_q(controller)
    q_goal = q_start.copy()
    q_goal[LEFT_JOINTS.index(probe_joint)] += args.delta
    print("probe_joint:", JOINT_NAMES[probe_joint], flush=True)
    print("q_start:", q_start.tolist(), flush=True)
    print("q_goal:", q_goal.tolist(), flush=True)

    try:
        command_trajectory(controller, q_start, q_goal, args.duration, args.weight, args.kp, args.kd, args.period, "move", probe_joint)
        hold_position(controller, q_goal, args.hold, args.weight, args.kp, args.kd, args.period, "hold", probe_joint)
        if not args.no_return:
            command_trajectory(
                controller,
                read_q(controller),
                q_start,
                args.return_duration,
                args.weight,
                args.kp,
                args.kd,
                args.period,
                "return",
                probe_joint,
            )
    finally:
        release(controller, read_q(controller), args.weight, args.kp, args.kd)
        print("arm_sdk released.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
