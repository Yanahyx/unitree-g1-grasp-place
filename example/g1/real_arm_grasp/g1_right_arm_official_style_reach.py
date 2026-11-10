#!/usr/bin/env python3
"""Official-style G1 right-arm reach test.

This script is intentionally separate from example/g1/grasp. It follows the
Unitree arm_sdk routine more closely than the old grasp scripts:

* publish to rt/arm_sdk
* write the full upper-body command group every cycle
* right arm moves to an IK target
* left arm and waist hold their measured start positions
* dry-run by default; motion requires --execute --yes

It is a real-robot validation tool, not a full grasp pipeline.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

GRASP_DIR = Path(__file__).resolve().parents[1] / "grasp"
if str(GRASP_DIR) not in sys.path:
    sys.path.insert(0, str(GRASP_DIR))

from g1_right_arm_body_ik_example import (  # noqa: E402
    G1JointIndex,
    JOINT_NAMES,
    format_joint_delta_table,
    format_joint_table,
    matrix_to_rpy,
    model_to_motor_q,
    motor_to_model_q,
    right_arm_joints,
    right_hand_pose,
    solve_position_ik,
    validate_target,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402
from unitree_sdk2py.core.channel import ChannelPublisher  # noqa: E402
from unitree_sdk2py.core.channel import ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402


LEFT_ARM_7DOF = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
]

WAIST_JOINTS = [
    G1JointIndex.WaistYaw,
    G1JointIndex.WaistRoll,
    G1JointIndex.WaistPitch,
]

INTERFACE_CANDIDATES = ("eth1", "wlp0s20f3", "eth0", "enx68da73a54ce1")


def smoothstep(ratio: float) -> float:
    ratio = min(max(ratio, 0.0), 1.0)
    return ratio * ratio * (3.0 - 2.0 * ratio)


def smootherstep(ratio: float) -> float:
    ratio = min(max(ratio, 0.0), 1.0)
    return ratio * ratio * ratio * (ratio * (ratio * 6.0 - 15.0) + 10.0)


def interface_is_up(name: str) -> bool:
    operstate = Path("/sys/class/net") / name / "operstate"
    if not operstate.exists():
        return False
    return operstate.read_text(encoding="utf-8").strip() in ("up", "unknown")


def detect_interface(preferred: Optional[str]) -> Optional[str]:
    if preferred:
        return preferred
    names = {path.name for path in Path("/sys/class/net").iterdir()} if Path("/sys/class/net").exists() else set()
    for name in INTERFACE_CANDIDATES:
        if name in names and interface_is_up(name):
            return name
    for name in INTERFACE_CANDIDATES:
        if name in names:
            return name
    return None


def joint_positions(state: LowState_, joints: Iterable[int]) -> Dict[int, float]:
    return {joint: float(state.motor_state[joint].q) for joint in joints}


def joint_vector(state: LowState_, joints: Sequence[int]) -> np.ndarray:
    return np.asarray([float(state.motor_state[joint].q) for joint in joints], dtype=float)


def joint_velocity_abs_max(state: LowState_, joints: Sequence[int]) -> float:
    if not joints:
        return 0.0
    return float(max(abs(float(state.motor_state[joint].dq)) for joint in joints))


class OfficialStyleArmSdkController:
    def __init__(self, network_interface: Optional[str], control_dt: float):
        self.network_interface = network_interface
        self.control_dt = control_dt
        self.low_state: Optional[LowState_] = None
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.publisher: Optional[ChannelPublisher] = None
        self.subscriber: Optional[ChannelSubscriber] = None

    def init(self) -> None:
        if self.network_interface:
            ChannelFactoryInitialize(0, self.network_interface)
        else:
            ChannelFactoryInitialize(0)
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._low_state_handler, 10)

    def _low_state_handler(self, msg: LowState_) -> None:
        self.low_state = msg

    def wait_for_low_state(self, timeout: float) -> LowState_:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.low_state is not None:
                return self.low_state
            time.sleep(0.02)
        raise TimeoutError("timed out waiting for rt/lowstate")

    def write_group_command(
        self,
        q_left: Dict[int, float],
        q_right: Dict[int, float],
        q_waist: Dict[int, float],
        weight: float,
        kp_right: float,
        kd_right: float,
        kp_left: float,
        kd_left: float,
        kp_waist: float,
        kd_waist: float,
    ) -> None:
        assert self.publisher is not None
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = float(weight)
        self._write_joint_group(q_left, kp_left, kd_left)
        self._write_joint_group(q_right, kp_right, kd_right)
        self._write_joint_group(q_waist, kp_waist, kd_waist)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def _write_joint_group(self, targets: Dict[int, float], kp: float, kd: float) -> None:
        for joint, q_des in targets.items():
            cmd = self.low_cmd.motor_cmd[joint]
            cmd.mode = 1
            cmd.tau = 0.0
            cmd.q = float(q_des)
            cmd.dq = 0.0
            cmd.kp = float(kp)
            cmd.kd = float(kd)


def interpolate_positions(start: Dict[int, float], goal: Dict[int, float], ratio: float) -> Dict[int, float]:
    return {joint: start[joint] + ratio * (goal[joint] - start[joint]) for joint in start}


def run_timed_phase(
    controller: OfficialStyleArmSdkController,
    duration: float,
    q_left_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
    q_right_start: Dict[int, float],
    q_right_goal: Dict[int, float],
    weight_start: float,
    weight_goal: float,
    args: argparse.Namespace,
) -> None:
    start_time = time.monotonic()
    next_tick = start_time
    while True:
        now = time.monotonic()
        elapsed = now - start_time
        phase = min(max(elapsed / max(duration, 1e-6), 0.0), 1.0)
        ratio = smootherstep(phase)
        q_right = interpolate_positions(q_right_start, q_right_goal, ratio)
        weight = weight_start + smoothstep(phase) * (weight_goal - weight_start)
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
        if phase >= 1.0:
            break
        next_tick += controller.control_dt
        time.sleep(max(0.0, next_tick - time.monotonic()))


def hold_group(
    controller: OfficialStyleArmSdkController,
    hold_time: float,
    q_left_hold: Dict[int, float],
    q_right_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
    weight: float,
    args: argparse.Namespace,
) -> None:
    deadline = time.monotonic() + max(0.0, hold_time)
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
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
        time.sleep(max(0.0, next_tick - time.monotonic()))


def release_group(
    controller: OfficialStyleArmSdkController,
    q_left_hold: Dict[int, float],
    q_right_hold: Dict[int, float],
    q_waist_hold: Dict[int, float],
    start_weight: float,
    args: argparse.Namespace,
) -> None:
    steps = max(1, int(args.release_duration / controller.control_dt))
    for step in range(steps):
        ratio = 1.0 - (step + 1) / steps
        weight = max(0.0, start_weight * ratio)
        controller.write_group_command(
            q_left=q_left_hold,
            q_right=q_right_hold,
            q_waist=q_waist_hold,
            weight=weight,
            kp_right=args.kp_right * ratio,
            kd_right=args.kd_right * ratio,
            kp_left=args.kp_left * ratio,
            kd_left=args.kd_left * ratio,
            kp_waist=args.kp_waist * ratio,
            kd_waist=args.kd_waist * ratio,
        )
        time.sleep(controller.control_dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-style G1 right-arm reach test.")
    parser.add_argument("network_interface", nargs="?", default=None, help="DDS interface, e.g. eth1.")
    parser.add_argument("--target", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--execute", action="store_true", help="Actually publish rt/arm_sdk commands.")
    parser.add_argument("--yes", action="store_true", help="Confirm the robot has clear space.")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--return-duration", type=float, default=4.0)
    parser.add_argument("--release-duration", type=float, default=1.0)
    parser.add_argument("--no-return", action="store_true")
    parser.add_argument("--weight", type=float, default=0.55, help="arm_sdk enable weight. Official examples use 1.0.")
    parser.add_argument("--allow-full-weight", action="store_true")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.execute and not args.yes:
        print("Refusing to move without --yes.", file=sys.stderr)
        return 1
    if args.weight <= 0.0 or args.weight > 1.0:
        print("--weight must be in (0, 1].", file=sys.stderr)
        return 2
    if args.weight > 0.70 and not args.allow_full_weight:
        print("Refusing --weight above 0.70 without --allow-full-weight.", file=sys.stderr)
        return 2
    if min(args.duration, args.hold, args.return_duration, args.release_duration) < 0.0:
        print("Durations must be non-negative.", file=sys.stderr)
        return 2

    target = np.asarray(args.target, dtype=float)
    try:
        validate_target(target)
    except ValueError as exc:
        print(f"Refusing target: {exc}", file=sys.stderr)
        return 2

    network_interface = detect_interface(args.network_interface)
    print(f"network_interface: {network_interface}")
    controller = OfficialStyleArmSdkController(network_interface, control_dt=0.02)
    controller.init()
    state = controller.wait_for_low_state(5.0)

    right_joints = right_arm_joints(7)
    all_arms = LEFT_ARM_7DOF + right_joints
    arm_dq_max = joint_velocity_abs_max(state, all_arms)
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
    current_motor_q = joint_vector(state, right_joints)
    waist_q = joint_vector(state, WAIST_JOINTS)
    signs = np.ones(7)
    seed_model_q = motor_to_model_q(current_motor_q, signs)

    ik = solve_position_ik(
        target=target,
        waist_q=waist_q,
        joint_indices=right_joints,
        signs=signs,
        seed=seed_model_q,
        tolerance=args.tolerance,
    )
    q_right_goal_values = model_to_motor_q(ik.q_model, signs)
    q_right_goal = {joint: float(value) for joint, value in zip(right_joints, q_right_goal_values)}
    reached_pose = right_hand_pose(ik.q_model, waist_q, right_joints)
    reached_rpy = matrix_to_rpy(reached_pose[:3, :3])
    max_delta = float(np.max(np.abs(q_right_goal_values - current_motor_q)))
    motion_duration = max(args.duration, max_delta / max(args.max_joint_speed, 1e-6))

    print("target_pelvis_xyz:", np.array2string(target, precision=4))
    print("ik_success:", ik.success)
    print(f"ik_residual_m: {ik.residual:.4f}")
    print("ik_reached_xyz:", np.array2string(ik.reached, precision=4))
    print("ik_reached_rpy:", np.array2string(reached_rpy, precision=4))
    print(f"max_right_joint_delta_rad: {max_delta:.4f}")
    print(f"motion_duration_s: {motion_duration:.2f}")
    print("right_start:")
    print(format_joint_table(right_joints, current_motor_q))
    print("right_goal:")
    print(format_joint_table(right_joints, q_right_goal_values))
    print("right_delta:")
    print(format_joint_delta_table(right_joints, current_motor_q, q_right_goal_values))
    print("left_hold:")
    print(format_joint_table(LEFT_ARM_7DOF, [q_left_hold[joint] for joint in LEFT_ARM_7DOF]))
    print("waist_hold:")
    print(format_joint_table(WAIST_JOINTS, [q_waist_hold[joint] for joint in WAIST_JOINTS]))

    if not ik.success:
        print("Refusing execution because IK did not meet tolerance.", file=sys.stderr)
        return 4
    if max_delta > args.max_joint_delta:
        print(
            f"Refusing execution: max joint delta {max_delta:.3f} exceeds {args.max_joint_delta:.3f}.",
            file=sys.stderr,
        )
        return 5
    if not args.execute:
        print("Dry run only. Add --execute --yes to move the robot.")
        return 0

    q_right_last = dict(q_right_start)
    exit_code = 0
    try:
        print("Stage 1: official-style full-group ramp to right-arm target.")
        run_timed_phase(
            controller,
            motion_duration,
            q_left_hold,
            q_waist_hold,
            q_right_start,
            q_right_goal,
            0.0,
            args.weight,
            args,
        )
        q_right_last = dict(q_right_goal)
        if args.hold > 0.0:
            print(f"Stage 2: hold target for {args.hold:.2f}s.")
            hold_group(controller, args.hold, q_left_hold, q_right_goal, q_waist_hold, args.weight, args)
        if not args.no_return and args.return_duration > 0.0:
            print("Stage 3: return right arm to start.")
            run_timed_phase(
                controller,
                args.return_duration,
                q_left_hold,
                q_waist_hold,
                q_right_goal,
                q_right_start,
                args.weight,
                args.weight,
                args,
            )
            q_right_last = dict(q_right_start)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        exit_code = 130
    finally:
        print("Stage 4: release arm_sdk.")
        release_group(controller, q_left_hold, q_right_last, q_waist_hold, args.weight, args)
        print("arm_sdk released.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
