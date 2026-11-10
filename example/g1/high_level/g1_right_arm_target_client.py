#!/usr/bin/env python3
"""Safer G1 right-arm target client with an action-style API.

This script keeps the old IK scripts untouched.  It still uses rt/arm_sdk
internally because the public Python SDK does not expose a task-space arm RPC,
but it wraps that low-level channel behind a small high-level client and adds
guards that are important while the robot is standing:

* dry-run by default
* workspace and IK error checks
* joint delta and joint speed limits
* gradual arm_sdk weight ramp
* no waist commands
* modest left-arm hold only
* release in finally on Ctrl-C or errors
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


Joint = Tuple[str, int]

RIGHT_ARM: List[Joint] = [
    ("right_shoulder_pitch", 22),
    ("right_shoulder_roll", 23),
    ("right_shoulder_yaw", 24),
    ("right_elbow", 25),
    ("right_wrist_roll", 26),
    ("right_wrist_pitch", 27),
    ("right_wrist_yaw", 28),
]

LEFT_ARM: List[Joint] = [
    ("left_shoulder_pitch", 15),
    ("left_shoulder_roll", 16),
    ("left_shoulder_yaw", 17),
    ("left_elbow", 18),
    ("left_wrist_roll", 19),
    ("left_wrist_pitch", 20),
    ("left_wrist_yaw", 21),
]

WAIST_JOINTS = [12, 13, 14]
ARM_SDK_WEIGHT_JOINT = 29

RIGHT_ARM_CHAIN = [
    ([0.0039563, -0.10021, 0.23778], [-0.27931, 5.4949e-05, 0.00019159], [0, 1, 0], [-3.0892, 2.6704]),
    ([0, -0.038, -0.013831], [0.27925, 0, 0], [1, 0, 0], [-2.2515, 1.5882]),
    ([0, -0.00624, -0.1032], [0, 0, 0], [0, 0, 1], [-2.618, 2.618]),
    ([0.015783, 0, -0.080518], [0, 0, 0], [0, 1, 0], [-1.0472, 2.0944]),
    ([0.100, -0.00188791, -0.010], [0, 0, 0], [1, 0, 0], [-1.972222054, 1.972222054]),
    ([0.038, 0, 0], [0, 0, 0], [0, 1, 0], [-1.614429558, 1.614429558]),
    ([0.046, 0, 0], [0, 0, 0], [0, 0, 1], [-1.614429558, 1.614429558]),
]

PALM_FIXED_XYZ = np.array([0.0415, -0.003, 0.0], dtype=float)
DEFAULT_WORKSPACE = (0.10, 0.42, -0.35, 0.12, 0.05, 0.45)


def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def axis_angle(axis, q):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(q), math.sin(q)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def transform(xyz, rot):
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


def fk(q):
    t = np.eye(4)
    for qi, (xyz, rpy, axis, _) in zip(q, RIGHT_ARM_CHAIN):
        t = t @ transform(np.array(xyz, dtype=float), rpy_matrix(rpy))
        t = t @ transform(np.zeros(3), axis_angle(axis, qi))
    return t @ transform(PALM_FIXED_XYZ, np.eye(3))


def palm_position(q):
    return fk(q)[:3, 3]


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def joint_positions(low_state, joints: Sequence[Joint]):
    return np.array([low_state.motor_state[index].q for _, index in joints], dtype=float)


def joint_velocities(low_state, joints: Sequence[Joint]):
    return np.array([low_state.motor_state[index].dq for _, index in joints], dtype=float)


def write_joint_group(cmd, joints: Sequence[Joint], q_values, kp, kd):
    for (_, index), q in zip(joints, q_values):
        cmd.motor_cmd[index].tau = 0.0
        cmd.motor_cmd[index].q = float(q)
        cmd.motor_cmd[index].dq = 0.0
        cmd.motor_cmd[index].kp = float(kp)
        cmd.motor_cmd[index].kd = float(kd)


def solve_ik(q_start, target_xyz):
    lower = np.array([limit[0] for _, _, _, limit in RIGHT_ARM_CHAIN], dtype=float)
    upper = np.array([limit[1] for _, _, _, limit in RIGHT_ARM_CHAIN], dtype=float)
    q_start = np.clip(q_start, lower, upper)

    def residual(q):
        return np.concatenate([palm_position(q) - target_xyz, 0.03 * (q - q_start)])

    result = least_squares(
        residual,
        q_start,
        bounds=(lower, upper),
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
        max_nfev=300,
    )
    error = float(np.linalg.norm(palm_position(result.x) - target_xyz))
    return result.x, error, result


class LowStateReader:
    def __init__(self):
        self.low_state = None

    def init(self):
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._handler, 10)

    def _handler(self, msg):
        self.low_state = msg

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.low_state is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.low_state is None:
            raise RuntimeError("No rt/lowstate received before timeout.")
        return self.low_state


@dataclass
class TargetPlan:
    target_xyz: np.ndarray
    q_right_start: np.ndarray
    q_right_goal: np.ndarray
    q_left_hold: np.ndarray
    palm_start_xyz: np.ndarray
    palm_goal_xyz: np.ndarray
    ik_error: float
    duration: float
    max_joint_delta: float


class G1RightArmTargetClient:
    """Action-style wrapper around rt/arm_sdk right-arm target control."""

    def __init__(
        self,
        state_timeout=3.0,
        max_ik_error=0.025,
        max_joint_delta=0.65,
        max_joint_speed=0.35,
        workspace=DEFAULT_WORKSPACE,
        arm_sdk_weight=0.55,
        right_kp=28.0,
        right_kd=1.4,
        left_hold_kp=18.0,
        left_hold_kd=1.0,
        rate=50.0,
    ):
        self.state_timeout = state_timeout
        self.max_ik_error = max_ik_error
        self.max_joint_delta = max_joint_delta
        self.max_joint_speed = max_joint_speed
        self.workspace = workspace
        self.arm_sdk_weight = arm_sdk_weight
        self.right_kp = right_kp
        self.right_kd = right_kd
        self.left_hold_kp = left_hold_kp
        self.left_hold_kd = left_hold_kd
        self.rate = rate

        self.reader = LowStateReader()
        self.publisher = None
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.last_plan = None
        self.last_q_right = None
        self.last_q_left = None

    def Init(self):
        self.reader.init()
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()

    def _check_workspace(self, target_xyz):
        xmin, xmax, ymin, ymax, zmin, zmax = self.workspace
        x, y, z = target_xyz
        if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
            raise ValueError(
                "Target outside workspace: "
                f"x in [{xmin}, {xmax}], y in [{ymin}, {ymax}], z in [{zmin}, {zmax}]"
            )

    def _check_current_motion(self, low_state):
        arm_dq = np.concatenate(
            [joint_velocities(low_state, RIGHT_ARM), joint_velocities(low_state, LEFT_ARM)]
        )
        waist_dq = np.array([low_state.motor_state[i].dq for i in WAIST_JOINTS], dtype=float)
        if np.max(np.abs(arm_dq)) > 0.8:
            raise RuntimeError("Arm joints are already moving too fast; refusing to take over.")
        if np.max(np.abs(waist_dq)) > 0.5:
            raise RuntimeError("Waist joints are moving; refusing to mix arm_sdk into standing balance.")

    def PlanTarget(self, target_xyz, requested_duration):
        target_xyz = np.asarray(target_xyz, dtype=float)
        self._check_workspace(target_xyz)

        low_state = self.reader.wait(self.state_timeout)
        self._check_current_motion(low_state)

        q_right_start = joint_positions(low_state, RIGHT_ARM)
        q_left_hold = joint_positions(low_state, LEFT_ARM)
        q_right_goal, ik_error, result = solve_ik(q_right_start, target_xyz)

        palm_start = palm_position(q_right_start)
        palm_goal = palm_position(q_right_goal)
        max_delta = float(np.max(np.abs(q_right_goal - q_right_start)))
        duration = max(float(requested_duration), max_delta / self.max_joint_speed)

        if ik_error > self.max_ik_error:
            raise RuntimeError(f"IK error {ik_error:.4f} m exceeds {self.max_ik_error:.4f} m.")
        if max_delta > self.max_joint_delta:
            raise RuntimeError(f"Max joint delta {max_delta:.3f} rad exceeds {self.max_joint_delta:.3f} rad.")
        if not result.success:
            raise RuntimeError(f"IK solver failed: {result.message}")

        plan = TargetPlan(
            target_xyz=target_xyz,
            q_right_start=q_right_start,
            q_right_goal=q_right_goal,
            q_left_hold=q_left_hold,
            palm_start_xyz=palm_start,
            palm_goal_xyz=palm_goal,
            ik_error=ik_error,
            duration=duration,
            max_joint_delta=max_delta,
        )
        self.last_plan = plan
        self.last_q_right = q_right_start.copy()
        self.last_q_left = q_left_hold.copy()
        return plan

    def ExecuteTarget(self, plan, hold_duration=1.0, return_to_start=True):
        if self.publisher is None:
            raise RuntimeError("Client not initialized. Call Init() first.")

        dt = 1.0 / self.rate
        start_time = time.monotonic()
        total = plan.duration + max(0.0, hold_duration)
        if return_to_start:
            total += plan.duration
        next_tick = start_time

        try:
            while True:
                now = time.monotonic()
                elapsed = now - start_time
                if elapsed > total:
                    break

                if now < next_tick:
                    time.sleep(min(next_tick - now, 0.005))
                    continue
                next_tick += dt

                if elapsed < plan.duration:
                    phase = elapsed / plan.duration
                    r = smootherstep(phase)
                    q_right = (1.0 - r) * plan.q_right_start + r * plan.q_right_goal
                    weight = self.arm_sdk_weight * smootherstep(min(1.0, phase / 0.25))
                elif elapsed < plan.duration + hold_duration:
                    q_right = plan.q_right_goal
                    weight = self.arm_sdk_weight
                else:
                    phase = (elapsed - plan.duration - hold_duration) / plan.duration
                    r = smootherstep(phase)
                    q_right = (1.0 - r) * plan.q_right_goal + r * plan.q_right_start
                    weight = self.arm_sdk_weight

                self._write_command(weight, q_right, plan.q_left_hold)
                self.last_q_right = q_right.copy()
                self.last_q_left = plan.q_left_hold.copy()
        finally:
            self.Release()

    def Release(self):
        if self.publisher is None:
            return
        if self.last_q_right is None or self.last_q_left is None:
            return

        dt = 1.0 / self.rate
        for step in range(60):
            ratio = 1.0 - (step + 1) / 60.0
            weight = self.arm_sdk_weight * ratio
            self._write_command(
                weight,
                self.last_q_right,
                self.last_q_left,
                right_kp=self.right_kp * ratio,
                right_kd=self.right_kd * ratio,
                left_kp=self.left_hold_kp * ratio,
                left_kd=self.left_hold_kd * ratio,
            )
            time.sleep(dt)

    def _write_command(
        self,
        weight,
        q_right,
        q_left,
        right_kp=None,
        right_kd=None,
        left_kp=None,
        left_kd=None,
    ):
        right_kp = self.right_kp if right_kp is None else right_kp
        right_kd = self.right_kd if right_kd is None else right_kd
        left_kp = self.left_hold_kp if left_kp is None else left_kp
        left_kd = self.left_hold_kd if left_kd is None else left_kd

        self.cmd.motor_cmd[ARM_SDK_WEIGHT_JOINT].q = float(weight)
        write_joint_group(self.cmd, RIGHT_ARM, q_right, right_kp, right_kd)
        write_joint_group(self.cmd, LEFT_ARM, q_left, left_kp, left_kd)

        # Intentionally do not write WaistYaw/WaistRoll/WaistPitch.
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.publisher.Write(self.cmd)


def print_plan(plan):
    print("plan:")
    print(f"  target_xyz: {plan.target_xyz.tolist()}")
    print(f"  palm_start_xyz: {plan.palm_start_xyz.tolist()}")
    print(f"  palm_goal_xyz: {plan.palm_goal_xyz.tolist()}")
    print(f"  ik_error_m: {plan.ik_error:.4f}")
    print(f"  max_joint_delta_rad: {plan.max_joint_delta:.3f}")
    print(f"  duration_s: {plan.duration:.2f}")
    print("  q_right_goal:")
    for (name, _), q in zip(RIGHT_ARM, plan.q_right_goal):
        print(f"    {name}: {q:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Safer high-level-style G1 right-arm target client.")
    parser.add_argument("interface", help="Network interface connected to the robot, e.g. enx68da73a54ce1")
    parser.add_argument("--x", type=float, required=True, help="Target palm x in torso_link frame, meters.")
    parser.add_argument("--y", type=float, required=True, help="Target palm y in torso_link frame, meters.")
    parser.add_argument("--z", type=float, required=True, help="Target palm z in torso_link frame, meters.")
    parser.add_argument("--duration", type=float, default=4.0, help="Requested move duration in seconds.")
    parser.add_argument("--hold-duration", type=float, default=1.0, help="Target hold duration in seconds.")
    parser.add_argument("--no-return", action="store_true", help="Do not return to the starting right-arm pose.")
    parser.add_argument("--execute", action="store_true", help="Actually publish rt/arm_sdk commands.")
    parser.add_argument("--yes", action="store_true", help="Confirm clear space, stable standing, and an operator at E-stop.")
    parser.add_argument("--arm-sdk-weight", type=float, default=0.55, help="Max arm_sdk blend weight, default 0.55.")
    parser.add_argument("--allow-full-weight", action="store_true", help="Allow --arm-sdk-weight above 0.70.")
    parser.add_argument("--max-ik-error", type=float, default=0.025)
    parser.add_argument("--max-joint-delta", type=float, default=0.65)
    parser.add_argument("--max-joint-speed", type=float, default=0.35)
    parser.add_argument(
        "--workspace",
        nargs=6,
        type=float,
        default=list(DEFAULT_WORKSPACE),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Allowed target workspace in torso_link frame, meters.",
    )
    parser.add_argument("--right-kp", type=float, default=28.0)
    parser.add_argument("--right-kd", type=float, default=1.4)
    parser.add_argument("--left-hold-kp", type=float, default=18.0)
    parser.add_argument("--left-hold-kd", type=float, default=1.0)
    parser.add_argument("--state-timeout", type=float, default=3.0)
    args = parser.parse_args()

    if args.duration <= 0.0 or args.hold_duration < 0.0:
        raise SystemExit("--duration must be positive and --hold-duration cannot be negative.")
    if args.arm_sdk_weight <= 0.0 or args.arm_sdk_weight > 1.0:
        raise SystemExit("--arm-sdk-weight must be in (0, 1].")
    if args.arm_sdk_weight > 0.70 and not args.allow_full_weight:
        raise SystemExit("Refusing arm_sdk weight above 0.70 without --allow-full-weight.")
    if args.execute and not args.yes:
        raise SystemExit("Refusing --execute without --yes.")

    ChannelFactoryInitialize(0, args.interface)
    client = G1RightArmTargetClient(
        state_timeout=args.state_timeout,
        max_ik_error=args.max_ik_error,
        max_joint_delta=args.max_joint_delta,
        max_joint_speed=args.max_joint_speed,
        workspace=tuple(args.workspace),
        arm_sdk_weight=args.arm_sdk_weight,
        right_kp=args.right_kp,
        right_kd=args.right_kd,
        left_hold_kp=args.left_hold_kp,
        left_hold_kd=args.left_hold_kd,
    )
    client.Init()

    target = np.array([args.x, args.y, args.z], dtype=float)
    plan = client.PlanTarget(target, args.duration)
    print_plan(plan)

    if not args.execute:
        print("dry_run: no motion command published. Add --execute --yes to move.")
        return

    client.ExecuteTarget(plan, hold_duration=args.hold_duration, return_to_start=not args.no_return)
    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
