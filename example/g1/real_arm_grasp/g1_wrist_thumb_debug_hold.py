#!/usr/bin/env python3
"""Hold the right arm at a pregrasp point and interactively tune wrist/thumb.

This script is for real-robot debugging. It moves the right hand to one
pregrasp target, keeps publishing the full upper-body arm_sdk group, and lets
the operator try different wrist angles and RH56E2 thumb values.
"""

import argparse
import socket
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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
    validate_target,
)
from g1_right_arm_official_style_reach import (  # noqa: E402
    LEFT_ARM_7DOF,
    WAIST_JOINTS,
    OfficialStyleArmSdkController,
    detect_interface,
    joint_positions,
    joint_vector,
    joint_velocity_abs_max,
    release_group,
    run_timed_phase,
)
from rh56e2_hand_only import write_hand_pose  # noqa: E402
from rh56e2_modbus_probe import ModbusTcpClient, REG, fmt  # noqa: E402
from rh56e2_modbus_move_once import read_state, write_registers  # noqa: E402


DEFAULT_PREGRASP_TARGET = [0.30, -0.35, 0.15]
DEFAULT_HAND_ANGLE = [1000, 1000, 1000, 1000, 1000, -1]
DEFAULT_GRASP_ANGLE = [0, 0, 0, 0, 400, -1]
POS_SET_ADDR = 1474
POS_ACT_ADDR = 1534
CURRENT_ADDR = 1594


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug right wrist pitch and RH56E2 thumb values at pregrasp.")
    parser.add_argument("network_interface", nargs="?", default=None, help="DDS interface, e.g. eth1.")
    parser.add_argument("--execute", action="store_true", help="Actually move the robot.")
    parser.add_argument("--yes", action="store_true", help="Confirm the robot and hand have clear space.")
    parser.add_argument("--pregrasp-target", nargs=3, type=float, default=DEFAULT_PREGRASP_TARGET)
    parser.add_argument("--wrist-roll-target", type=float, default=None)
    parser.add_argument("--wrist-pitch-target", type=float, default=-0.50)
    parser.add_argument("--wrist-yaw-target", type=float, default=None)
    parser.add_argument("--weight", type=float, default=0.75)
    parser.add_argument("--move-duration", type=float, default=6.0)
    parser.add_argument("--adjust-duration", type=float, default=1.0)
    parser.add_argument("--release-duration", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.06)
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
    parser.add_argument("--no-release-on-exit", action="store_true")

    parser.add_argument("--hand-ip", default="192.168.124.210")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--hand-unit-id", type=int, default=1)
    parser.add_argument("--hand-timeout", type=float, default=2.0)
    parser.add_argument("--hand-speed", type=int, default=1000)
    parser.add_argument("--hand-force", type=int, default=2000)
    parser.add_argument("--hand-hold", type=float, default=0.5)
    parser.add_argument("--hand-angle", nargs=6, type=int, default=DEFAULT_HAND_ANGLE)
    parser.add_argument("--grasp-angle", nargs=6, type=int, default=DEFAULT_GRASP_ANGLE)
    parser.add_argument("--read-hand-after", action="store_true", help="Read hand angleAct after each hand command.")
    parser.add_argument("--skip-hand", action="store_true")
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
        read_after=args.read_hand_after,
    )


def validate_hand_values(values: Sequence[int]) -> None:
    if len(values) != 6:
        raise ValueError("expected six hand angle values")
    if any(value < -1 or value > 1000 for value in values):
        raise ValueError("hand angle values must be -1 or in 0..1000")


def hand_tcp_preflight(args: argparse.Namespace) -> None:
    with socket.create_connection((args.hand_ip, args.hand_port), timeout=args.hand_timeout):
        return


def read_hand_state(args: argparse.Namespace, label: str = "debug") -> None:
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print("TCP connected")
        read_state(client, label)


def read_hand_low_level(args: argparse.Namespace, label: str = "debug") -> None:
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print("TCP connected")
        angle_set = client.read_holding_registers(REG["angleSet"], 6)
        angle_act = client.read_holding_registers(REG["angleAct"], 6)
        pos_set = client.read_holding_registers(POS_SET_ADDR, 6)
        pos_act = client.read_holding_registers(POS_ACT_ADDR, 6)
        current = client.read_holding_registers(CURRENT_ADDR, 6)
        err = client.read_holding_registers(REG["errCode"], 3)
        print(f"{label} angleSet[6]: {fmt(angle_set)}")
        print(f"{label} angleAct[6]: {fmt(angle_act)}")
        print(f"{label} posSet[6]: {fmt(pos_set)}")
        print(f"{label} posAct[6]: {fmt(pos_act)}")
        print(f"{label} current[6]: {fmt(current)}")
        print(f"{label} errCode_words[3]: {fmt(err)}")


def write_single_register(client: ModbusTcpClient, address: int, value: int) -> None:
    if client._sock is None:
        raise RuntimeError("not connected")
    client._transaction_id = (client._transaction_id + 1) & 0xFFFF
    function = 0x06
    pdu = bytes([function]) + address.to_bytes(2, "big") + (value & 0xFFFF).to_bytes(2, "big")
    mbap = (
        client._transaction_id.to_bytes(2, "big")
        + (0).to_bytes(2, "big")
        + (len(pdu) + 1).to_bytes(2, "big")
        + bytes([client.unit_id])
    )
    client._sock.sendall(mbap + pdu)

    header = client._recv_exact(7)
    rx_tid = int.from_bytes(header[0:2], "big")
    if rx_tid != client._transaction_id:
        raise RuntimeError(f"transaction mismatch: sent {client._transaction_id}, got {rx_tid}")
    payload = client._recv_exact(int.from_bytes(header[4:6], "big") - 1)
    if payload[0] & 0x80:
        code = payload[1] if len(payload) > 1 else None
        raise RuntimeError(f"Modbus single-write exception: {code}")
    if payload[0] != function:
        raise RuntimeError(f"unexpected single-write response function: 0x{payload[0]:02x}")


def read_registers_raw(args: argparse.Namespace, address: int, count: int) -> None:
    if count < 1 or count > 20:
        raise ValueError("raw register count must be in 1..20")
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print("TCP connected")
        values = client.read_holding_registers(address, count)
        print(f"reg[{address}:{address + count - 1}]: {fmt(values)}")


def write_register_raw(args: argparse.Namespace, address: int, value: int) -> None:
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print("TCP connected")
        print(f"write_single_register addr={address} value={value}")
        write_single_register(client, address, value)
        time.sleep(args.hand_hold)


def validate_pos_values(values: Sequence[int]) -> None:
    if len(values) != 6:
        raise ValueError("expected six POS_SET values")
    if any(value < -1 or value > 2000 for value in values):
        raise ValueError("POS_SET values must be -1 or in 0..2000")


def write_hand_pos(args: argparse.Namespace, values: Sequence[int], label: str) -> None:
    validate_pos_values(values)
    speed = [args.hand_speed] * 6
    force = [args.hand_force] * 6
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print("TCP connected")
        print(f"hand_speedSet: {fmt(speed)}")
        write_registers(client, REG["speedSet"], speed)
        time.sleep(0.2)
        print(f"hand_forceSet: {fmt(force)}")
        write_registers(client, REG["forceSet"], force)
        time.sleep(0.2)
        print(f"hand_posSet_{label}: {fmt(values)}")
        write_registers(client, POS_SET_ADDR, list(values))
        time.sleep(args.hand_hold)


def write_hand_pos_channel(args: argparse.Namespace, index: int, value: int) -> None:
    if index < 0 or index > 5:
        raise ValueError("POS_SET channel index must be in 0..5")
    if value < -1 or value > 2000:
        raise ValueError("POS_SET channel value must be -1 or in 0..2000")
    speed = [args.hand_speed] * 6
    force = [args.hand_force] * 6
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print("TCP connected")
        print(f"hand_speedSet: {fmt(speed)}")
        write_registers(client, REG["speedSet"], speed)
        time.sleep(0.2)
        print(f"hand_forceSet: {fmt(force)}")
        write_registers(client, REG["forceSet"], force)
        time.sleep(0.2)
        print(f"hand_posSet_A{index}: {value}")
        write_single_register(client, POS_SET_ADDR + index, value)
        time.sleep(args.hand_hold)


def run_hand_action(args: argparse.Namespace, action_id: int) -> None:
    if action_id < 1 or action_id > 40:
        raise ValueError("action id must be in 1..40")
    with ModbusTcpClient(args.hand_ip, args.hand_port, args.hand_timeout, args.hand_unit_id) as client:
        print("TCP connected")
        print(f"hand_actionSeq: {action_id}")
        write_registers(client, REG["actionSeq"], [action_id])
        time.sleep(0.2)
        print("hand_actionRun: 1")
        write_registers(client, REG["actionRun"], [1])
        time.sleep(args.hand_hold)


def smoothstep(ratio: float) -> float:
    ratio = min(max(ratio, 0.0), 1.0)
    return ratio * ratio * (3.0 - 2.0 * ratio)


class HoldLoop:
    def __init__(
        self,
        controller: OfficialStyleArmSdkController,
        q_left: Dict[int, float],
        q_right: Dict[int, float],
        q_waist: Dict[int, float],
        args: argparse.Namespace,
    ):
        self.controller = controller
        self.q_left = dict(q_left)
        self.q_right = dict(q_right)
        self.q_waist = dict(q_waist)
        self.args = args
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="arm_sdk_hold", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def snapshot(self) -> Dict[int, float]:
        with self.lock:
            return dict(self.q_right)

    def ramp_to(self, q_right_goal: Dict[int, float], duration: float) -> None:
        with self.lock:
            q_start = dict(self.q_right)
        steps = max(1, int(max(0.0, duration) / self.controller.control_dt))
        for step in range(steps):
            ratio = smoothstep((step + 1) / steps)
            q_next = {
                joint: q_start[joint] + ratio * (q_right_goal[joint] - q_start[joint])
                for joint in q_start
            }
            with self.lock:
                self.q_right = q_next
            time.sleep(self.controller.control_dt)

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            with self.lock:
                q_right = dict(self.q_right)
            self.controller.write_group_command(
                q_left=self.q_left,
                q_right=q_right,
                q_waist=self.q_waist,
                weight=self.args.weight,
                kp_right=self.args.kp_right,
                kd_right=self.args.kd_right,
                kp_left=self.args.kp_left,
                kd_left=self.args.kd_left,
                kp_waist=self.args.kp_waist,
                kd_waist=self.args.kd_waist,
            )
            next_tick += self.controller.control_dt
            time.sleep(max(0.0, next_tick - time.monotonic()))


def solve_pregrasp_q(
    target: np.ndarray,
    waist_q: np.ndarray,
    right_joints: Sequence[int],
    seed_motor_q: np.ndarray,
    tolerance: float,
) -> tuple:
    signs = np.ones(7)
    ik = solve_position_ik(
        target=target,
        waist_q=waist_q,
        joint_indices=right_joints,
        signs=signs,
        seed=motor_to_model_q(seed_motor_q, signs),
        tolerance=tolerance,
    )
    q_values = model_to_motor_q(ik.q_model, signs)
    q_target = {joint: float(value) for joint, value in zip(right_joints, q_values)}
    return ik, q_values, q_target


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


def validate_wrist_value(joint: int, value: float) -> None:
    low, high = WRIST_LIMITS[joint]
    if not low <= value <= high:
        raise ValueError(f"{WRIST_NAMES[joint]} must be in [{low:.4f}, {high:.4f}] rad")


def set_wrist_joint(
    q_values: np.ndarray,
    q_target: Dict[int, float],
    right_joints: Sequence[int],
    joint: int,
    value: Optional[float],
) -> None:
    if value is None or joint not in right_joints:
        return
    validate_wrist_value(joint, float(value))
    idx = right_joints.index(joint)
    q_values[idx] = float(value)
    q_target[joint] = float(value)


def format_help() -> str:
    return """Commands:
  wr RAD                 set RightWristRoll
  wp RAD                 set RightWristPitch, e.g. wp -0.8
  wy RAD                 set RightWristYaw
  w ROLL PITCH YAW       set all three wrist joints
  b VALUE                set A4 only, currently observed as thumb motion
  t VALUE                set A5 only, legacy thumb-rotation guess
  ch IDX VALUE           set one hand channel A0..A5, e.g. ch 5 500
  hand A0 A1 A2 A3 A4 A5 set all six hand angles
  posch IDX VALUE        set one POS_SET channel P0..P5, e.g. posch 5 600
  pos5 VALUE             alias for posch 5 VALUE
  pos P0 P1 P2 P3 P4 P5  set all six POS_SET values, -1 or 0..2000
  open                   send startup hand-angle preset
  grasp                  send grasp-angle preset
  read                   read RH56E2 angleAct/err/temp over Modbus
  readpos                read angleSet/angleAct/posSet/posAct/current
  readreg ADDR COUNT     read raw holding registers
  reg ADDR VALUE         write one raw register with Modbus function 0x06
  action ID              run built-in/user hand action sequence 1..40
  set PITCH BEND ROT     set wrist pitch, thumb_bend, thumb_rotation together
  set3 ROLL PITCH YAW BEND ROT
                         set all wrist joints plus thumb_bend/thumb_rotation
  show                   print current wrist/thumb targets
  help                   print this help
  q                      release at current pose and exit

Hand channel note:
  A0..A5 are raw RH56E2 Modbus angleSet channels. The thumb labels are not
  guaranteed; use ch/read to confirm the real mapping on this hand.
  VALUE may be -1, which follows the vendor demo convention: do not set that channel.
  POS_SET is lower-level actuator position control from the vendor manual:
  P5 is thumb rotation actuator position. Use small steps while testing.
"""


def update_thumb(hand_angle: List[int], index: int, value: int) -> List[int]:
    if value < -1 or value > 1000:
        raise ValueError("thumb value must be -1 or in 0..1000")
    updated = list(hand_angle)
    updated[index] = int(value)
    return updated


def update_hand_channel(hand_angle: List[int], index: int, value: int) -> List[int]:
    if index < 0 or index > 5:
        raise ValueError("channel index must be in 0..5")
    if value < -1 or value > 1000:
        raise ValueError("channel value must be -1 or in 0..1000")
    updated = list(hand_angle)
    updated[index] = int(value)
    return updated


def ramp_wrist_joint(hold: HoldLoop, joint: int, value: float, duration: float) -> None:
    validate_wrist_value(joint, value)
    q_goal = hold.snapshot()
    q_goal[joint] = value
    hold.ramp_to(q_goal, duration)
    print(f"{WRIST_NAMES[joint]} -> {value:+.4f} rad")


def ramp_wrist_triplet(hold: HoldLoop, roll: float, pitch: float, yaw: float, duration: float) -> None:
    values = {
        G1JointIndex.RightWristRoll: roll,
        G1JointIndex.RightWristPitch: pitch,
        G1JointIndex.RightWristYaw: yaw,
    }
    for joint, value in values.items():
        validate_wrist_value(joint, value)
    q_goal = hold.snapshot()
    q_goal.update(values)
    hold.ramp_to(q_goal, duration)
    print(
        "wrist -> "
        f"roll={roll:+.4f}, pitch={pitch:+.4f}, yaw={yaw:+.4f} rad"
    )


def interactive_loop(
    hold: HoldLoop,
    args: argparse.Namespace,
    right_joints: Sequence[int],
    hand_angle: List[int],
) -> List[int]:
    print(format_help())
    while True:
        try:
            line = input("debug> ").strip()
        except EOFError:
            print()
            return hand_angle
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd in ("q", "quit", "exit"):
                return hand_angle
            if cmd in ("h", "help", "?"):
                print(format_help())
                continue
            if cmd == "show":
                q_now = hold.snapshot()
                for joint in (
                    G1JointIndex.RightWristRoll,
                    G1JointIndex.RightWristPitch,
                    G1JointIndex.RightWristYaw,
                ):
                    print(f"{WRIST_NAMES[joint]}: {q_now.get(joint, float('nan')):+.4f} rad")
                print("hand_angle:", " ".join(str(v) for v in hand_angle))
                print(f"A4: {hand_angle[4]}, A5: {hand_angle[5]}")
                print("note: A4/A5 thumb labels are unverified for this RH56E2 Modbus path")
                continue
            if cmd == "read":
                read_hand_state(args)
                continue
            if cmd == "readpos":
                read_hand_low_level(args)
                continue
            if cmd == "readreg":
                if len(parts) != 3:
                    raise ValueError("usage: readreg ADDR COUNT")
                read_registers_raw(args, int(parts[1]), int(parts[2]))
                continue
            if cmd == "reg":
                if len(parts) != 3:
                    raise ValueError("usage: reg ADDR VALUE")
                write_register_raw(args, int(parts[1]), int(parts[2]))
                continue
            if cmd in ("wr", "roll"):
                if len(parts) != 2:
                    raise ValueError("usage: wr RAD")
                ramp_wrist_joint(hold, G1JointIndex.RightWristRoll, float(parts[1]), args.adjust_duration)
                continue
            if cmd in ("wp", "pitch"):
                if len(parts) != 2:
                    raise ValueError("usage: wp RAD")
                ramp_wrist_joint(hold, G1JointIndex.RightWristPitch, float(parts[1]), args.adjust_duration)
                continue
            if cmd in ("wy", "yaw"):
                if len(parts) != 2:
                    raise ValueError("usage: wy RAD")
                ramp_wrist_joint(hold, G1JointIndex.RightWristYaw, float(parts[1]), args.adjust_duration)
                continue
            if cmd == "w":
                if len(parts) != 4:
                    raise ValueError("usage: w ROLL PITCH YAW")
                ramp_wrist_triplet(hold, float(parts[1]), float(parts[2]), float(parts[3]), args.adjust_duration)
                continue
            if cmd == "b":
                if len(parts) != 2:
                    raise ValueError("usage: b VALUE")
                hand_angle = update_thumb(hand_angle, 4, int(parts[1]))
                write_hand_pose(hand_args(args), hand_angle, "A4")
                continue
            if cmd == "t":
                if len(parts) != 2:
                    raise ValueError("usage: t VALUE")
                hand_angle = update_thumb(hand_angle, 5, int(parts[1]))
                write_hand_pose(hand_args(args), hand_angle, "A5")
                continue
            if cmd == "ch":
                if len(parts) != 3:
                    raise ValueError("usage: ch IDX VALUE")
                index = int(parts[1])
                value = int(parts[2])
                hand_angle = update_hand_channel(hand_angle, index, value)
                write_hand_pose(hand_args(args), hand_angle, f"A{index}")
                continue
            if cmd == "hand":
                if len(parts) != 7:
                    raise ValueError("usage: hand A0 A1 A2 A3 A4 A5")
                hand_angle = [int(value) for value in parts[1:]]
                validate_hand_values(hand_angle)
                write_hand_pose(hand_args(args), hand_angle, "custom")
                continue
            if cmd == "pos5":
                if len(parts) != 2:
                    raise ValueError("usage: pos5 VALUE")
                write_hand_pos_channel(args, 5, int(parts[1]))
                continue
            if cmd == "posch":
                if len(parts) != 3:
                    raise ValueError("usage: posch IDX VALUE")
                write_hand_pos_channel(args, int(parts[1]), int(parts[2]))
                continue
            if cmd == "pos":
                if len(parts) != 7:
                    raise ValueError("usage: pos P0 P1 P2 P3 P4 P5")
                values = [int(value) for value in parts[1:]]
                write_hand_pos(args, values, "custom")
                continue
            if cmd == "action":
                if len(parts) != 2:
                    raise ValueError("usage: action ID")
                run_hand_action(args, int(parts[1]))
                continue
            if cmd == "open":
                hand_angle = list(args.hand_angle)
                write_hand_pose(hand_args(args), hand_angle, "open")
                continue
            if cmd == "grasp":
                hand_angle = list(args.grasp_angle)
                write_hand_pose(hand_args(args), hand_angle, "grasp")
                continue
            if cmd == "set":
                if len(parts) != 4:
                    raise ValueError("usage: set PITCH BEND ROT")
                pitch = float(parts[1])
                bend = int(parts[2])
                rotation = int(parts[3])
                hand_angle = update_thumb(update_thumb(hand_angle, 4, bend), 5, rotation)
                ramp_wrist_joint(hold, G1JointIndex.RightWristPitch, pitch, args.adjust_duration)
                write_hand_pose(hand_args(args), hand_angle, "wrist_thumb")
                continue
            if cmd == "set3":
                if len(parts) != 6:
                    raise ValueError("usage: set3 ROLL PITCH YAW BEND ROT")
                roll = float(parts[1])
                pitch = float(parts[2])
                yaw = float(parts[3])
                bend = int(parts[4])
                rotation = int(parts[5])
                hand_angle = update_thumb(update_thumb(hand_angle, 4, bend), 5, rotation)
                ramp_wrist_triplet(hold, roll, pitch, yaw, args.adjust_duration)
                write_hand_pose(hand_args(args), hand_angle, "wrist3_thumb")
                continue
            print(f"Unknown command: {cmd}")
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if args.execute and not args.yes:
        print("Refusing to move without --yes.", file=sys.stderr)
        return 1
    if args.weight <= 0.0 or args.weight > 1.0:
        print("--weight must be in (0, 1].", file=sys.stderr)
        return 2
    if min(args.move_duration, args.adjust_duration, args.release_duration) < 0.0:
        print("Durations must be non-negative.", file=sys.stderr)
        return 2
    try:
        if args.wrist_roll_target is not None:
            validate_wrist_value(G1JointIndex.RightWristRoll, args.wrist_roll_target)
        validate_wrist_value(G1JointIndex.RightWristPitch, args.wrist_pitch_target)
        if args.wrist_yaw_target is not None:
            validate_wrist_value(G1JointIndex.RightWristYaw, args.wrist_yaw_target)
    except ValueError as exc:
        print(f"Refusing wrist target: {exc}", file=sys.stderr)
        return 2
    try:
        validate_hand_values(args.hand_angle)
        validate_hand_values(args.grasp_angle)
    except ValueError as exc:
        print(f"Refusing hand command: {exc}", file=sys.stderr)
        return 2
    if args.execute and not args.skip_hand:
        try:
            hand_tcp_preflight(args)
        except OSError as exc:
            print(f"hand_preflight_connect failed: {exc}", file=sys.stderr)
            return 6

    target = np.asarray(args.pregrasp_target, dtype=float)
    try:
        validate_target(target)
    except ValueError as exc:
        print(f"Refusing pregrasp target: {exc}", file=sys.stderr)
        return 2

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

    ik, q_goal_values, q_goal = solve_pregrasp_q(target, waist_q, right_joints, q_start_values, args.tolerance)
    set_wrist_joint(q_goal_values, q_goal, right_joints, G1JointIndex.RightWristRoll, args.wrist_roll_target)
    set_wrist_joint(q_goal_values, q_goal, right_joints, G1JointIndex.RightWristPitch, args.wrist_pitch_target)
    set_wrist_joint(q_goal_values, q_goal, right_joints, G1JointIndex.RightWristYaw, args.wrist_yaw_target)
    reached_pose = right_hand_pose(motor_to_model_q(q_goal_values, np.ones(7)), waist_q, right_joints)
    command_reached = reached_pose[:3, 3]
    command_residual = float(np.linalg.norm(target - command_reached))
    max_delta = float(np.max(np.abs(q_goal_values - q_start_values)))
    move_duration = max(args.move_duration, max_delta / max(args.max_joint_speed, 1e-6))

    print("pregrasp_target_pelvis_xyz:", np.array2string(target, precision=4))
    print("ik_success:", ik.success)
    print(f"ik_residual_m: {ik.residual:.4f}")
    print("ik_reached_xyz:", np.array2string(ik.reached, precision=4))
    print("wrist_targets:")
    print(f"  RightWristRoll: {q_goal[G1JointIndex.RightWristRoll]:+.4f} rad")
    print(f"  RightWristPitch: {q_goal[G1JointIndex.RightWristPitch]:+.4f} rad")
    print(f"  RightWristYaw: {q_goal[G1JointIndex.RightWristYaw]:+.4f} rad")
    print("command_reached_xyz:", np.array2string(command_reached, precision=4))
    print(f"command_residual_m: {command_residual:.4f}")
    print(f"max_right_joint_delta_rad: {max_delta:.4f}")
    print(f"move_duration_s: {move_duration:.2f}")
    print("right_goal:")
    print(format_joint_table(right_joints, q_goal_values))
    print("right_delta:")
    print(format_joint_delta_table(right_joints, q_start_values, q_goal_values))

    if not ik.success:
        print("Refusing execution because IK did not meet tolerance.", file=sys.stderr)
        return 4
    if max_delta > args.max_joint_delta:
        print(f"Refusing execution: max joint delta {max_delta:.3f} exceeds {args.max_joint_delta:.3f}.", file=sys.stderr)
        return 5
    if not args.execute:
        print("Dry run only. Add --execute --yes to move the robot and enter the debug prompt.")
        return 0

    hold = None
    exit_code = 0
    try:
        print("Stage 1: move to pregrasp target and keep it there.")
        run_timed_phase(
            controller,
            move_duration,
            q_left_hold,
            q_waist_hold,
            q_right_start,
            q_goal,
            0.0,
            args.weight,
            args,
        )
        hold = HoldLoop(controller, q_left_hold, q_goal, q_waist_hold, args)
        hold.start()
        hand_angle = list(args.hand_angle)
        if not args.skip_hand:
            print("Stage 2: send startup hand angle.")
            write_hand_pose(hand_args(args), hand_angle, "startup")
        print("Stage 3: interactive wrist/thumb debug. The arm will stay at pregrasp while this prompt is open.")
        interactive_loop(hold, args, right_joints, hand_angle)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        exit_code = 130
    finally:
        q_right_last = q_goal if hold is None else hold.snapshot()
        if hold is not None:
            hold.stop()
        if args.no_release_on_exit:
            print("Leaving without release because --no-release-on-exit was set.")
        else:
            print("Release arm_sdk at current pose. No retract motion is commanded.")
            release_group(controller, q_left_hold, q_right_last, q_waist_hold, args.weight, args)
            print("arm_sdk released.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
