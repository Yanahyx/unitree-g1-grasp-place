#!/usr/bin/env python3
"""Open the RH56E2 hand, wait for contact, then close from hand feedback.

This is intentionally hand-only: it does not publish arm or locomotion DDS
commands.  The operator first lets the hand open, establishes an empty-hand
feedback baseline, and then places the bottle against the open fingers.  A
forceAct/current delta must persist for several polls before the thumb and
fingers are closed.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


HIGH_LEVEL_DIR = Path(__file__).resolve().parents[1] / "high_level"
if str(HIGH_LEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(HIGH_LEVEL_DIR))

from rh56e2_modbus_move_once import write_registers  # noqa: E402
from rh56e2_modbus_probe import ModbusTcpClient, REG, fmt  # noqa: E402


# The existing force-feedback grasp scripts use this register for signed motor
# current feedback.  forceAct is part of the common RH56E2 register map.
CURRENT_ADDR = 1594
CHANNEL_LABELS = (
    "A0/finger0",
    "A1/finger1",
    "A2/finger2",
    "A3/finger3",
    "A4/thumb_bend",
    "A5/thumb_rotation",
)
DEFAULT_OPEN_ANGLE = [1000, 1000, 1000, 1000, 1000, 1000]
DEFAULT_CLOSE_ANGLE = [0, 0, 0, 0, 400, 300]


def signed16(values: Sequence[int]) -> np.ndarray:
    return np.asarray([value - 65536 if value >= 32768 else value for value in values], dtype=int)


def validate_angles(values: Sequence[int], label: str) -> None:
    if len(values) != 6 or any(value < -1 or value > 1000 for value in values):
        raise ValueError(f"{label} must contain six values in -1..1000")


def feedback_snapshot(client: ModbusTcpClient) -> Dict[str, np.ndarray]:
    return {
        "force": signed16(client.read_holding_registers(REG["forceAct"], 6)),
        "current": signed16(client.read_holding_registers(CURRENT_ADDR, 6)),
        "angle": np.asarray(client.read_holding_registers(REG["angleAct"], 6), dtype=int),
    }


def write_pose(
    client: ModbusTcpClient,
    angles: Sequence[int],
    speed: int,
    force: int,
    label: str,
) -> None:
    validate_angles(angles, label)
    speed_values = [int(speed)] * 6
    force_values = [int(force)] * 6
    print(f"hand_speedSet_{label}: {fmt(speed_values)}")
    write_registers(client, REG["speedSet"], speed_values)
    time.sleep(0.1)
    print(f"hand_forceSet_{label}: {fmt(force_values)}")
    write_registers(client, REG["forceSet"], force_values)
    time.sleep(0.1)
    print(f"hand_angleSet_{label}: {fmt(list(angles))}")
    write_registers(client, REG["angleSet"], list(angles))


def channel_label(index: int) -> str:
    return CHANNEL_LABELS[index] if 0 <= index < len(CHANNEL_LABELS) else f"A{index}"


def feedback_reason(
    snapshot: Dict[str, np.ndarray],
    baseline: Dict[str, np.ndarray],
    force_delta_threshold: int,
    current_delta_threshold: int,
) -> Optional[str]:
    force_delta = np.abs(snapshot["force"] - baseline["force"])
    current_delta = np.abs(snapshot["current"] - baseline["current"])
    force_index = int(np.argmax(force_delta))
    current_index = int(np.argmax(current_delta))
    max_force = int(force_delta[force_index])
    max_current = int(current_delta[current_index])
    if max_force >= force_delta_threshold:
        return f"force_delta={max_force} channel={channel_label(force_index)}"
    if max_current >= current_delta_threshold:
        return f"current_delta={max_current} channel={channel_label(current_index)}"
    return None


def thumb_waypoints(start: int, target: int, step: int) -> Sequence[int]:
    if start == target:
        return [target]
    direction = 1 if target > start else -1
    values = list(range(start + direction * step, target + direction, direction * step))
    if not values or values[-1] != target:
        values.append(target)
    return [max(0, min(1000, int(value))) for value in values]


def next_step_toward(
    current: Sequence[int],
    target: Sequence[int],
    step: int,
) -> list:
    values = []
    for start, goal in zip(current, target):
        if goal < 0 or start == goal:
            values.append(int(start))
        elif start > goal:
            values.append(max(int(goal), int(start) - int(step)))
        else:
            values.append(min(int(goal), int(start) + int(step)))
    return values


def incremental_close_after_feedback(
    client: ModbusTcpClient,
    args: argparse.Namespace,
    close_angles: Sequence[int],
) -> None:
    # Keep a cushion on the four finger channels.  The feedback loop, not a
    # full-close preset, decides when the bottle is held.
    safe_target = list(close_angles)
    for index in range(4):
        if safe_target[index] >= 0:
            safe_target[index] = max(int(safe_target[index]), int(args.min_finger_angle))

    command = list(args.open_angle)
    print(f"incremental_close_safe_target: {fmt(safe_target)}")
    for step_index in range(1, args.max_close_steps + 1):
        next_command = next_step_toward(command, safe_target, args.close_step)
        if next_command == command:
            print("incremental_close_stop: safe_target_reached")
            return

        before = feedback_snapshot(client)
        write_pose(
            client,
            next_command,
            args.close_speed,
            args.close_force,
            f"increment_{step_index}",
        )
        time.sleep(args.step_hold)
        after = feedback_snapshot(client)
        reason = feedback_reason(
            after,
            before,
            args.feedback_force_delta,
            args.feedback_current_delta,
        )
        print(f"incremental_close_step: {step_index}")
        print(f"incremental_close_command: {fmt(next_command)}")
        print(f"incremental_close_angleAct: {fmt(after['angle'].tolist())}")
        if reason is not None:
            print(f"incremental_close_stop: {reason}")
            return

        # If a finger cannot follow the small command, treat that as contact
        # even when the force register is slow to update.
        stalled = []
        for index in range(4):
            if safe_target[index] < 0 or next_command[index] <= safe_target[index]:
                continue
            if int(after["angle"][index]) > int(next_command[index]) + args.angle_tolerance:
                stalled.append(channel_label(index))
        if stalled:
            print(f"incremental_close_stop: angle_stall channels={','.join(stalled)}")
            return
        command = next_command

    print("incremental_close_stop: max_close_steps_reached")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open RH56E2, wait for force/current contact, and close the hand only."
    )
    parser.add_argument("--ip", default="192.168.124.210")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--yes", action="store_true", help="Confirm the hand and operator area are clear.")
    parser.add_argument("--open-angle", nargs=6, type=int, default=DEFAULT_OPEN_ANGLE)
    parser.add_argument("--close-angle", nargs=6, type=int, default=DEFAULT_CLOSE_ANGLE)
    parser.add_argument("--open-speed", type=int, default=100)
    parser.add_argument("--open-force", type=int, default=200)
    parser.add_argument("--close-speed", type=int, default=100)
    parser.add_argument("--close-force", type=int, default=500)
    parser.add_argument("--open-hold", type=float, default=1.0)
    parser.add_argument("--baseline-settle", type=float, default=0.5)
    parser.add_argument("--close-hold", type=float, default=1.0)
    parser.add_argument("--step-hold", type=float, default=0.4)
    parser.add_argument("--close-step", type=int, default=80)
    parser.add_argument("--max-close-steps", type=int, default=10)
    parser.add_argument(
        "--min-finger-angle",
        type=int,
        default=250,
        help="Do not command A0..A3 below this angle during feedback-controlled closure.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=int,
        default=120,
        help="Actual-angle lag treated as contact/stall after a close step.",
    )
    parser.add_argument("--feedback-timeout", type=float, default=30.0)
    parser.add_argument("--feedback-poll", type=float, default=0.05)
    parser.add_argument("--feedback-force-delta", type=int, default=60)
    parser.add_argument("--feedback-current-delta", type=int, default=35)
    parser.add_argument("--trigger-consecutive", type=int, default=3)
    parser.add_argument("--thumb-channel", type=int, default=5)
    parser.add_argument("--thumb-step", type=int, default=40)
    parser.add_argument("--thumb-speed", type=int, default=100)
    parser.add_argument("--thumb-force", type=int, default=500)
    parser.add_argument("--thumb-hold", type=float, default=0.05)
    parser.add_argument("--thumb-delay", type=float, default=0.4)
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not wait for Enter; establish the baseline immediately after opening.",
    )
    parser.add_argument(
        "--close-on-timeout",
        action="store_true",
        help="Close after timeout even without feedback; disabled by default for safety.",
    )
    parser.add_argument(
        "--no-close-confirm",
        action="store_true",
        help="Skip the operator confirmation after contact detection; unsafe for manual placement.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.yes:
        print("Refusing to move the hand without --yes.", file=sys.stderr)
        return 1
    if args.timeout <= 0.0 or args.feedback_timeout < 0.0 or args.feedback_poll <= 0.0:
        print("Timeouts and feedback poll must be positive (feedback timeout may be zero).", file=sys.stderr)
        return 2
    if args.trigger_consecutive <= 0 or args.close_step <= 0 or args.max_close_steps <= 0:
        print("--trigger-consecutive, --close-step, and --max-close-steps must be positive.", file=sys.stderr)
        return 2
    if not 0 <= args.thumb_channel < 6:
        print("--thumb-channel must be in 0..5.", file=sys.stderr)
        return 2
    if not 0 <= args.min_finger_angle <= 1000 or args.angle_tolerance < 0 or args.step_hold < 0.0:
        print("Invalid incremental-close safety limits.", file=sys.stderr)
        return 2
    for name, values in (("open-angle", args.open_angle), ("close-angle", args.close_angle)):
        try:
            validate_angles(values, name)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    for name, value, maximum in (
        ("open-speed", args.open_speed, 1000),
        ("close-speed", args.close_speed, 1000),
        ("thumb-speed", args.thumb_speed, 1000),
        ("open-force", args.open_force, 3000),
        ("close-force", args.close_force, 3000),
        ("thumb-force", args.thumb_force, 3000),
    ):
        if not 0 <= value <= maximum:
            print(f"--{name} must be in 0..{maximum}.", file=sys.stderr)
            return 2

    print(f"hand_connect: {args.ip}:{args.port}")
    try:
        with ModbusTcpClient(args.ip, args.port, args.timeout, args.unit_id) as client:
            print("TCP connected")
            print("Stage 1: opening the hand.")
            write_pose(client, args.open_angle, args.open_speed, args.open_force, "open")
            time.sleep(args.open_hold)

            if not args.no_prompt:
                input("保持手前没有瓶子，确认手已张开后按 Enter 建立空手基线：")
            time.sleep(args.baseline_settle)
            baseline = feedback_snapshot(client)
            print(f"feedback_baseline_forceAct_signed[6]: {fmt(baseline['force'].tolist())}")
            print(f"feedback_baseline_current_signed[6]: {fmt(baseline['current'].tolist())}")
            print("请把咖啡瓶贴到张开的手指内侧并保持，检测到接触后会自动闭合；超时默认保持张开。")

            deadline = time.monotonic() + args.feedback_timeout if args.feedback_timeout > 0 else None
            consecutive = 0
            trigger_snapshot = None
            trigger_reason = None
            while deadline is None or time.monotonic() < deadline:
                snapshot = feedback_snapshot(client)
                reason = feedback_reason(
                    snapshot,
                    baseline,
                    args.feedback_force_delta,
                    args.feedback_current_delta,
                )
                if reason is None:
                    consecutive = 0
                else:
                    consecutive += 1
                    if consecutive >= args.trigger_consecutive:
                        trigger_snapshot = snapshot
                        trigger_reason = reason
                        break
                time.sleep(args.feedback_poll)

            if trigger_snapshot is None:
                print("feedback_triggered: false")
                if not args.close_on_timeout:
                    print("timeout_policy: keep hand open")
                    return 3
                print("timeout_policy: close requested")
            else:
                print(f"feedback_triggered: true ({trigger_reason})")
                print(f"feedback_trigger_forceAct_signed[6]: {fmt(trigger_snapshot['force'].tolist())}")
                print(f"feedback_trigger_current_signed[6]: {fmt(trigger_snapshot['current'].tolist())}")
                print(f"feedback_trigger_angleAct[6]: {fmt(trigger_snapshot['angle'].tolist())}")

            if trigger_snapshot is not None and not args.no_close_confirm:
                input("已检测到接触。确认瓶子位置合适后按 Enter，开始分段小步闭合：")

            print("Stage 2: feedback-triggered incremental close; no full-close preset will be sent.")
            incremental_close_after_feedback(client, args, args.close_angle)
            final = feedback_snapshot(client)
            print(f"feedback_final_forceAct_signed[6]: {fmt(final['force'].tolist())}")
            print(f"feedback_final_current_signed[6]: {fmt(final['current'].tolist())}")
            print(f"feedback_final_angleAct[6]: {fmt(final['angle'].tolist())}")
            return 0
    except KeyboardInterrupt:
        print("Interrupted; no automatic close was sent.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Hand feedback sequence failed: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
