#!/usr/bin/env python3
"""Control only the Inspire RH56E2/RH56-series hand over Modbus TCP.

This script is extracted from g1_move_right_hand_ik_once.py. It only writes
hand speed, force, and angle registers. It does not publish rt/arm_sdk and will
not command the G1 arm, legs, or locomotion.
"""

import argparse
import sys
import time
from pathlib import Path

HIGH_LEVEL_DIR = Path(__file__).resolve().parents[1] / "high_level"
if str(HIGH_LEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(HIGH_LEVEL_DIR))

from rh56e2_modbus_move_once import read_state, write_registers
from rh56e2_modbus_probe import ModbusTcpClient, REG, fmt


def write_hand_pose(args, angle, label):
    speed = [args.speed] * 6
    force = [args.force] * 6

    print(f"hand_connect: {args.ip}:{args.port}")
    with ModbusTcpClient(args.ip, args.port, args.timeout, args.unit_id) as client:
        print("TCP connected")
        if args.read_before:
            read_state(client, "before")

        print(f"hand_speedSet: {fmt(speed)}")
        write_registers(client, REG["speedSet"], speed)
        time.sleep(0.2)

        print(f"hand_forceSet: {fmt(force)}")
        write_registers(client, REG["forceSet"], force)
        time.sleep(0.2)

        print(f"hand_angleSet_{label}: {fmt(angle)}")
        write_registers(client, REG["angleSet"], angle)
        time.sleep(args.hold)

        if args.read_after:
            read_state(client, "after")

    print(f"hand_{label}_done")


def main():
    parser = argparse.ArgumentParser(
        description="Move only the RH56E2/RH56-series hand over Modbus TCP."
    )
    parser.add_argument("--ip", default="192.168.124.210", help="RH56E2 hand IP.")
    parser.add_argument("--port", type=int, default=6000, help="RH56E2 Modbus TCP port.")
    parser.add_argument("--unit-id", type=int, default=1, help="Modbus unit id.")
    parser.add_argument("--timeout", type=float, default=2.0, help="TCP timeout in seconds.")
    parser.add_argument("--speed", type=int, default=250, help="Hand speed, 0..1000.")
    parser.add_argument("--force", type=int, default=250, help="Hand force, 0..1000.")
    parser.add_argument("--hold", type=float, default=2.0, help="Seconds to wait after angle command.")
    parser.add_argument("--read-before", action="store_true", help="Read hand state before moving.")
    parser.add_argument("--read-after", action="store_true", help="Read hand state after moving.")
    parser.add_argument(
        "--angle",
        type=int,
        nargs=6,
        default=None,
        metavar=("A0", "A1", "A2", "A3", "A4", "A5"),
        help="Six hand angle values in 0..1000.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Use the same open angle preset as g1_move_right_hand_ik_once.py.",
    )
    parser.add_argument(
        "--grasp",
        action="store_true",
        help="Use the same grasp angle preset as g1_move_right_hand_ik_once.py.",
    )
    parser.add_argument("--yes", action="store_true", help="Confirm the hand has clear space.")
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing to move the hand without --yes.")
    if args.speed < 0 or args.speed > 1000:
        raise SystemExit("--speed must be in 0..1000.")
    if args.force < 0 or args.force > 1000:
        raise SystemExit("--force must be in 0..1000.")
    if args.hold < 0:
        raise SystemExit("--hold cannot be negative.")

    selected_modes = sum([args.angle is not None, args.open, args.grasp])
    if selected_modes != 1:
        raise SystemExit("Use exactly one of --angle, --open, or --grasp.")

    if args.open:
        angle = [700, 700, 700, 700, 700, 700]
        label = "open"
    elif args.grasp:
        angle = [0, 0, 0, 0, 0, 0]
        label = "grasp"
    else:
        angle = args.angle
        label = "custom"

    if any(value < 0 or value > 1000 for value in angle):
        raise SystemExit("All angle values must be in 0..1000.")

    write_hand_pose(args, angle, label)


if __name__ == "__main__":
    main()
