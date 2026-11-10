#!/usr/bin/env python3
"""Minimal RH56E2/RH56-series Modbus TCP motion test.

This sends one low-speed, low-force angle command, then reads back state.
Use only when the hand is clear of objects and people.
"""

import argparse
import time
from typing import List

from rh56e2_modbus_probe import ModbusTcpClient, REG, fmt, split_bytes_from_words


def write_registers(client: ModbusTcpClient, address: int, values: List[int]) -> None:
    if client._sock is None:
        raise RuntimeError("not connected")

    client._transaction_id = (client._transaction_id + 1) & 0xFFFF
    function = 0x10
    count = len(values)
    byte_count = count * 2
    pdu = bytes([function]) + address.to_bytes(2, "big")
    pdu += count.to_bytes(2, "big") + bytes([byte_count])
    for value in values:
        pdu += (value & 0xFFFF).to_bytes(2, "big")

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
        raise RuntimeError(f"Modbus write exception: {code}")
    if payload[0] != function:
        raise RuntimeError(f"unexpected write response function: 0x{payload[0]:02x}")


def read_state(client: ModbusTcpClient, label: str) -> None:
    angle = client.read_holding_registers(REG["angleAct"], 6)
    err = client.read_holding_registers(REG["errCode"], 3)
    temp = client.read_holding_registers(REG["temp"], 3)
    print(f"{label} angleAct[6]: {fmt(angle)}")
    print(f"{label} errCode[6 bytes]: {fmt(split_bytes_from_words(err))}")
    print(f"{label} temp[6 bytes]: {fmt(split_bytes_from_words(temp))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one safe RH56E2 Modbus motion")
    parser.add_argument("--ip", default="192.168.124.210")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--angle",
        type=int,
        nargs=6,
        default=[300, 300, 300, 300, 300, 300],
        metavar=("A0", "A1", "A2", "A3", "A4", "A5"),
    )
    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--force", type=int, default=200)
    args = parser.parse_args()

    print(f"Connecting to {args.ip}:{args.port}")
    with ModbusTcpClient(args.ip, args.port, args.timeout, args.unit_id) as client:
        print("TCP connected")
        read_state(client, "before")

        speed = [args.speed] * 6
        force = [args.force] * 6
        print(f"write speedSet: {fmt(speed)}")
        write_registers(client, REG["speedSet"], speed)
        time.sleep(0.2)

        print(f"write forceSet: {fmt(force)}")
        write_registers(client, REG["forceSet"], force)
        time.sleep(0.2)

        print(f"write angleSet: {fmt(args.angle)}")
        write_registers(client, REG["angleSet"], args.angle)
        time.sleep(2.0)

        read_state(client, "after")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
