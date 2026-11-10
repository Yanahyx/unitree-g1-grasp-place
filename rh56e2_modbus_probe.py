#!/usr/bin/env python3
"""Read-only Modbus TCP probe for Inspire RH56E2/RH56-series hands.

This script does not send motion commands. It only reads identification and
status registers documented in the RH56-series examples/manuals.
"""

import argparse
import socket
import struct
import sys
from typing import Iterable, List, Sequence


REG = {
    "ID": 1000,
    "baudrate": 1001,
    "angleSet": 1486,
    "forceSet": 1498,
    "speedSet": 1522,
    "angleAct": 1546,
    "forceAct": 1582,
    "errCode": 1606,
    "statusCode": 1612,
    "temp": 1618,
    "actionSeq": 2320,
    "actionRun": 2322,
}


class ModbusTcpError(RuntimeError):
    pass


class ModbusTcpClient:
    def __init__(self, host: str, port: int, timeout: float, unit_id: int) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.unit_id = unit_id
        self._transaction_id = 0
        self._sock: socket.socket | None = None

    def __enter__(self) -> "ModbusTcpClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        self._sock = socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        )
        self._sock.settimeout(self.timeout)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def read_holding_registers(self, address: int, count: int) -> List[int]:
        if self._sock is None:
            raise ModbusTcpError("not connected")
        if not 1 <= count <= 125:
            raise ValueError("Modbus register count must be 1..125")

        self._transaction_id = (self._transaction_id + 1) & 0xFFFF
        function = 0x03
        pdu = struct.pack(">BHH", function, address, count)
        mbap = struct.pack(
            ">HHHB", self._transaction_id, 0, len(pdu) + 1, self.unit_id
        )
        self._sock.sendall(mbap + pdu)

        header = self._recv_exact(7)
        rx_tid, protocol_id, length, rx_unit_id = struct.unpack(">HHHB", header)
        if rx_tid != self._transaction_id:
            raise ModbusTcpError(
                f"transaction mismatch: sent {self._transaction_id}, got {rx_tid}"
            )
        if protocol_id != 0:
            raise ModbusTcpError(f"unexpected protocol id: {protocol_id}")
        if rx_unit_id != self.unit_id:
            raise ModbusTcpError(f"unexpected unit id: {rx_unit_id}")

        payload = self._recv_exact(length - 1)
        if not payload:
            raise ModbusTcpError("empty Modbus payload")

        rx_function = payload[0]
        if rx_function & 0x80:
            code = payload[1] if len(payload) > 1 else None
            raise ModbusTcpError(f"Modbus exception for 0x{function:02x}: {code}")
        if rx_function != function:
            raise ModbusTcpError(f"unexpected function: 0x{rx_function:02x}")

        byte_count = payload[1]
        expected_bytes = count * 2
        if byte_count != expected_bytes:
            raise ModbusTcpError(
                f"unexpected byte count: got {byte_count}, expected {expected_bytes}"
            )
        values = struct.unpack(">" + "H" * count, payload[2 : 2 + byte_count])
        return list(values)

    def _recv_exact(self, size: int) -> bytes:
        if self._sock is None:
            raise ModbusTcpError("not connected")
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._sock.recv(size - len(chunks))
            if not chunk:
                raise ModbusTcpError("connection closed while reading")
            chunks.extend(chunk)
        return bytes(chunks)


def split_bytes_from_words(words: Sequence[int]) -> List[int]:
    values: List[int] = []
    for word in words:
        values.append(word & 0xFF)
        values.append((word >> 8) & 0xFF)
    return values


def fmt(values: Iterable[int]) -> str:
    return " ".join(str(v) for v in values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only RH56E2/RH56-series Modbus TCP probe"
    )
    parser.add_argument("--ip", default="192.168.11.210")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    print(
        f"Connecting to RH56 hand at {args.ip}:{args.port}, "
        f"unit_id={args.unit_id}, timeout={args.timeout}s"
    )

    try:
        with ModbusTcpClient(args.ip, args.port, args.timeout, args.unit_id) as client:
            print("TCP connected")

            hand_id = client.read_holding_registers(REG["ID"], 1)
            baudrate = client.read_holding_registers(REG["baudrate"], 1)
            angle_act = client.read_holding_registers(REG["angleAct"], 6)
            force_act = client.read_holding_registers(REG["forceAct"], 6)
            err_words = client.read_holding_registers(REG["errCode"], 3)
            status_words = client.read_holding_registers(REG["statusCode"], 3)
            temp_words = client.read_holding_registers(REG["temp"], 3)

            print(f"ID: {fmt(hand_id)}")
            print(f"baudrate register: {fmt(baudrate)}")
            print(f"angleAct[6]: {fmt(angle_act)}")
            print(f"forceAct[6]: {fmt(force_act)}")
            print(f"errCode[6 bytes]: {fmt(split_bytes_from_words(err_words))}")
            print(f"statusCode[6 bytes]: {fmt(split_bytes_from_words(status_words))}")
            print(f"temp[6 bytes]: {fmt(split_bytes_from_words(temp_words))}")

    except (OSError, ModbusTcpError) as exc:
        print(f"Probe failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
