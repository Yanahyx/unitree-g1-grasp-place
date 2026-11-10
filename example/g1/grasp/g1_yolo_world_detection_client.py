#!/usr/bin/env python3
"""Client for the persistent G1 YOLO-World detection server."""

import argparse
import json
import time
import urllib.error
import urllib.request


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Query persistent YOLO-World detector.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--object", default="bottle")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--health", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base = f"http://{args.host}:{args.port}"
    if args.health:
        print(json.dumps(get_json(base + "/health", args.timeout), indent=2, ensure_ascii=False))
        return 0

    for index in range(args.repeat):
        t0 = time.monotonic()
        result = post_json(
            base + "/detect",
            {"object": args.object, "conf": args.conf, "imgsz": args.imgsz},
            args.timeout,
        )
        result["client_round_trip_s"] = time.monotonic() - t0
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if index + 1 < args.repeat:
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
