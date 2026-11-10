#!/usr/bin/env python3
"""Start or reuse local G1 detectors without issuing robot commands."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
SERVICES = {
    "grasp": (8765, "bottle", "g1_yolo_world_detection_server.py"),
    "place": (8766, "cardboard box", "g1_yolo_world_place_detection_server.py"),
}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def health(port, object_name, name):
    try:
        with OPENER.open("http://127.0.0.1:%d/health" % port, timeout=2) as r:
            data = json.load(r)
        return (data.get("ok") is True and data.get("object") == object_name
                and data.get("source") == "http-rgbd"
                and data.get("camera", {}).get("rgbd_npz_url") == "http://127.0.0.1:18082/rgbd.npz"
                and (name != "place" or data.get("supports_image_npy") is True))
    except (OSError, ValueError, AttributeError):
        return False


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def detector_pids(filename, port):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode().split("\0")
            if not any(Path(arg).name == filename for arg in args if arg):
                continue
            actual = 8765
            for i, arg in enumerate(args):
                if arg == "--port":
                    actual = int(args[i + 1])
                elif arg.startswith("--port="):
                    actual = int(arg.split("=", 1)[1])
            if actual == port:
                found.append(int(entry.name))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return found


def ensure(name, timeout):
    port, object_name, filename = SERVICES[name]
    script = ROOT / "example/g1/grasp" / filename
    log_path = Path("/tmp") / (script.stem + ".log")
    state_dir = Path.home() / ".cache/unitree-detectors"
    state_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with (state_dir / (name + ".lock")).open("a") as lock:
        announced = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not announced:
                    print("[%s] Another launcher is starting this service; waiting." % name, flush=True)
                    announced = True
                if time.monotonic() >= deadline:
                    raise RuntimeError("%s: startup lock timeout; no duplicate started" % name)
                time.sleep(1)

        pids = detector_pids(filename, port)
        if pids and health(port, object_name, name):
            print("[%s] Already ready on %d; reusing PID(s) %s." % (name, port, pids), flush=True)
            return
        if listening(port):
            raise RuntimeError("%s: port %d occupied but expected service is not healthy. No process killed. Check %s" % (name, port, log_path))
        child = None
        if pids:
            print("[%s] Existing process %s is initializing; no duplicate started." % (name, pids), flush=True)
        else:
            if not script.is_file():
                raise RuntimeError("Missing detector script: %s" % script)
            argv = [sys.executable, "-u", str(script), "--host", "127.0.0.1", "--port", str(port),
                    "--object", object_name, "--source", "http-rgbd", "--fallback-source", "none",
                    "--rgbd-npz-url", "http://127.0.0.1:18082/rgbd.npz", "--no-warmup-predict"]
            with log_path.open("ab") as log:
                log.write(("\n=== launcher %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S")).encode())
                log.flush()
                child = subprocess.Popen(argv, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, close_fds=True)
            print("[%s] Started PID %d; waiting for health. Log: %s" % (name, child.pid, log_path), flush=True)
        while time.monotonic() < deadline:
            if child is not None and child.poll() is not None:
                raise RuntimeError("%s: detector exited (%s); log: %s" % (name, child.returncode, log_path))
            if detector_pids(filename, port) and health(port, object_name, name):
                print("[%s] Ready on %d." % (name, port), flush=True)
                return
            if not detector_pids(filename, port):
                raise RuntimeError("%s: detector disappeared; log: %s" % (name, log_path))
            time.sleep(1)
        raise RuntimeError("%s: health timeout; existing process left running. Retry this launcher, not nohup. Log: %s" % (name, log_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=["all", "grasp", "place"], nargs="?", default="all")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.timeout <= 1800:
        parser.error("--timeout must be between 0 and 1800 seconds")
    failed = False
    for name in SERVICES if args.service == "all" else [args.service]:
        try:
            ensure(name, args.timeout)
        except (OSError, RuntimeError) as exc:
            failed = True
            print("ERROR: %s" % exc, file=sys.stderr, flush=True)
    print("Detector health does not confirm camera frames or object detection. RGB-D service on 18082 must be running.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
