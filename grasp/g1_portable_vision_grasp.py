#!/usr/bin/env python3
"""Minimal RGB-D YOLO-World grasp entry point.

Only three runtime parameters are exposed:
  --camera-pitch
  --grasp-offset X Y Z
  --object NAME

The remaining values are fixed to the currently tested G1 + RealSense +
RH56E2 setup. The script does not save RGB/depth images locally.
"""

import argparse
import base64
import io
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GRASP_SCRIPT = ROOT / "example" / "g1" / "grasp" / "g1_right_arm_body_ik_then_hand.py"

INTERFACE_CANDIDATES = ("wlp0s20f3", "eth1", "eth0", "enx68da73a54ce1")
SSH_HOST = "unitree@192.168.123.164"
SSH_PASSWORD = os.environ.get("ROBOT_SSH_PASSWORD", "")
DEPTH_SERIAL = "243322071448"
MODEL_PATH = "yolov8s-world.pt"

CAMERA_IN_PELVIS_XYZ = np.asarray([0.05366, 0.01753, 0.47387], dtype=float)

CONF = 0.10
CAPTURE_TIMEOUT = 60
WORKSPACE = [0.10, 0.42, -0.35, 0.12, 0.05, 0.50]


REMOTE_CAPTURE_CODE = r"""
import base64
import io
import json
import os
import sys
import time

import numpy as np
import pyrealsense2 as rs

serial = os.environ.get("DEPTH_DEVICE_SERIAL", "").strip()
pipeline = rs.pipeline()
config = rs.config()
if serial:
    config.enable_device(serial)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
try:
    align = rs.align(rs.stream.color)
    frames = None
    for _ in range(30):
        frames = align.process(pipeline.wait_for_frames(5000))
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        raise RuntimeError("missing depth or color frame")

    color_intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    color = np.asanyarray(color_frame.get_data()).copy()
    depth = np.asanyarray(depth_frame.get_data()).copy()

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        color_bgr=color,
        depth_raw=depth,
        depth_scale=np.asarray(depth_scale, dtype=np.float64),
        fx=np.asarray(color_intr.fx, dtype=np.float64),
        fy=np.asarray(color_intr.fy, dtype=np.float64),
        cx=np.asarray(color_intr.ppx, dtype=np.float64),
        cy=np.asarray(color_intr.ppy, dtype=np.float64),
    )
    sys.stdout.write("__G1_RGBD_NPZ_BEGIN__\n")
    sys.stdout.write(base64.b64encode(buf.getvalue()).decode("ascii"))
    sys.stdout.write("\n__G1_RGBD_NPZ_END__\n")
    sys.stdout.flush()
finally:
    pipeline.stop()
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal vision-to-grasp script.")
    parser.add_argument("--interface", default=None, help="DDS network interface, for example wlp0s20f3.")
    parser.add_argument("--host", default=SSH_HOST, help="Robot SSH target, for example unitree@192.168.1.103.")
    parser.add_argument("--password", default=SSH_PASSWORD, help="Robot SSH password.")
    parser.add_argument("--camera-pitch", type=float, default=0.35)
    parser.add_argument("--grasp-offset", nargs=3, type=float, default=[-0.22, 0.04, 0.25])
    parser.add_argument("--object", default="bottle")
    parser.add_argument("--hand-ip", default="192.168.124.210")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--announce-text", default="抓取完成")
    parser.add_argument("--announce-speaker-id", type=int, default=0)
    parser.add_argument("--no-announce", action="store_true")
    return parser.parse_args()


def interface_is_up(name):
    operstate = Path("/sys/class/net") / name / "operstate"
    if not operstate.exists():
        return False
    return operstate.read_text(encoding="utf-8").strip() in ("up", "unknown")


def detect_interface(preferred=None):
    if preferred in ("", "none", "local"):
        return None
    if preferred:
        return preferred

    env_interface = os.environ.get("G1_DDS_INTERFACE", "").strip()
    if env_interface:
        return env_interface

    names = {path.name for path in Path("/sys/class/net").iterdir()} if Path("/sys/class/net").exists() else set()
    if not names:
        try:
            names = {
                line.split(": ", 2)[1]
                for line in subprocess.check_output(["ip", "-o", "link", "show"], text=True).splitlines()
                if ": " in line
            }
        except Exception:
            names = set()

    for name in INTERFACE_CANDIDATES:
        if name in names and interface_is_up(name):
            return name

    for name in INTERFACE_CANDIDATES:
        if name in names:
            return name
    return INTERFACE_CANDIDATES[0]


def roty(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def optical_to_pelvis_rotation(camera_pitch_rad):
    # RealSense optical frame: x right, y down, z forward.
    # Approximate pelvis frame: x forward, y left, z up.
    level = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    return roty(camera_pitch_rad) @ level


def camera_to_pelvis(point_camera, camera_pitch):
    return CAMERA_IN_PELVIS_XYZ + optical_to_pelvis_rotation(camera_pitch) @ point_camera


def run_ssh_capture(args):
    try:
        import pexpect
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: pexpect") from exc

    remote_bootstrap = (
        "import base64; "
        f"exec(base64.b64decode({base64.b64encode(REMOTE_CAPTURE_CODE.encode()).decode()!r}))"
    )
    remote = f"DEPTH_DEVICE_SERIAL={shlex.quote(DEPTH_SERIAL)} python3 -c {shlex.quote(remote_bootstrap)}"
    cmd = (
        "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-o LogLevel=ERROR {args.host} {shlex.quote(remote)}"
    )
    child = pexpect.spawn(cmd, encoding="utf-8", timeout=CAPTURE_TIMEOUT, maxread=1024 * 1024)
    output = []
    while True:
        index = child.expect([r"(?i)password:", pexpect.EOF, pexpect.TIMEOUT])
        output.append(child.before or "")
        if index == 0:
            child.sendline(args.password)
            continue
        if index == 1:
            break
        child.close(force=True)
        raise RuntimeError("SSH RGB-D capture timed out")

    text = "".join(output)
    if child.exitstatus not in (0, None):
        raise RuntimeError(f"SSH RGB-D capture failed: {child.exitstatus}\n{text[-1000:]}")

    start = text.find("__G1_RGBD_NPZ_BEGIN__")
    end = text.find("__G1_RGBD_NPZ_END__")
    if start < 0 or end < 0:
        raise RuntimeError(f"RGB-D payload markers not found\n{text[-1000:]}")
    payload = text[start + len("__G1_RGBD_NPZ_BEGIN__") : end].strip()
    data = np.load(io.BytesIO(base64.b64decode(payload)))
    return {
        "color_bgr": data["color_bgr"],
        "depth_raw": data["depth_raw"],
        "depth_scale": float(data["depth_scale"]),
        "fx": float(data["fx"]),
        "fy": float(data["fy"]),
        "cx": float(data["cx"]),
        "cy": float(data["cy"]),
    }


def run_local_capture():
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyrealsense2 is not available locally") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if DEPTH_SERIAL:
        config.enable_device(DEPTH_SERIAL)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    try:
        align = rs.align(rs.stream.color)
        frames = None
        for _ in range(30):
            frames = align.process(pipeline.wait_for_frames(5000))
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("missing depth or color frame")

        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        return {
            "color_bgr": np.asanyarray(color_frame.get_data()).copy(),
            "depth_raw": np.asanyarray(depth_frame.get_data()).copy(),
            "depth_scale": float(depth_sensor.get_depth_scale()),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
        }
    finally:
        pipeline.stop()


def run_capture(args):
    try:
        print("capture: local RealSense", flush=True)
        return run_local_capture()
    except Exception as exc:
        print(f"capture: local RealSense failed: {exc}", flush=True)
        print("capture: fallback to SSH RealSense", flush=True)
        return run_ssh_capture(args)


def load_model(object_name):
    try:
        from ultralytics import YOLOWorld
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: ultralytics") from exc

    model_candidates = [
        Path.cwd() / MODEL_PATH,
        Path(__file__).resolve().parent / MODEL_PATH,
        ROOT / MODEL_PATH,
    ]
    model_path = next((path for path in model_candidates if path.exists()), Path(MODEL_PATH))
    model = YOLOWorld(str(model_path))
    model.set_classes([object_name])
    return model, str(model_path)


def detect_object(model, image_bgr, object_name):
    results = model.predict(source=image_bgr, conf=CONF, imgsz=640, stream=False, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no result")
    result = results[0]
    names = result.names or {}
    best = None
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, str(cls_id))
        conf = float(box.conf[0])
        if class_name != object_name:
            continue
        xyxy = [float(v) for v in box.xyxy[0]]
        det = {
            "class_name": class_name,
            "confidence": conf,
            "bbox_xyxy": xyxy,
            "center_xy": [(xyxy[0] + xyxy[2]) * 0.5, (xyxy[1] + xyxy[3]) * 0.5],
        }
        if best is None or conf > best["confidence"]:
            best = det
    if best is None:
        raise RuntimeError(f"no detection for object={object_name!r} with conf >= {CONF}")
    return best


def depth_point_for_detection(capture, det):
    depth = capture["depth_raw"]
    scale = capture["depth_scale"]
    x1, y1, x2, y2 = det["bbox_xyxy"]
    cx_det, cy_det = det["center_xy"]

    w = max(6, int((x2 - x1) * 0.35))
    h = max(6, int((y2 - y1) * 0.35))
    x0 = max(0, int(round(cx_det - w * 0.5)))
    x3 = min(depth.shape[1], int(round(cx_det + w * 0.5)))
    y0 = max(0, int(round(cy_det - h * 0.5)))
    y3 = min(depth.shape[0], int(round(cy_det + h * 0.5)))

    crop = depth[y0:y3, x0:x3].astype(np.float32) * scale
    valid = crop[np.isfinite(crop) & (crop > 0.05)]
    if valid.size == 0:
        raise RuntimeError("no valid depth samples in detection crop")

    z = float(np.median(valid))
    u = float(cx_det)
    v = float(cy_det)
    x = (u - capture["cx"]) * z / capture["fx"]
    y = (v - capture["cy"]) * z / capture["fy"]
    return np.asarray([x, y, z], dtype=float), {
        "depth_crop_xyxy": [x0, y0, x3, y3],
        "valid_depth_samples": int(valid.size),
        "depth_median_m": z,
    }


def in_workspace(point):
    xmin, xmax, ymin, ymax, zmin, zmax = WORKSPACE
    return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax and zmin <= point[2] <= zmax


def build_grasp_command(target, args):
    network_interface = detect_interface(args.interface)
    cmd = [
        sys.executable,
        str(GRASP_SCRIPT),
        "--target-frame",
        "pelvis",
        "--base-link-in-pelvis-xyz",
        "-0.003680",
        "0.000030",
        "0.460180",
        "--base-link-in-pelvis-rpy",
        "0.000000",
        "0.000000",
        "0.000000",
        "--joint-signs",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "--arm-dof",
        "7",
        "--tolerance",
        "0.025",
        "--duration",
        "2.0",
        "--drop-duration",
        "10.0",
        "--hold-after-hand",
        "1.5",
        "--hold-after-drop",
        "1.0",
        "--max-joint-speed",
        "0.12",
        "--max-joint-delta",
        "2.8",
        "--arm-sdk-weight",
        "0.45",
        "--arm-sdk-ramp-in-ratio",
        "0.25",
        "--release-duration",
        "1.0",
        "--kp",
        "30.0",
        "--kd",
        "2.0",
        "--hand-ip",
        args.hand_ip,
        "--hand-port",
        str(args.hand_port),
        "--hand-unit-id",
        "1",
        "--hand-timeout",
        "2.0",
        "--hand-speed",
        "200",
        "--hand-force",
        "1000",
        "--hand-hold",
        "5.0",
        "--hand-open-angle",
        "1000",
        "1000",
        "1000",
        "1000",
        "1000",
        "0",
        "--hand-grasp-angle",
        "0",
        "0",
        "0",
        "0",
        "0",
        "400",
        "--hand-sequence",
        "grasp",
        "--yes",
        "--no-prompt",
        "--target",
        f"{target[0]:.6f}",
        f"{target[1]:.6f}",
        f"{target[2]:.6f}",
        "--wrist-roll-offset",
        "0.0",
        "--wrist-pitch-offset",
        "-0.5",
        "--wrist-yaw-offset",
        "0.0",
        "--no-return-after-drop",
    ]
    if network_interface is not None:
        cmd.insert(2, network_interface)
    if args.no_announce:
        cmd.append("--no-announce")
    else:
        cmd.extend(
            [
                "--announce-text",
                args.announce_text,
                "--announce-speaker-id",
                str(args.announce_speaker_id),
            ]
        )
    return cmd


def main():
    args = parse_args()

    capture = run_capture(args)
    model, model_path = load_model(args.object)
    det = detect_object(model, capture["color_bgr"], args.object)
    point_camera, depth_stats = depth_point_for_detection(capture, det)
    point_pelvis = camera_to_pelvis(point_camera, args.camera_pitch)
    grasp_target = point_pelvis + np.asarray(args.grasp_offset, dtype=float)

    summary = {
        "object": args.object,
        "model": model_path,
        "detection": det,
        "depth_stats": depth_stats,
        "point_camera_m": point_camera.tolist(),
        "point_pelvis_before_offset_m": point_pelvis.tolist(),
        "grasp_offset_m": list(args.grasp_offset),
        "grasp_target_m": grasp_target.tolist(),
        "workspace_ok": bool(in_workspace(grasp_target)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    cmd = build_grasp_command(grasp_target, args)
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
