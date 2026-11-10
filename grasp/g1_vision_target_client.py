#!/usr/bin/env python3
"""Run RGB-D YOLO detection, then call high_level/g1_right_arm_target_client.py."""

import argparse
import os
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
VISION_DIR = ROOT / "example" / "g1" / "vision"
HIGH_LEVEL_DIR = ROOT / "example" / "g1" / "high_level"
if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))

from g1_rgbd_yolo_arm_probe import build_parser as build_vision_parser  # noqa: E402
from g1_rgbd_yolo_arm_probe import run as run_vision_probe  # noqa: E402


def parse_vec3(values):
    return np.asarray([float(value) for value in values], dtype=float)


def roty(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def optical_to_body_rotation(camera_pitch_rad):
    r_body_camera_level = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    return roty(camera_pitch_rad) @ r_body_camera_level


def camera_point_to_body(point_camera, camera_xyz, camera_pitch_rad):
    return np.asarray(camera_xyz, dtype=float) + optical_to_body_rotation(camera_pitch_rad) @ np.asarray(
        point_camera, dtype=float
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Detect an object with RGB-D YOLO and pass its xyz to g1_right_arm_target_client.py."
    )
    parser.add_argument("--interface", default="enx68da73a54ce1")
    parser.add_argument("--password", default=os.environ.get("ROBOT_SSH_PASSWORD", ""))
    parser.add_argument("--host", default="unitree@192.168.123.164")
    parser.add_argument("--depth-device-serial", default="243322071448")
    parser.add_argument("--rgb-source", choices=("realsense", "head", "videohub", "auto"), default="realsense")
    parser.add_argument("--capture-timeout", type=float, default=40.0)
    parser.add_argument("--output-dir", default="tmp/g1_vision_target_client")

    parser.add_argument("--model", default="yolov8s-world.pt")
    parser.add_argument("--yolo-world", dest="yolo_world", action="store_true", default=True)
    parser.add_argument("--no-yolo-world", dest="yolo_world", action="store_false")
    parser.add_argument("--yoloe", action="store_true")
    parser.add_argument("--text-classes", nargs="+", default=["cup"])
    parser.add_argument("--object", default="cup")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--min-confidence", type=float, default=0.15)
    parser.add_argument("--workspace", nargs=6, type=float, default=[0.10, 0.55, -0.35, 0.12, 0.05, 0.50])

    parser.add_argument("--vision-frame", choices=("camera", "body"), default="camera")
    parser.add_argument("--camera-in-body-xyz", nargs=3, type=float, default=[0.05366, 0.01753, 0.47387])
    parser.add_argument("--camera-pitch", type=float, default=0.35)
    parser.add_argument("--target-offset", nargs=3, type=float, default=[-0.08, 0.0, 0.08])
    parser.add_argument("--fixed-target", nargs=3, type=float, default=None)
    parser.add_argument("--allow-not-ready", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print and plan only; do not pass --execute.")

    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--hold-duration", type=float, default=1.5)
    parser.add_argument("--no-return", action="store_true")
    parser.add_argument("--arm-sdk-weight", type=float, default=0.45)
    parser.add_argument("--max-ik-error", type=float, default=0.025)
    parser.add_argument("--max-joint-delta", type=float, default=2.80)
    parser.add_argument("--max-joint-speed", type=float, default=0.12)
    parser.add_argument("--right-kp", type=float, default=30.0)
    parser.add_argument("--right-kd", type=float, default=2.0)
    parser.add_argument("--left-hold-kp", type=float, default=18.0)
    parser.add_argument("--left-hold-kd", type=float, default=1.0)
    parser.add_argument("--yes", action="store_true", help="Required to execute arm motion.")
    return parser


def default_vision_args():
    return build_vision_parser().parse_args([])


def build_vision_args(args):
    vision_args = default_vision_args()
    vision_args.host = args.host
    vision_args.password = args.password
    vision_args.depth_device_serial = args.depth_device_serial
    vision_args.rgb_source = args.rgb_source
    vision_args.capture_timeout = args.capture_timeout
    vision_args.output_dir = args.output_dir
    vision_args.model = args.model
    vision_args.yolo_world = args.yolo_world
    vision_args.yoloe = args.yoloe
    vision_args.text_classes = args.text_classes
    vision_args.object = args.object
    vision_args.class_name = None
    vision_args.conf = args.conf
    vision_args.min_confidence = args.min_confidence
    vision_args.workspace = args.workspace
    vision_args.interface = args.interface
    return vision_args


def select_target(args, vision_result):
    if args.fixed_target is not None:
        return "fixed", parse_vec3(args.fixed_target)
    if args.vision_frame == "camera":
        if "point_camera_m" not in vision_result:
            raise RuntimeError("vision result has no point_camera_m")
        target = camera_point_to_body(vision_result["point_camera_m"], args.camera_in_body_xyz, args.camera_pitch)
        return "vision_camera_approx_body", target + parse_vec3(args.target_offset)
    if "target_palm_torso_m" not in vision_result:
        raise RuntimeError("vision result has no target_palm_torso_m")
    return "vision_body_passthrough", parse_vec3(vision_result["target_palm_torso_m"]) + parse_vec3(args.target_offset)


def build_target_client_command(args, target, execute):
    script = HIGH_LEVEL_DIR / "g1_right_arm_target_client.py"
    cmd = [
        sys.executable,
        str(script),
        args.interface,
        "--x",
        f"{target[0]:.6f}",
        "--y",
        f"{target[1]:.6f}",
        "--z",
        f"{target[2]:.6f}",
        "--duration",
        str(args.duration),
        "--hold-duration",
        str(args.hold_duration),
        "--arm-sdk-weight",
        str(args.arm_sdk_weight),
        "--max-ik-error",
        str(args.max_ik_error),
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--max-joint-speed",
        str(args.max_joint_speed),
        "--workspace",
        *[str(value) for value in args.workspace],
        "--right-kp",
        str(args.right_kp),
        "--right-kd",
        str(args.right_kd),
        "--left-hold-kp",
        str(args.left_hold_kp),
        "--left-hold-kd",
        str(args.left_hold_kd),
    ]
    if args.no_return:
        cmd.append("--no-return")
    if execute:
        cmd.extend(["--execute", "--yes"])
    return cmd


def main():
    args = build_parser().parse_args()
    execute = not args.dry_run
    if execute and not args.yes:
        raise SystemExit("Refusing to execute arm target client without --yes. Use --dry-run to only plan.")

    vision_result = run_vision_probe(build_vision_args(args))
    if not vision_result.get("ready_for_arm") and not args.allow_not_ready and args.fixed_target is None:
        print(json.dumps(vision_result, indent=2, ensure_ascii=False))
        raise SystemExit("vision target is not ready_for_arm; pass --allow-not-ready to override.")

    source, target = select_target(args, vision_result)
    cmd = build_target_client_command(args, target, execute)
    print(
        json.dumps(
            {
                "target_source": source,
                "target_client_target": target.tolist(),
                "vision_ready_for_arm": bool(vision_result.get("ready_for_arm")),
                "selected_detection": vision_result.get("selected_detection"),
                "target_client_command": " ".join(cmd),
                "execute": execute,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
