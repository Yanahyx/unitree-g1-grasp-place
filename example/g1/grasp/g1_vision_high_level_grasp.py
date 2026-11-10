#!/usr/bin/env python3
"""Run RGB-D YOLO detection, then call the original high_level grasp script."""

import argparse
import os
import json
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Detect an object with RGB-D YOLO and pass its xyz to high_level/g1_move_right_hand_ik_once.py."
    )
    parser.add_argument("--interface", default="enx68da73a54ce1")
    parser.add_argument("--password", default=os.environ.get("ROBOT_SSH_PASSWORD", ""))
    parser.add_argument("--host", default="unitree@192.168.123.164")
    parser.add_argument("--depth-device-serial", default="243322071448")
    parser.add_argument("--rgb-source", choices=("realsense", "head", "videohub", "auto"), default="realsense")
    parser.add_argument("--capture-timeout", type=float, default=40.0)
    parser.add_argument("--output-dir", default="tmp/g1_vision_high_level_grasp")

    parser.add_argument("--model", default="yolov8s-world.pt")
    parser.add_argument("--yolo-world", dest="yolo_world", action="store_true", default=True)
    parser.add_argument("--no-yolo-world", dest="yolo_world", action="store_false")
    parser.add_argument("--yoloe", action="store_true")
    parser.add_argument("--text-classes", nargs="+", default=["cup"])
    parser.add_argument("--object", default="cup")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--workspace", nargs=6, type=float, default=[0.10, 0.42, -0.35, 0.12, 0.05, 0.50])

    parser.add_argument(
        "--fixed-grasp-target",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Use this target instead of the vision xyz; useful for high_level regression tests.",
    )
    parser.add_argument(
        "--grasp-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Offset added to the vision xyz before calling the high_level grasp script.",
    )
    parser.add_argument("--allow-not-ready", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run vision and print the high_level command without executing it.")

    parser.add_argument("--move-duration", type=float, default=2.0)
    parser.add_argument("--hold-duration", type=float, default=1.5)
    parser.add_argument("--return-duration", type=float, default=2.0)
    parser.add_argument("--kp", type=float, default=30.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--hold-kp", type=float, default=18.0)
    parser.add_argument("--hold-kd", type=float, default=1.0)
    parser.add_argument("--no-hand-grasp", action="store_true")

    parser.add_argument("--hand-ip", default="192.168.124.210")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--hand-unit-id", type=int, default=1)
    parser.add_argument("--hand-timeout", type=float, default=2.0)
    parser.add_argument("--hand-speed", type=int, default=200)
    parser.add_argument("--hand-force", type=int, default=1000)
    parser.add_argument("--hand-open-angle", nargs=6, type=int, default=[1000, 1000, 1000, 1000, 1000, 0])
    parser.add_argument("--hand-grasp-angle", nargs=6, type=int, default=[0, 0, 0, 0, 0, 400])
    parser.add_argument("--yes", action="store_true", help="Required to execute high_level grasp motion.")
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
    if args.fixed_grasp_target is not None:
        source = "fixed"
        target = parse_vec3(args.fixed_grasp_target)
    else:
        if "target_palm_torso_m" not in vision_result:
            raise RuntimeError("vision result has no target_palm_torso_m")
        source = "vision"
        target = parse_vec3(vision_result["target_palm_torso_m"])
    target = target + parse_vec3(args.grasp_offset)
    return source, target


def build_high_level_command(args, target):
    script = HIGH_LEVEL_DIR / "g1_move_right_hand_ik_once.py"
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
        "--move-duration",
        str(args.move_duration),
        "--hold-duration",
        str(args.hold_duration),
        "--return-duration",
        str(args.return_duration),
        "--kp",
        str(args.kp),
        "--kd",
        str(args.kd),
        "--hold-kp",
        str(args.hold_kp),
        "--hold-kd",
        str(args.hold_kd),
        "--hand-ip",
        args.hand_ip,
        "--hand-port",
        str(args.hand_port),
        "--hand-unit-id",
        str(args.hand_unit_id),
        "--hand-timeout",
        str(args.hand_timeout),
        "--hand-speed",
        str(args.hand_speed),
        "--hand-force",
        str(args.hand_force),
        "--hand-open-angle",
        *[str(value) for value in args.hand_open_angle],
        "--hand-grasp-angle",
        *[str(value) for value in args.hand_grasp_angle],
        "--yes",
    ]
    if args.no_hand_grasp:
        cmd.append("--no-hand-grasp")
    return cmd


def main():
    args = build_parser().parse_args()
    if not args.dry_run and not args.yes:
        raise SystemExit("Refusing to execute high_level grasp without --yes. Use --dry-run to only print the command.")

    vision_result = run_vision_probe(build_vision_args(args))
    if not vision_result.get("ready_for_arm") and not args.allow_not_ready and args.fixed_grasp_target is None:
        print(json.dumps(vision_result, indent=2, ensure_ascii=False))
        raise SystemExit("vision target is not ready_for_arm; pass --allow-not-ready to override.")

    source, target = select_target(args, vision_result)
    cmd = build_high_level_command(args, target)
    summary = {
        "target_source": source,
        "high_level_target": target.tolist(),
        "vision_ready_for_arm": bool(vision_result.get("ready_for_arm")),
        "selected_detection": vision_result.get("selected_detection"),
        "high_level_command": " ".join(cmd),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
