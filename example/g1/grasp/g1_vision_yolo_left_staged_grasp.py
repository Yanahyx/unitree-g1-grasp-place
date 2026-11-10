#!/usr/bin/env python3
"""Run RGB-D YOLO detection, then execute a staged G1 left-arm grasp."""

import argparse
import os
import math
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
VISION_DIR = ROOT / "example" / "g1" / "vision"
GRASP_DIR = ROOT / "example" / "g1" / "grasp"
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


def optical_to_pelvis_rotation(camera_pitch_rad):
    # RealSense optical frame: x right, y down, z forward.
    # Approximate pelvis/body frame here: x forward, y left, z up.
    r_pelvis_camera_level = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    return roty(camera_pitch_rad) @ r_pelvis_camera_level


def camera_point_to_pelvis(point_camera, camera_xyz, camera_pitch_rad):
    point_camera = np.asarray(point_camera, dtype=float)
    camera_xyz = np.asarray(camera_xyz, dtype=float)
    return camera_xyz + optical_to_pelvis_rotation(camera_pitch_rad) @ point_camera


def build_parser():
    parser = argparse.ArgumentParser(
        description="Detect an object with RGB-D YOLO and run a staged left-arm grasp."
    )
    parser.add_argument("--interface", default="enx68da73a54ce1")
    parser.add_argument("--password", default=os.environ.get("ROBOT_SSH_PASSWORD", ""))
    parser.add_argument("--host", default="unitree@192.168.123.164")
    parser.add_argument("--depth-device-serial", default="243322071448")
    parser.add_argument("--rgb-source", choices=("realsense", "head", "videohub", "auto"), default="realsense")
    parser.add_argument("--capture-timeout", type=float, default=40.0)
    parser.add_argument("--output-dir", default="tmp/g1_vision_yolo_left_staged_grasp")

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
        help="Use this grasp target instead of the vision xyz; useful for regression testing.",
    )
    parser.add_argument(
        "--grasp-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Offset added to the vision xyz before passing it to the grasp script.",
    )
    parser.add_argument("--pregrasp-offset", nargs=3, type=float, default=[-0.10, 0.06, 0.12])
    parser.add_argument("--align-offset", nargs=3, type=float, default=[-0.05, 0.04, 0.05])
    parser.add_argument("--approach-offset", nargs=3, type=float, default=[0.0, 0.02, 0.02])
    parser.add_argument("--lift-offset", nargs=3, type=float, default=[0.0, 0.02, 0.18])
    parser.add_argument("--no-feedback-control", dest="feedback_control", action="store_false")
    parser.set_defaults(feedback_control=True)
    parser.add_argument("--feedback-pos-threshold", type=float, default=0.035)
    parser.add_argument("--feedback-step-m", type=float, default=0.025)
    parser.add_argument("--feedback-log-period", type=float, default=0.5)
    parser.add_argument("--feedback-timeout-scale", type=float, default=1.8)
    parser.add_argument(
        "--vision-frame",
        choices=("camera", "pelvis"),
        default="camera",
        help="Interpret vision xyz as raw camera coordinates or already-estimated pelvis coordinates.",
    )
    parser.add_argument(
        "--camera-in-pelvis-xyz",
        nargs=3,
        type=float,
        default=[0.05366, 0.01753, 0.47387],
        metavar=("X", "Y", "Z"),
        help="Approximate head/RealSense optical center in pelvis frame, meters.",
    )
    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=0.0,
        help="Approximate camera pitch correction in radians. Positive pitches camera-forward points upward in pelvis.",
    )
    parser.add_argument("--target-frame", choices=("base_link", "body", "pelvis", "map"), default="pelvis")
    parser.add_argument("--allow-not-ready", action="store_true", help="Continue even if vision ready_for_arm is false.")
    parser.add_argument("--dry-run", action="store_true", help="Run vision and print the grasp command without executing it.")

    parser.add_argument("--base-link-in-pelvis-xyz", nargs=3, type=float, default=[-0.00368, 0.00003, 0.46018])
    parser.add_argument("--base-link-in-pelvis-rpy", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--joint-signs", nargs=7, type=float, default=[1, 1, 1, 1, 1, 1, 1])
    parser.add_argument("--arm-dof", type=int, choices=(5, 7), default=7)
    parser.add_argument("--tolerance", type=float, default=0.025)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--drop-duration", type=float, default=10.0)
    parser.add_argument("--hold-after-hand", type=float, default=1.5)
    parser.add_argument("--hold-after-drop", type=float, default=1.0)
    parser.add_argument("--max-joint-speed", type=float, default=0.12)
    parser.add_argument("--max-joint-delta", type=float, default=2.80)
    parser.add_argument("--arm-sdk-weight", type=float, default=0.45)
    parser.add_argument("--arm-sdk-ramp-in-ratio", type=float, default=0.25)
    parser.add_argument("--release-duration", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=30.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--return-after-drop", action="store_true")
    parser.add_argument("--wrist-roll-offset", type=float, default=0.0)
    parser.add_argument("--wrist-pitch-offset", type=float, default=-0.50)
    parser.add_argument("--wrist-yaw-offset", type=float, default=0.0)

    parser.add_argument("--hand-ip", default="192.168.124.211")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--hand-unit-id", type=int, default=1)
    parser.add_argument("--hand-timeout", type=float, default=2.0)
    parser.add_argument("--hand-speed", type=int, default=200)
    parser.add_argument("--hand-force", type=int, default=1000)
    parser.add_argument("--hand-hold", type=float, default=5.0)
    parser.add_argument("--hand-open-angle", nargs=6, type=int, default=[1000, 1000, 1000, 1000, 1000, 0])
    parser.add_argument("--hand-grasp-angle", nargs=6, type=int, default=[0, 0, 0, 0, 0, 400])
    parser.add_argument("--hand-sequence", choices=("open", "grasp", "open-grasp", "grasp-open"), default="grasp")
    parser.add_argument("--yes", action="store_true", help="Required to execute grasp motion.")
    return parser


def namespace_from_parser(parser):
    return parser.parse_args([])


def build_vision_args(args):
    vision_args = namespace_from_parser(build_vision_parser())
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


def select_grasp_target(args, vision_result):
    if args.fixed_grasp_target is not None:
        source = "fixed"
        target = parse_vec3(args.fixed_grasp_target)
    else:
        if args.vision_frame == "camera":
            if "point_camera_m" not in vision_result:
                raise RuntimeError("vision result has no point_camera_m")
            source = "vision_camera_approx_pelvis"
            target = camera_point_to_pelvis(
                vision_result["point_camera_m"],
                args.camera_in_pelvis_xyz,
                args.camera_pitch,
            )
        else:
            if "target_palm_torso_m" not in vision_result:
                raise RuntimeError("vision result has no target_palm_torso_m")
            source = "vision_pelvis_passthrough"
            target = parse_vec3(vision_result["target_palm_torso_m"])
    target = target + parse_vec3(args.grasp_offset)
    return source, target


def build_grasp_command(args, target):
    script = GRASP_DIR / "g1_left_arm_staged_grasp.py"
    cmd = [
        sys.executable,
        str(script),
        args.interface,
        "--target-frame",
        args.target_frame,
        "--base-link-in-pelvis-xyz",
        *[f"{value:.6f}" for value in args.base_link_in_pelvis_xyz],
        "--base-link-in-pelvis-rpy",
        *[f"{value:.6f}" for value in args.base_link_in_pelvis_rpy],
        "--joint-signs",
        *[str(int(value)) if float(value).is_integer() else str(value) for value in args.joint_signs],
        "--arm-dof",
        str(args.arm_dof),
        "--tolerance",
        str(args.tolerance),
        "--duration",
        str(args.duration),
        "--hold-after-hand",
        str(args.hold_after_hand),
        "--max-joint-speed",
        str(args.max_joint_speed),
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--arm-sdk-weight",
        str(args.arm_sdk_weight),
        "--arm-sdk-ramp-in-ratio",
        str(args.arm_sdk_ramp_in_ratio),
        "--release-duration",
        str(args.release_duration),
        "--kp",
        str(args.kp),
        "--kd",
        str(args.kd),
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
        "--hand-hold",
        str(args.hand_hold),
        "--hand-open-angle",
        *[str(value) for value in args.hand_open_angle],
        "--hand-grasp-angle",
        *[str(value) for value in args.hand_grasp_angle],
        "--pregrasp-offset",
        *[f"{value:.6f}" for value in args.pregrasp_offset],
        "--align-offset",
        *[f"{value:.6f}" for value in args.align_offset],
        "--approach-offset",
        *[f"{value:.6f}" for value in args.approach_offset],
        "--lift-offset",
        *[f"{value:.6f}" for value in args.lift_offset],
        "--feedback-pos-threshold",
        str(args.feedback_pos_threshold),
        "--feedback-step-m",
        str(args.feedback_step_m),
        "--feedback-log-period",
        str(args.feedback_log_period),
        "--feedback-timeout-scale",
        str(args.feedback_timeout_scale),
        "--yes",
        "--no-prompt",
        "--target",
        *[f"{value:.6f}" for value in target],
        "--wrist-roll-offset",
        str(args.wrist_roll_offset),
        "--wrist-pitch-offset",
        str(args.wrist_pitch_offset),
        "--wrist-yaw-offset",
        str(args.wrist_yaw_offset),
    ]
    if args.feedback_control:
        cmd.append("--feedback-control")
    if not args.return_after_drop:
        cmd.append("--no-return-after-drop")
    return cmd


def main():
    args = build_parser().parse_args()
    if not args.dry_run and not args.yes:
        raise SystemExit("Refusing to execute grasp without --yes. Use --dry-run to only print the command.")

    vision_result = run_vision_probe(build_vision_args(args))
    if not vision_result.get("ready_for_arm") and not args.allow_not_ready and args.fixed_grasp_target is None:
        print(json.dumps(vision_result, indent=2, ensure_ascii=False))
        raise SystemExit("vision target is not ready_for_arm; pass --allow-not-ready to override.")

    source, target = select_grasp_target(args, vision_result)
    cmd = build_grasp_command(args, target)
    summary = {
        "target_source": source,
        "grasp_target": target.tolist(),
        "vision_ready_for_arm": bool(vision_result.get("ready_for_arm")),
        "vision_frame": args.vision_frame,
        "camera_in_pelvis_xyz": list(args.camera_in_pelvis_xyz),
        "camera_pitch": args.camera_pitch,
        "pregrasp_offset": list(args.pregrasp_offset),
        "align_offset": list(args.align_offset),
        "approach_offset": list(args.approach_offset),
        "lift_offset": list(args.lift_offset),
        "feedback_control": bool(args.feedback_control),
        "feedback_pos_threshold": args.feedback_pos_threshold,
        "feedback_step_m": args.feedback_step_m,
        "selected_detection": vision_result.get("selected_detection"),
        "grasp_command": " ".join(cmd),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
