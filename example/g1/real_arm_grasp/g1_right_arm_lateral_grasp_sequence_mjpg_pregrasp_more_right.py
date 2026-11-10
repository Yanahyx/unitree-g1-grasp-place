#!/usr/bin/env python3
"""Fixed-point lateral grasp sequence using HTTP MJPEG RGB for vision.

Sequence:
1. Move right hand to PREGRASP_TARGET in pelvis frame.
2. Open RH56E2 hand to the tested maximum-open preset.
3. Move the whole right hand to the final grasp target.
4. Close the hand to grasp.
5. Move back to PREGRASP_TARGET.
6. Return the right arm to its start posture and release arm_sdk.

This script is a copy of g1_right_arm_lateral_grasp_sequence.py with the
vision RGB source changed to an HTTP MJPEG stream. The original script is left
untouched as a backup.
"""

import argparse
import os
import io
import socket
import sys
import urllib.request
from argparse import Namespace
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
G1_DIR = THIS_DIR.parents[0]
GRASP_DIR = G1_DIR / "grasp"
HANDS_DIR = G1_DIR / "hands"
for path in (THIS_DIR, GRASP_DIR, HANDS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from g1_right_arm_body_ik_example import (  # noqa: E402
    G1JointIndex,
    format_joint_delta_table,
    format_joint_table,
    model_to_motor_q,
    motor_to_model_q,
    right_arm_joints,
    right_hand_pose,
    solve_position_ik,
    validate_target,
)
from g1_portable_vision_grasp import (  # noqa: E402
    camera_to_pelvis,
    depth_point_for_detection,
    detect_object,
    load_model,
    run_capture,
)
from g1_right_arm_official_style_reach import (  # noqa: E402
    LEFT_ARM_7DOF,
    WAIST_JOINTS,
    OfficialStyleArmSdkController,
    detect_interface,
    hold_group,
    joint_positions,
    joint_vector,
    joint_velocity_abs_max,
    release_group,
    run_timed_phase,
)
from rh56e2_hand_only import write_hand_pose  # noqa: E402
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient  # noqa: E402


MJPG_URL = "http://192.168.1.103:18082/color.mjpg"
DEPTH_SERIAL = "243322071448"
PREGRASP_TARGET = np.asarray([0.30, -0.40, 0.12], dtype=float)
FIXED_GRASP_Z = 0.12
DEFAULT_HAND_OPEN_ANGLE = [1000, 1000, 1000, 1000, 1000, 350]
DEFAULT_HAND_GRASP_ANGLE = [0, 0, 0, 0, 300, -1]
DEFAULT_WRIST_ROLL = 0.30
DEFAULT_WRIST_PITCH = -0.80
DEFAULT_WRIST_YAW = 0.50
WRIST_LIMITS = {
    G1JointIndex.RightWristRoll: (-1.9722, 1.9722),
    G1JointIndex.RightWristPitch: (-1.6144, 1.6144),
    G1JointIndex.RightWristYaw: (-1.6144, 1.6144),
}
WRIST_NAMES = {
    G1JointIndex.RightWristRoll: "RightWristRoll",
    G1JointIndex.RightWristPitch: "RightWristPitch",
    G1JointIndex.RightWristYaw: "RightWristYaw",
}
HAND_NETWORK_HINT = """Hand TCP preflight failed.

The RH56E2 hand is expected at 192.168.124.210:6000. If the robot was rebooted,
add the temporary secondary address on eth1, then probe the hand:

  sudo ip addr add 192.168.124.164/24 dev eth1

  cd /home/unitree/unitree_sdk2_python
  python3 example/g1/high_level/rh56e2_modbus_probe.py \\
    --ip 192.168.124.210 \\
    --port 6000

Expected probe output includes: TCP connected and ID: 1.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-point lateral right-arm grasp sequence.")
    parser.add_argument("network_interface", nargs="?", default=None, help="DDS interface, e.g. eth1.")
    parser.add_argument(
        "--pregrasp-target",
        nargs=3,
        type=float,
        default=PREGRASP_TARGET.tolist(),
        metavar=("X", "Y", "Z"),
        help="Saved first point in pelvis frame. Default: 0.30 -0.30 0.12.",
    )
    parser.add_argument("--left-distance", type=float, default=0.10, help="Pelvis-frame +Y lateral move before grasp.")
    parser.add_argument("--use-vision", action="store_true", help="Use RGB-D vision to override final grasp XY.")
    parser.add_argument("--object", default="bottle", help="Open-vocabulary object name for YOLO-World.")
    parser.add_argument("--camera-pitch", type=float, default=0.8)
    parser.add_argument("--fixed-grasp-z", type=float, default=FIXED_GRASP_Z)
    parser.add_argument("--vision-xy-offset", nargs=2, type=float, default=[0.0, 0.0], metavar=("X", "Y"))
    parser.add_argument("--vision-host", default="unitree@192.168.1.103")
    parser.add_argument("--vision-password", default=os.environ.get("ROBOT_SSH_PASSWORD", ""))
    parser.add_argument(
        "--vision-source",
        choices=("mjpg", "realsense"),
        default="mjpg",
        help="Vision source used when --use-vision is set. Default: mjpg.",
    )
    parser.add_argument("--mjpg-url", default=MJPG_URL, help="HTTP MJPEG RGB stream URL.")
    parser.add_argument(
        "--mjpg-assumed-depth",
        type=float,
        default=0.57,
        help="Fallback assumed object depth in meters if --mjpg-depth-source assumed is used.",
    )
    parser.add_argument(
        "--mjpg-depth-source",
        choices=("http-rgbd", "ros", "realsense", "assumed"),
        default="http-rgbd",
        help="Depth source paired with MJPEG RGB. Default: second RealSense HTTP rgbd.npz.",
    )
    parser.add_argument("--rgbd-npz-url", default="http://192.168.1.103:18082/rgbd.npz")
    parser.add_argument("--ros-depth-topic", default="/d455/aligned_depth_to_color/image_raw")
    parser.add_argument("--ros-camera-info-topic", default="/d455/color/camera_info")
    parser.add_argument("--ros-timeout", type=float, default=3.0)
    parser.add_argument("--depth-serial", default=DEPTH_SERIAL, help="RealSense serial for depth-only capture.")
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument("--depth-warmup-frames", type=int, default=15)
    parser.add_argument("--mjpg-fx", type=float, default=612.0, help="RGB camera fx at reference size.")
    parser.add_argument("--mjpg-fy", type=float, default=612.0, help="RGB camera fy at reference size.")
    parser.add_argument("--mjpg-cx", type=float, default=320.0, help="RGB camera cx at reference size.")
    parser.add_argument("--mjpg-cy", type=float, default=240.0, help="RGB camera cy at reference size.")
    parser.add_argument("--mjpg-reference-width", type=float, default=640.0)
    parser.add_argument("--mjpg-reference-height", type=float, default=480.0)
    parser.add_argument("--execute", action="store_true", help="Actually move the arm and hand.")
    parser.add_argument("--yes", action="store_true", help="Confirm the robot and hand have clear space.")
    parser.add_argument("--weight", type=float, default=0.75)
    parser.add_argument("--allow-full-weight", action="store_true")
    parser.add_argument("--pre-duration", type=float, default=6.0)
    parser.add_argument("--lateral-duration", type=float, default=3.0)
    parser.add_argument("--return-duration", type=float, default=5.0)
    parser.add_argument("--hold-before-open", type=float, default=0.5)
    parser.add_argument("--hold-after-open", type=float, default=0.5)
    parser.add_argument("--hold-before-grasp", type=float, default=0.5)
    parser.add_argument("--hold-after-grasp", type=float, default=0.8)
    parser.add_argument(
        "--wrist-roll-target",
        type=float,
        default=DEFAULT_WRIST_ROLL,
        help="RightWristRoll target used for the grasp wrist pose.",
    )
    parser.add_argument(
        "--wrist-pitch-target",
        type=float,
        default=DEFAULT_WRIST_PITCH,
        help="RightWristPitch target used for the grasp wrist pose.",
    )
    parser.add_argument(
        "--wrist-yaw-target",
        type=float,
        default=DEFAULT_WRIST_YAW,
        help="RightWristYaw target used for the grasp wrist pose.",
    )
    parser.add_argument(
        "--no-vertical-wrist",
        action="store_true",
        help="Disable the explicit wrist roll/pitch/yaw override.",
    )
    parser.add_argument("--release-duration", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.025)
    parser.add_argument("--max-joint-delta", type=float, default=2.2)
    parser.add_argument("--max-joint-speed", type=float, default=0.35)
    parser.add_argument("--max-start-arm-dq", type=float, default=0.8)
    parser.add_argument("--max-start-waist-dq", type=float, default=0.4)
    parser.add_argument("--kp-right", type=float, default=60.0)
    parser.add_argument("--kd-right", type=float, default=1.5)
    parser.add_argument("--kp-left", type=float, default=60.0)
    parser.add_argument("--kd-left", type=float, default=1.5)
    parser.add_argument("--kp-waist", type=float, default=60.0)
    parser.add_argument("--kd-waist", type=float, default=1.5)
    parser.add_argument("--hand-ip", default="192.168.124.210")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--hand-unit-id", type=int, default=1)
    parser.add_argument("--hand-timeout", type=float, default=2.0)
    parser.add_argument("--hand-speed", type=int, default=200)
    parser.add_argument("--hand-force", type=int, default=2000)
    parser.add_argument("--hand-hold", type=float, default=0.8)
    parser.add_argument("--hand-open-angle", nargs=6, type=int, default=DEFAULT_HAND_OPEN_ANGLE)
    parser.add_argument("--hand-grasp-angle", nargs=6, type=int, default=DEFAULT_HAND_GRASP_ANGLE)
    parser.add_argument("--skip-hand", action="store_true", help="Run arm motion only; do not command RH56E2.")
    parser.add_argument("--announce-text", default="抓取完成")
    parser.add_argument("--announce-speaker-id", type=int, default=0)
    parser.add_argument("--no-announce", action="store_true")
    return parser.parse_args()


def hand_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        ip=args.hand_ip,
        port=args.hand_port,
        unit_id=args.hand_unit_id,
        timeout=args.hand_timeout,
        speed=args.hand_speed,
        force=args.hand_force,
        hold=args.hand_hold,
        read_before=False,
        read_after=False,
    )


def solve_right_q(
    target: np.ndarray,
    waist_q: np.ndarray,
    right_joints: Sequence[int],
    seed_motor_q: np.ndarray,
    tolerance: float,
) -> Tuple[object, np.ndarray, Dict[int, float]]:
    signs = np.ones(7)
    seed_model_q = motor_to_model_q(seed_motor_q, signs)
    ik = solve_position_ik(
        target=target,
        waist_q=waist_q,
        joint_indices=right_joints,
        signs=signs,
        seed=seed_model_q,
        tolerance=tolerance,
    )
    q_values = model_to_motor_q(ik.q_model, signs)
    return ik, q_values, {joint: float(value) for joint, value in zip(right_joints, q_values)}


def validate_wrist_target(joint: int, value: float) -> None:
    low, high = WRIST_LIMITS[joint]
    if not low <= value <= high:
        raise ValueError(f"{WRIST_NAMES[joint]} must be within [{low:.4f}, {high:.4f}] rad")


def apply_wrist_targets(
    q_values: np.ndarray,
    q_targets: Dict[int, float],
    right_joints: Sequence[int],
    wrist_roll_target: float,
    wrist_pitch_target: float,
    wrist_yaw_target: float,
) -> Tuple[np.ndarray, Dict[int, float]]:
    adjusted_values = np.asarray(q_values, dtype=float).copy()
    adjusted_targets = dict(q_targets)
    wrist_targets = {
        G1JointIndex.RightWristRoll: wrist_roll_target,
        G1JointIndex.RightWristPitch: wrist_pitch_target,
        G1JointIndex.RightWristYaw: wrist_yaw_target,
    }
    for joint, value in wrist_targets.items():
        if joint in right_joints:
            index = right_joints.index(joint)
            adjusted_values[index] = float(value)
            adjusted_targets[joint] = float(value)
    return adjusted_values, adjusted_targets


def segment_duration(default_duration: float, q_from: np.ndarray, q_to: np.ndarray, max_speed: float) -> float:
    delta = float(np.max(np.abs(q_to - q_from)))
    return max(default_duration, delta / max(max_speed, 1e-6))


def print_plan_item(name: str, target: np.ndarray, ik, q_values: np.ndarray, q_prev: np.ndarray, right_joints) -> None:
    print(f"{name}_target_pelvis_xyz:", np.array2string(target, precision=4))
    print(f"{name}_ik_success:", ik.success)
    print(f"{name}_ik_residual_m: {ik.residual:.4f}")
    print(f"{name}_ik_reached_xyz:", np.array2string(ik.reached, precision=4))
    print(f"{name}_right_goal:")
    print(format_joint_table(right_joints, q_values))
    print(f"{name}_right_delta:")
    print(format_joint_delta_table(right_joints, q_prev, q_values))


def print_command_position(name: str, target: np.ndarray, q_values: np.ndarray, waist_q: np.ndarray, right_joints) -> None:
    model_q = motor_to_model_q(q_values, np.ones(7))
    pose = right_hand_pose(model_q, waist_q, right_joints)
    reached = pose[:3, 3]
    residual = float(np.linalg.norm(target - reached))
    print(f"{name}_command_reached_xyz:", np.array2string(reached, precision=4))
    print(f"{name}_command_residual_m: {residual:.4f}")


def validate_hand_args(args: argparse.Namespace) -> int:
    if args.hand_speed < 0 or args.hand_speed > 1000 or args.hand_force < 0 or args.hand_force > 3000:
        print("Refusing hand command: speed must be in 0..1000 and force must be in 0..3000.", file=sys.stderr)
        return 2
    if any(value < -1 or value > 1000 for value in args.hand_open_angle + args.hand_grasp_angle):
        print("Refusing hand command: all hand angles must be -1 or in 0..1000.", file=sys.stderr)
        return 2
    return 0


def hand_tcp_preflight(args: argparse.Namespace) -> bool:
    try:
        with socket.create_connection((args.hand_ip, args.hand_port), timeout=args.hand_timeout):
            return True
    except OSError as exc:
        print(f"hand_preflight_connect: {args.hand_ip}:{args.hand_port} failed: {exc}", file=sys.stderr)
        print(HAND_NETWORK_HINT, file=sys.stderr)
        return False


def announce_success(args: argparse.Namespace) -> None:
    if args.no_announce or not args.announce_text:
        return
    try:
        audio_client = AudioClient()
        audio_client.SetTimeout(10.0)
        audio_client.Init()
        code = audio_client.TtsMaker(args.announce_text, args.announce_speaker_id)
        if code != 0:
            print(f"Warning: TTS announcement failed with code {code}.", file=sys.stderr)
        else:
            print(f"TTS announced: {args.announce_text}")
    except Exception as exc:
        print(f"Warning: TTS announcement failed: {exc}", file=sys.stderr)


def read_mjpg_rgb_frame(url: str) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required to read the MJPEG stream") from exc

    capture = cv2.VideoCapture(url)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open MJPEG stream: {url}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read one MJPEG frame: {url}")
        return frame
    finally:
        capture.release()


def read_http_rgbd_npz(url: str) -> Dict[str, object]:
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            payload = response.read()
    except Exception as exc:
        raise RuntimeError(f"failed to read HTTP RGB-D npz: {url}: {exc}") from exc

    try:
        data = np.load(io.BytesIO(payload))
        return {
            "source": "http_rgbd_npz",
            "color_bgr": np.asarray(data["color_bgr"]).copy(),
            "depth_raw": np.asarray(data["depth_raw"]).copy(),
            "depth_scale": float(data["depth_scale"]),
            "fx": float(data["fx"]),
            "fy": float(data["fy"]),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
            "rgbd_npz_url": url,
            "stamp": float(data["stamp"]) if "stamp" in data.files else None,
            "alignment_note": "RGB and depth are read from the same second-RealSense HTTP rgbd.npz buffer.",
        }
    except Exception as exc:
        raise RuntimeError(f"failed to decode HTTP RGB-D npz: {url}: {exc}") from exc


def scaled_mjpg_intrinsics(args: argparse.Namespace, image_shape: Sequence[int]) -> Tuple[float, float, float, float]:
    height, width = image_shape[:2]
    sx = float(width) / max(float(args.mjpg_reference_width), 1.0)
    sy = float(height) / max(float(args.mjpg_reference_height), 1.0)
    return (
        float(args.mjpg_fx) * sx,
        float(args.mjpg_fy) * sy,
        float(args.mjpg_cx) * sx,
        float(args.mjpg_cy) * sy,
    )


def read_realsense_depth_only(args: argparse.Namespace) -> Dict[str, object]:
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyrealsense2 is required for --mjpg-depth-source realsense") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if args.depth_serial:
        config.enable_device(args.depth_serial)
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.depth_fps)

    profile = pipeline.start(config)
    try:
        frames = None
        for _ in range(max(1, args.depth_warmup_frames)):
            frames = pipeline.wait_for_frames(5000)
        depth_frame = frames.get_depth_frame() if frames is not None else None
        if not depth_frame:
            raise RuntimeError("missing RealSense depth frame")
        intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        return {
            "source": "realsense_depth_only",
            "depth_raw": np.asanyarray(depth_frame.get_data()).copy(),
            "depth_scale": float(depth_sensor.get_depth_scale()),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "depth_serial": args.depth_serial,
        }
    finally:
        pipeline.stop()


def image_msg_to_depth_array(msg) -> Tuple[np.ndarray, float]:
    encoding = msg.encoding.lower()
    if encoding in ("16uc1", "mono16"):
        dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
        scale = 0.001
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
        scale = 1.0
    else:
        raise RuntimeError(f"unsupported ROS depth encoding: {msg.encoding}")

    array = np.frombuffer(msg.data, dtype=dtype)
    expected = int(msg.height) * int(msg.width)
    if array.size < expected:
        raise RuntimeError(f"ROS depth payload too small: got {array.size}, expected {expected}")
    array = array[:expected].reshape(int(msg.height), int(msg.width)).copy()
    return array, scale


def read_ros_aligned_depth(args: argparse.Namespace) -> Dict[str, object]:
    try:
        import rospy
        from sensor_msgs.msg import CameraInfo, Image
    except ModuleNotFoundError:
        ros_python = Path("/opt/ros/noetic/lib/python3/dist-packages")
        if ros_python.exists() and str(ros_python) not in sys.path:
            sys.path.insert(0, str(ros_python))
        try:
            import rospy
            from sensor_msgs.msg import CameraInfo, Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("rospy and sensor_msgs are required for --mjpg-depth-source ros") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("g1_mjpg_grasp_depth_reader", anonymous=True, disable_signals=True)

    depth_msg = rospy.wait_for_message(args.ros_depth_topic, Image, timeout=args.ros_timeout)
    info_msg = rospy.wait_for_message(args.ros_camera_info_topic, CameraInfo, timeout=args.ros_timeout)
    depth, scale = image_msg_to_depth_array(depth_msg)
    if len(info_msg.K) >= 6 and info_msg.K[0] > 0.0 and info_msg.K[4] > 0.0:
        fx = float(info_msg.K[0])
        fy = float(info_msg.K[4])
        cx = float(info_msg.K[2])
        cy = float(info_msg.K[5])
    else:
        raise RuntimeError("ROS CameraInfo has no usable K matrix")

    return {
        "source": "ros_aligned_depth",
        "depth_raw": depth,
        "depth_scale": scale,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "ros_depth_topic": args.ros_depth_topic,
        "ros_camera_info_topic": args.ros_camera_info_topic,
        "ros_depth_encoding": depth_msg.encoding,
        "alignment_note": "Depth is read from ROS aligned_depth_to_color topic and paired with MJPEG RGB.",
    }


def depth_point_for_mjpg_detection(
    depth_capture: Dict[str, object],
    det: Dict[str, object],
    rgb_shape: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    depth = depth_capture["depth_raw"]
    scale = float(depth_capture["depth_scale"])
    rgb_h, rgb_w = rgb_shape[:2]
    depth_h, depth_w = depth.shape[:2]
    sx = float(depth_w) / max(float(rgb_w), 1.0)
    sy = float(depth_h) / max(float(rgb_h), 1.0)

    x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
    cx_rgb, cy_rgb = [float(v) for v in det["center_xy"]]
    cx_depth = cx_rgb * sx
    cy_depth = cy_rgb * sy

    w = max(6, int((x2 - x1) * sx * 0.35))
    h = max(6, int((y2 - y1) * sy * 0.35))
    x0 = max(0, int(round(cx_depth - w * 0.5)))
    x3 = min(depth_w, int(round(cx_depth + w * 0.5)))
    y0 = max(0, int(round(cy_depth - h * 0.5)))
    y3 = min(depth_h, int(round(cy_depth + h * 0.5)))

    crop = depth[y0:y3, x0:x3].astype(np.float32) * scale
    valid = crop[np.isfinite(crop) & (crop > 0.05)]
    if valid.size == 0:
        raise RuntimeError("no valid depth samples in mapped detection crop")

    z = float(np.median(valid))
    x = (cx_depth - float(depth_capture["cx"])) * z / float(depth_capture["fx"])
    y = (cy_depth - float(depth_capture["cy"])) * z / float(depth_capture["fy"])
    return np.asarray([x, y, z], dtype=float), {
        "rgb_source": "mjpg",
        "depth_source": depth_capture.get("source", "depth_capture"),
        "rgb_size_wh": [int(rgb_w), int(rgb_h)],
        "depth_size_wh": [int(depth_w), int(depth_h)],
        "rgb_to_depth_scale_xy": [sx, sy],
        "rgb_center_xy": [cx_rgb, cy_rgb],
        "depth_center_xy": [cx_depth, cy_depth],
        "depth_crop_xyxy": [x0, y0, x3, y3],
        "valid_depth_samples": int(valid.size),
        "depth_median_m": z,
        "depth_intrinsics": {
            "fx": float(depth_capture["fx"]),
            "fy": float(depth_capture["fy"]),
            "cx": float(depth_capture["cx"]),
            "cy": float(depth_capture["cy"]),
        },
        "rgbd_npz_url": depth_capture.get("rgbd_npz_url"),
        "rgbd_stamp": depth_capture.get("stamp"),
        "ros_depth_topic": depth_capture.get("ros_depth_topic"),
        "ros_camera_info_topic": depth_capture.get("ros_camera_info_topic"),
        "ros_depth_encoding": depth_capture.get("ros_depth_encoding"),
        "alignment_note": depth_capture.get(
            "alignment_note",
            "RGB/depth pixels are scaled by image size; verify stream alignment if targets drift.",
        ),
    }


def fixed_depth_point_for_detection(args: argparse.Namespace, image_bgr: np.ndarray, det: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, object]]:
    if args.mjpg_assumed_depth <= 0.0:
        raise RuntimeError("--mjpg-assumed-depth must be positive")
    fx, fy, cx, cy = scaled_mjpg_intrinsics(args, image_bgr.shape)
    u, v = (float(det["center_xy"][0]), float(det["center_xy"][1]))
    z = float(args.mjpg_assumed_depth)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    height, width = image_bgr.shape[:2]
    return np.asarray([x, y, z], dtype=float), {
        "rgb_source": "mjpg",
        "rgb_url": args.mjpg_url,
        "rgb_size_wh": [int(width), int(height)],
        "reference_size_wh": [float(args.mjpg_reference_width), float(args.mjpg_reference_height)],
        "intrinsics_used": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "assumed_depth_m": z,
        "note": "RGB-only MJPEG fallback; point_camera uses --mjpg-assumed-depth.",
    }


def compute_vision_grasp_target(args: argparse.Namespace, pre_target: np.ndarray) -> np.ndarray:
    model, model_path = load_model(args.object)

    if args.vision_source == "mjpg":
        if args.mjpg_depth_source == "http-rgbd":
            capture_info = read_http_rgbd_npz(args.rgbd_npz_url)
            color_bgr = capture_info["color_bgr"]
        else:
            color_bgr = read_mjpg_rgb_frame(args.mjpg_url)
            capture_info = {
                "source": "mjpg",
                "color_bgr": color_bgr,
            }
        det = detect_object(model, color_bgr, args.object)
        if args.mjpg_depth_source == "http-rgbd":
            point_camera, depth_stats = depth_point_for_mjpg_detection(capture_info, det, color_bgr.shape)
        elif args.mjpg_depth_source == "ros":
            depth_capture = read_ros_aligned_depth(args)
            point_camera, depth_stats = depth_point_for_mjpg_detection(depth_capture, det, color_bgr.shape)
        elif args.mjpg_depth_source == "realsense":
            depth_capture = read_realsense_depth_only(args)
            point_camera, depth_stats = depth_point_for_mjpg_detection(depth_capture, det, color_bgr.shape)
        else:
            point_camera, depth_stats = fixed_depth_point_for_detection(args, color_bgr, det)
    else:
        capture_args = Namespace(
            host=args.vision_host,
            password=args.vision_password,
            interface=args.network_interface,
        )
        capture_info = run_capture(capture_args)
        det = detect_object(model, capture_info["color_bgr"], args.object)
        point_camera, depth_stats = depth_point_for_detection(capture_info, det)

    point_pelvis = camera_to_pelvis(point_camera, args.camera_pitch)

    target = pre_target.copy()
    target[0] = point_pelvis[0] + args.vision_xy_offset[0]
    target[1] = point_pelvis[1] + args.vision_xy_offset[1]
    target[2] = args.fixed_grasp_z

    print("vision_model:", model_path)
    print("vision_source:", args.vision_source)
    print("vision_detection:", det)
    print("vision_depth_stats:", depth_stats)
    print("vision_point_camera_m:", np.array2string(point_camera, precision=4))
    print("vision_point_pelvis_m:", np.array2string(point_pelvis, precision=4))
    print("vision_xy_offset_m:", np.array2string(np.asarray(args.vision_xy_offset, dtype=float), precision=4))
    print("vision_fixed_grasp_z:", f"{args.fixed_grasp_z:.4f}")
    return target


def main() -> int:
    args = parse_args()
    if args.execute and not args.yes:
        print("Refusing to move without --yes.", file=sys.stderr)
        return 1
    if args.weight <= 0.0 or args.weight > 1.0:
        print("--weight must be in (0, 1].", file=sys.stderr)
        return 2
    if args.weight > 1.0 and not args.allow_full_weight:
        print("Refusing --weight above 1.0 without --allow-full-weight.", file=sys.stderr)
        return 2
    if args.left_distance < 0.0 and not args.use_vision:
        print("--left-distance must be non-negative when --use-vision is not set.", file=sys.stderr)
        return 2
    if not args.no_vertical_wrist:
        try:
            validate_wrist_target(G1JointIndex.RightWristRoll, args.wrist_roll_target)
            validate_wrist_target(G1JointIndex.RightWristPitch, args.wrist_pitch_target)
            validate_wrist_target(G1JointIndex.RightWristYaw, args.wrist_yaw_target)
        except ValueError as exc:
            print(f"Refusing wrist target: {exc}", file=sys.stderr)
            return 2
    if min(args.pre_duration, args.lateral_duration, args.return_duration, args.release_duration) < 0.0:
        print("Durations must be non-negative.", file=sys.stderr)
        return 2
    hand_error = validate_hand_args(args)
    if hand_error:
        return hand_error
    if args.execute and not args.skip_hand and not hand_tcp_preflight(args):
        return 6

    pre_target = np.asarray(args.pregrasp_target, dtype=float)
    if args.use_vision:
        grasp_target = compute_vision_grasp_target(args, pre_target)
        args.left_distance = float(grasp_target[1] - pre_target[1])
    else:
        grasp_target = pre_target.copy()
        grasp_target[1] += args.left_distance
    for label, target in (("pregrasp", pre_target), ("grasp", grasp_target)):
        try:
            validate_target(target)
        except ValueError as exc:
            print(f"Refusing {label} target: {exc}", file=sys.stderr)
            return 2

    print("saved_pregrasp_target:", np.array2string(pre_target, precision=4))
    print("final_grasp_target:", np.array2string(grasp_target, precision=4))
    print(f"computed_x_delta: {float(grasp_target[0] - pre_target[0]):.4f}")
    print(f"computed_left_distance: {args.left_distance:.4f}")

    network_interface = detect_interface(args.network_interface)
    print(f"network_interface: {network_interface}")
    controller = OfficialStyleArmSdkController(network_interface, control_dt=0.02)
    controller.init()
    state = controller.wait_for_low_state(5.0)

    right_joints = right_arm_joints(7)
    arm_dq_max = joint_velocity_abs_max(state, LEFT_ARM_7DOF + right_joints)
    waist_dq_max = joint_velocity_abs_max(state, WAIST_JOINTS)
    print(f"start_arm_dq_abs_max: {arm_dq_max:.4f}")
    print(f"start_waist_dq_abs_max: {waist_dq_max:.4f}")
    if arm_dq_max > args.max_start_arm_dq:
        print("Refusing: arms are already moving too fast.", file=sys.stderr)
        return 3
    if waist_dq_max > args.max_start_waist_dq:
        print("Refusing: waist is already moving too fast.", file=sys.stderr)
        return 3

    q_left_hold = joint_positions(state, LEFT_ARM_7DOF)
    q_waist_hold = joint_positions(state, WAIST_JOINTS)
    q_right_start = joint_positions(state, right_joints)
    q_start_values = joint_vector(state, right_joints)
    waist_q = joint_vector(state, WAIST_JOINTS)

    pre_ik, q_pre_values, q_pre = solve_right_q(pre_target, waist_q, right_joints, q_start_values, args.tolerance)
    if not args.no_vertical_wrist:
        q_pre_values, q_pre = apply_wrist_targets(
            q_pre_values,
            q_pre,
            right_joints,
            args.wrist_roll_target,
            args.wrist_pitch_target,
            args.wrist_yaw_target,
        )
    grasp_ik, q_grasp_values, q_grasp = solve_right_q(grasp_target, waist_q, right_joints, q_pre_values, args.tolerance)
    if not args.no_vertical_wrist:
        q_grasp_values, q_grasp = apply_wrist_targets(
            q_grasp_values,
            q_grasp,
            right_joints,
            args.wrist_roll_target,
            args.wrist_pitch_target,
            args.wrist_yaw_target,
        )
        print("wrist_override_enabled:")
        print(f"  RightWristRoll -> {args.wrist_roll_target:.4f} rad")
        print(f"  RightWristPitch -> {args.wrist_pitch_target:.4f} rad")
        print(f"  RightWristYaw -> {args.wrist_yaw_target:.4f} rad")
    else:
        print("wrist_override_enabled: false")

    print_plan_item("pregrasp", pre_target, pre_ik, q_pre_values, q_start_values, right_joints)
    print_command_position("pregrasp", pre_target, q_pre_values, waist_q, right_joints)
    print_plan_item("grasp", grasp_target, grasp_ik, q_grasp_values, q_pre_values, right_joints)
    print_command_position("grasp", grasp_target, q_grasp_values, waist_q, right_joints)
    print("left_hold:")
    print(format_joint_table(LEFT_ARM_7DOF, [q_left_hold[joint] for joint in LEFT_ARM_7DOF]))
    print("waist_hold:")
    print(format_joint_table(WAIST_JOINTS, [q_waist_hold[joint] for joint in WAIST_JOINTS]))

    if not pre_ik.success or not grasp_ik.success:
        print("Refusing execution because IK did not meet tolerance.", file=sys.stderr)
        return 4

    max_delta_pre = float(np.max(np.abs(q_pre_values - q_start_values)))
    max_delta_lateral = float(np.max(np.abs(q_grasp_values - q_pre_values)))
    max_delta = max(max_delta_pre, max_delta_lateral)
    print(f"max_pre_delta_rad: {max_delta_pre:.4f}")
    print(f"max_lateral_delta_rad: {max_delta_lateral:.4f}")
    if max_delta > args.max_joint_delta:
        print(f"Refusing execution: max joint delta {max_delta:.3f} exceeds {args.max_joint_delta:.3f}.", file=sys.stderr)
        return 5

    pre_duration = segment_duration(args.pre_duration, q_start_values, q_pre_values, args.max_joint_speed)
    lateral_duration = segment_duration(args.lateral_duration, q_pre_values, q_grasp_values, args.max_joint_speed)
    return_duration = segment_duration(args.return_duration, q_pre_values, q_start_values, args.max_joint_speed)
    print(f"pre_duration_s: {pre_duration:.2f}")
    print(f"lateral_duration_s: {lateral_duration:.2f}")
    print(f"return_duration_s: {return_duration:.2f}")

    if not args.execute:
        print("Dry run only. Add --execute --yes to move the robot.")
        return 0

    q_right_last = dict(q_right_start)
    exit_code = 0
    try:
        print("Stage 1: move to saved pregrasp target.")
        run_timed_phase(controller, pre_duration, q_left_hold, q_waist_hold, q_right_start, q_pre, 0.0, args.weight, args)
        q_right_last = dict(q_pre)
        hold_group(controller, args.hold_before_open, q_left_hold, q_pre, q_waist_hold, args.weight, args)

        if not args.skip_hand:
            print("Stage 2: open hand to maximum preset.")
            write_hand_pose(hand_args(args), args.hand_open_angle, "max_open")
            hold_group(controller, args.hold_after_open, q_left_hold, q_pre, q_waist_hold, args.weight, args)

        print("Stage 3: move whole hand to final grasp target.")
        run_timed_phase(controller, lateral_duration, q_left_hold, q_waist_hold, q_pre, q_grasp, args.weight, args.weight, args)
        q_right_last = dict(q_grasp)
        hold_group(controller, args.hold_before_grasp, q_left_hold, q_grasp, q_waist_hold, args.weight, args)

        if not args.skip_hand:
            print("Stage 4: grasp.")
            write_hand_pose(hand_args(args), args.hand_grasp_angle, "grasp")
            hold_group(controller, args.hold_after_grasp, q_left_hold, q_grasp, q_waist_hold, args.weight, args)

        print("Stage 5: move back to saved pregrasp target.")
        run_timed_phase(controller, lateral_duration, q_left_hold, q_waist_hold, q_grasp, q_pre, args.weight, args.weight, args)
        q_right_last = dict(q_pre)

        print("Stage 6: retract arm to start posture.")
        run_timed_phase(controller, return_duration, q_left_hold, q_waist_hold, q_pre, q_right_start, args.weight, args.weight, args)
        q_right_last = dict(q_right_start)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        exit_code = 130
    finally:
        print("Stage 7: release arm_sdk.")
        release_group(controller, q_left_hold, q_right_last, q_waist_hold, args.weight, args)
        print("arm_sdk released.")

    if exit_code == 0:
        announce_success(args)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
