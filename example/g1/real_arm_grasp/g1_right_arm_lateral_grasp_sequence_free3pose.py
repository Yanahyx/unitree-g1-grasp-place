#!/usr/bin/env python3
"""Use Free3Pose RGB object poses with the existing force-feedback side grasp.

Run on the robot (local D435i acquisition, no motion by default):
  python3 example/g1/real_arm_grasp/g1_right_arm_lateral_grasp_sequence_free3pose.py eth1 \
      --vision-xy-offset -0.06 0 --side-standoff-y 0 --no-announce

Add --execute --yes and the usual hand options to execute. --pose-only performs
capture, estimation and target conversion without creating a DDS participant.
--legacy-help lists the unchanged sequence's arm/hand options.

If the robot cannot reach the server but the computer can, add
--pose-run-on-robot when running this file ON THE COMPUTER. One SSH session
runs this same entry on the robot and forwards its HTTP requests through the
computer. The camera/arm/hand code still runs on the robot; the tunnel closes
with the command. The script and credentials must already be on the robot.

The original *_mjpg_force_thumb.py is loaded into an isolated module namespace;
its source is never edited. Only argument parsing, the vision target provider,
and the final pre-motion freshness check are adapted. Its IK, wrist posture,
fixed grasp Z, force feedback, thumb staging and return sequence are reused.

T_camera_object maps object-model coordinates to the color optical frame
(right/down/forward, meters). The optional --pose-grasp-offset-object is rotated
by the estimated R before translation. By default the model center is used;
the hand orientation is the original side-grasp posture, NOT the object's R.
The approximate camera mounting transform is also the old script's transform.
An optional calibrated --pose-extrinsics-json can supply T_pelvis_camera.

Credentials: FREE3POSE_API_KEY, --pose-token-file, or --pose-client-bundle.
The ZIP is read in memory; its token is never extracted or logged. A bundle at
~/.config/free3pose/robot_client_bundle.zip is discovered automatically.
Each capture creates frame.png, camera.json, capture.json, pose_result.json and
plan.json under <repo>/tmp/free3pose_runs. No old pose result is used as fallback.
Keep the robot, camera and object stationary during server inference.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
LEGACY_PATH = THIS_DIR / "g1_right_arm_lateral_grasp_sequence_mjpg_force_thumb.py"
SERVER_URL = "http://219.223.205.204:8766"
CAMERA_SERIAL = "243322071448"
CAMERA_FRAME = "d435i_color_optical_frame"
MARKER = "__FREE3POSE_CAPTURE__"

# Run in a bounded child process, locally or over SSH, so a blocked driver does
# not leave the grasp process waiting forever. No persistent camera service.
CAPTURE_WORKER = r'''
import base64, json, sys, time
import cv2
import numpy as np
import pyrealsense2 as rs

options = json.load(sys.stdin)
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(options['serial'])
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
active = pipeline.start(config)
try:
    frame = None
    for _ in range(options['warmup_frames']):
        candidate = pipeline.wait_for_frames(3000).get_color_frame()
        if candidate:
            frame = candidate
    if frame is None:
        raise RuntimeError('Camera did not return a color frame')
    received_ns = time.time_ns()
    intr = frame.profile.as_video_stream_profile().get_intrinsics()
    image = np.asanyarray(frame.get_data())
    if image.shape != (720, 1280, 3):
        raise RuntimeError('Unexpected native color shape: ' + str(image.shape))
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    domain = str(frame.get_frame_timestamp_domain()).split('.')[-1]
    frame_timestamp_ns = None
    if domain in ('global_time', 'system_time'):
        frame_timestamp_ns = int(round(frame.get_timestamp() * 1000000))
    dev = active.get_device()
    camera = {
        'model': dev.get_info(rs.camera_info.name),
        'serial_number': dev.get_info(rs.camera_info.serial_number),
        'source': 'Active color-frame SDK intrinsics',
        'stream_format': 'BGR8',
        'size_wh': [int(image.shape[1]), int(image.shape[0])],
        'K': [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
        'distortion_model': str(intr.model).split('.')[-1],
        'distortion_coefficients': list(intr.coeffs),
        'frame_id': options['frame_id'],
    }
    payload = {
        'camera': camera,
        'image_base64': base64.b64encode(encoded.tobytes()).decode('ascii'),
        'capture_timestamp_ns': frame_timestamp_ns,
        'frame_timestamp_ms': frame.get_timestamp(),
        'frame_timestamp_domain': domain,
        'frame_number': frame.get_frame_number(),
        'frame_received_wall_timestamp_ns': received_ns,
    }
    print('__FREE3POSE_CAPTURE__' + json.dumps(payload), flush=True)
finally:
    pipeline.stop()
'''


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--pose-server-url", default=SERVER_URL,
                        help="Server base URL; /v1/estimate is appended.")
    parser.add_argument("--pose-token-file", type=Path)
    parser.add_argument("--pose-client-bundle", type=Path,
                        help="Handoff ZIP containing free3pose_robot_client/api_token.txt.")
    parser.add_argument("--pose-timeout", type=float, default=330.0)
    parser.add_argument("--pose-max-age", type=float, default=180.0,
                        help="Maximum seconds since acquisition started before motion.")
    parser.add_argument("--pose-capture-source", choices=("local", "ssh"), default="local")
    parser.add_argument("--pose-camera-serial", default=CAMERA_SERIAL)
    parser.add_argument("--pose-warmup-frames", type=int, default=30)
    parser.add_argument("--pose-capture-timeout", type=float, default=35.0)
    parser.add_argument("--pose-ssh-host", default="unitree@192.168.123.164")
    parser.add_argument("--pose-ssh-password", default=os.environ.get("ROBOT_SSH_PASSWORD"),
                        help="SSH capture only; defaults to the legacy --vision-password.")
    parser.add_argument("--pose-ssh-known-hosts", type=Path)
    parser.add_argument("--pose-run-on-robot", action="store_true",
                        help="Computer launcher: SSH to robot and relay server traffic over the cable.")
    parser.add_argument("--pose-remote-repo", default="/home/unitree/unitree_sdk2_python")
    parser.add_argument("--pose-forward-port", type=int, default=18766,
                        help="Robot loopback port for the temporary SSH HTTP relay.")
    parser.add_argument("--pose-output-dir", type=Path, default=ROOT / "tmp/free3pose_runs")
    parser.add_argument("--pose-only", action="store_true",
                        help="Capture and estimate only; no DDS, arm or hand commands.")
    parser.add_argument("--pose-extrinsics-json", type=Path,
                        help='Optional {"T_pelvis_camera": 4x4 matrix}, meters.')
    parser.add_argument("--pose-grasp-offset-object", nargs=3, type=float, default=[0, 0, 0],
                        metavar=("X", "Y", "Z"), help="Grasp point in object coordinates, meters.")
    parser.add_argument("--pose-use-estimated-z", action="store_true",
                        help="Use pose-derived Z instead of the legacy fixed grasp height.")
    parser.add_argument("--pose-z-offset", type=float, default=0.0,
                        help="Pelvis Z correction, only with --pose-use-estimated-z.")
    parser.add_argument("--pose-max-body-angle", type=float, default=0.08,
                        help="Maximum IMU orientation change while awaiting inference, radians.")
    parser.add_argument("--pose-max-body-joint-delta", type=float, default=0.05,
                        help="Maximum leg/waist joint change before motion, radians.")
    parser.add_argument("--legacy-help", action="store_true")
    return parser


def load_legacy():
    if not LEGACY_PATH.is_file():
        raise ValueError("Original sequence must remain beside this new entry point")
    # A private module prevents these three adapter hooks from affecting other
    # imports of the original grasp program in the same interpreter.
    spec = importlib.util.spec_from_file_location("_free3pose_legacy_sequence", str(LEGACY_PATH))
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    return legacy


def parse_arguments(argv=None):
    parser = make_parser()
    pose, remaining = parser.parse_known_args(argv)
    legacy = load_legacy()
    old_argv = sys.argv
    try:
        sys.argv = [str(LEGACY_PATH)] + (["--help"] if pose.legacy_help else remaining)
        args = legacy.parse_args()
    finally:
        sys.argv = old_argv
    for name, value in vars(pose).items():
        setattr(args, name, value)
    args.use_vision = True
    validate_options(args)
    return args, legacy


def validate_options(args):
    for name in ("pose_timeout", "pose_max_age", "pose_capture_timeout",
                 "pose_max_body_angle", "pose_max_body_joint_delta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("--" + name.replace("_", "-") + " must be positive and finite")
    if args.pose_warmup_frames < 1:
        raise ValueError("--pose-warmup-frames must be positive")
    if args.execute and (not args.yes or args.pose_only):
        raise ValueError("Motion requires --execute --yes and cannot use --pose-only")
    if args.detector_server_url:
        raise ValueError("Replace --detector-server-url with --pose-server-url for this entry")
    if args.object not in ("bottle", "starbucks_bottle"):
        raise ValueError("This deployed pose model only supports the Starbucks bottle")
    if not np.all(np.isfinite(args.pose_grasp_offset_object)) or not math.isfinite(args.pose_z_offset):
        raise ValueError("Pose grasp offsets must be finite")
    if args.pose_z_offset and not args.pose_use_estimated_z:
        raise ValueError("--pose-z-offset requires --pose-use-estimated-z")
    if args.pose_camera_serial != CAMERA_SERIAL:
        raise ValueError("This handoff is configured for camera serial " + CAMERA_SERIAL)
    if not 1 <= args.pose_forward_port <= 65535:
        raise ValueError("--pose-forward-port must be in 1..65535")
    parsed = urllib.parse.urlsplit(args.pose_server_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
        raise ValueError("--pose-server-url must be an HTTP(S) base URL without credentials")
    if parsed.path.rstrip("/") or parsed.query or parsed.fragment:
        raise ValueError("Use the server base URL, without /v1/estimate or query parameters")
    if args.pose_run_on_robot and (parsed.scheme != "http" or args.pose_capture_source != "local"):
        raise ValueError("The computer launcher requires an HTTP server and local-on-robot camera capture")


def ssh_command(args):
    if args.pose_ssh_host.startswith("-"):
        raise ValueError("Invalid SSH robot host")
    command = ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new"]
    if args.pose_ssh_known_hosts:
        command += ["-o", "UserKnownHostsFile=" + str(args.pose_ssh_known_hosts.expanduser())]
    password = args.pose_ssh_password if args.pose_ssh_password is not None else args.vision_password
    read_fd = None
    if password:
        if not shutil.which("sshpass"):
            raise ValueError("SSH password operation requires sshpass, or an empty password for SSH keys")
        read_fd, write_fd = os.pipe()
        os.write(write_fd, (password + "\n").encode())
        os.close(write_fd)
        command = ["sshpass", "-d", str(read_fd)] + command
    return command, read_fd


def robot_arguments(argv, port):
    # Preserve exact original/pose flags, including negative numeric values.
    # These local launcher options must not recurse or expose an SSH password
    # in the remote process command line.
    local_flags = {"--pose-run-on-robot": 0, "--pose-remote-repo": 1,
                   "--pose-forward-port": 1, "--pose-ssh-host": 1,
                   "--pose-ssh-password": 1, "--pose-ssh-known-hosts": 1,
                   "--pose-server-url": 1}
    forwarded = []
    index = 0
    while index < len(argv):
        token = argv[index]
        name = token.split("=", 1)[0]
        if name in local_flags:
            index += 1 + (local_flags[name] if "=" not in token else 0)
        else:
            forwarded.append(token)
            index += 1
    return forwarded + ["--pose-server-url", "http://127.0.0.1:" + str(port)]


def run_on_robot(args, argv):
    server = urllib.parse.urlsplit(args.pose_server_url)
    host = server.hostname
    if ":" in host:
        host = "[" + host + "]"
    forward = "127.0.0.1:%d:%s:%d" % (args.pose_forward_port, host, server.port or 80)
    forwarded = robot_arguments(argv, args.pose_forward_port)
    remote = "cd " + shlex.quote(args.pose_remote_repo) + " && exec python3 -u " + shlex.quote(
        "example/g1/real_arm_grasp/" + Path(__file__).name)
    remote += " " + " ".join(shlex.quote(item) for item in forwarded)
    command, read_fd = ssh_command(args)
    command += ["-tt", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=3", "-R", forward, args.pose_ssh_host, remote]
    print("robot_launch:", args.pose_ssh_host, "server_via_computer:", args.pose_server_url, flush=True)
    try:
        child = subprocess.Popen(command, pass_fds=() if read_fd is None else (read_fd,))
        try:
            return child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    finally:
        if read_fd is not None:
            os.close(read_fd)


def read_api_key(args):
    token = os.environ.get("FREE3POSE_API_KEY", "").strip()
    if not token and args.pose_token_file:
        token = args.pose_token_file.expanduser().read_text().strip()
    elif not token:
        bundle = args.pose_client_bundle or Path.home() / ".config/free3pose/robot_client_bundle.zip"
        if not bundle.is_file():
            raise ValueError("Supply --pose-client-bundle ZIP, --pose-token-file or FREE3POSE_API_KEY")
        with zipfile.ZipFile(bundle) as archive:
            info = archive.getinfo("free3pose_robot_client/api_token.txt")
            if info.file_size > 4096:
                raise ValueError("Unexpected API token file size")
            token = archive.read(info).decode("utf-8").strip()
    if not token or "\n" in token or "\r" in token:
        raise ValueError("Missing or invalid pose API token")
    return token


def validate_camera(camera):
    if camera.get("serial_number") != CAMERA_SERIAL:
        raise ValueError("The supplied server model is configured for D435i " + CAMERA_SERIAL)
    if camera.get("size_wh") != [1280, 720] or camera.get("frame_id") != CAMERA_FRAME:
        raise ValueError("Expected native 1280x720 d435i_color_optical_frame")
    matrix = np.asarray(camera.get("K"), dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid active camera intrinsics")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("Invalid pinhole intrinsic matrix")
    coeffs = np.asarray(camera.get("distortion_coefficients"), dtype=float)
    if coeffs.shape != (5,) or not np.all(np.isfinite(coeffs)):
        raise ValueError("Expected five finite distortion coefficients")
    if np.any(coeffs != 0):
        raise ValueError("This integration expects the verified zero-distortion D435i color mode")


def capture_frame(args):
    options = {"serial": args.pose_camera_serial, "frame_id": CAMERA_FRAME,
               "warmup_frames": args.pose_warmup_frames}
    read_fd = None
    command = [sys.executable, "-u", "-c", CAPTURE_WORKER]
    if args.pose_capture_source == "ssh":
        command, read_fd = ssh_command(args)
        command += [args.pose_ssh_host, "python3 -u -c " + shlex.quote(CAPTURE_WORKER)]
    try:
        result = subprocess.run(
            command, input=json.dumps(options), text=True, capture_output=True,
            timeout=args.pose_capture_timeout, pass_fds=() if read_fd is None else (read_fd,),
        )
    finally:
        if read_fd is not None:
            os.close(read_fd)
    if result.returncode:
        raise RuntimeError("Camera capture failed: " + result.stderr[-1800:])
    lines = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise RuntimeError("Camera capture did not return one valid frame payload")
    payload = json.loads(lines[0])
    image_bytes = base64.b64decode(payload.pop("image_base64"), validate=True)
    validate_camera(payload["camera"])
    import cv2
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (720, 1280, 3):
        raise ValueError("Encoded image does not match active camera dimensions")
    return image_bytes, payload


def rigid_transform(value, label):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(label + " must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError(label + " has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0):
        raise ValueError(label + " rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-3):
        raise ValueError(label + " rotation must have determinant +1")
    return matrix


def validate_pose_result(result, image_bytes, camera, capture_id, timestamp_ns):
    if not isinstance(result, dict):
        raise ValueError("Pose server response is not an object")
    expected_hash = hashlib.sha256(image_bytes).hexdigest()
    if result.get("capture_id") != capture_id or result.get("image_sha256") != expected_hash:
        raise ValueError("Pose response does not match the submitted capture/image")
    if result.get("target_frame") != camera["frame_id"]:
        raise ValueError("Pose response camera frame mismatch")
    if "capture_timestamp_ns" in result and result["capture_timestamp_ns"] != timestamp_ns:
        raise ValueError("Pose response capture timestamp mismatch")
    if result.get("success") is not True:
        raise ValueError("Pose server returned no valid pose; inspect pose_result.json")
    transform = rigid_transform(result.get("T_camera_from_object_m"), "T_camera_from_object_m")
    rotation = np.asarray(result.get("R_camera_from_object"), dtype=float)
    translation = np.asarray(result.get("t_camera_from_object_m"), dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Pose R/t shapes must be 3x3 and 3")
    if not np.allclose(transform[:3, :3], rotation, atol=1e-5, rtol=0):
        raise ValueError("Pose R and T disagree")
    if not np.allclose(transform[:3, 3], translation, atol=1e-5, rtol=0):
        raise ValueError("Pose translation and T disagree")
    if not np.all(np.isfinite(translation)) or not 0.05 < translation[2] < 3.0:
        raise ValueError("Pose depth is invalid or outside the near-robot operating range")
    return transform


def request_pose(args, image_bytes, camera, capture_id, timestamp_ns, token):
    payload = {
        "object": "starbucks_bottle", "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "camera": camera, "capture_id": capture_id, "capture_timestamp_ns": timestamp_ns,
        "return_visualization": True,
    }
    request = urllib.request.Request(
        args.pose_server_url.rstrip("/") + "/v1/estimate",
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": token},
    )
    # Do not leak LAN images/credentials to an ambient HTTP proxy, or follow
    # redirects to another server with the API key header.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, new_url):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=args.pose_timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace").replace(token, "<redacted>")
        if exc.code == 503:
            raise RuntimeError("Pose server busy (HTTP 503); rerun to capture a NEW frame. " + detail) from exc
        raise RuntimeError("Pose server HTTP " + str(exc.code) + ": " + detail) from exc


def camera_transform(args):
    if args.pose_extrinsics_json:
        value = json.loads(args.pose_extrinsics_json.expanduser().read_text())
        return rigid_transform(value["T_pelvis_camera"], "T_pelvis_camera")
    # Exactly the legacy camera_to_pelvis convention; no new guessed extrinsics.
    c, s = math.cos(args.camera_pitch), math.sin(args.camera_pitch)
    rotation_y = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    optical_to_level = np.asarray([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    transform = np.eye(4)
    transform[:3, :3] = rotation_y @ optical_to_level
    transform[:3, 3] = [0.05366, 0.01753, 0.47387]
    return transform


def grasp_target(args, pre_target, camera_object):
    pelvis_camera = camera_transform(args)
    pelvis_object = pelvis_camera @ camera_object
    offset = np.asarray(args.pose_grasp_offset_object, dtype=float)
    point_camera = camera_object[:3, 3] + camera_object[:3, :3] @ offset
    point_pelvis = pelvis_object[:3, 3] + pelvis_object[:3, :3] @ offset
    target = np.asarray(pre_target, dtype=float).copy()
    target[0] = point_pelvis[0] + args.vision_xy_offset[0]
    object_y = point_pelvis[1] + args.vision_xy_offset[1]
    side = -1.0 if pre_target[1] <= object_y else 1.0
    target[1] = object_y + side * args.side_standoff_y
    target[2] = point_pelvis[2] + args.pose_z_offset if args.pose_use_estimated_z else args.fixed_grasp_z
    if not np.all(np.isfinite(target)):
        raise ValueError("Computed grasp target is non-finite")
    return target, {
        "T_pelvis_camera": pelvis_camera.tolist(), "T_pelvis_object": pelvis_object.tolist(),
        "camera_extrinsics_source": "calibrated_file" if args.pose_extrinsics_json else "legacy_approximate_mount",
        "grasp_offset_object_m": offset.tolist(), "grasp_point_camera_m": point_camera.tolist(),
        "grasp_point_pelvis_m": point_pelvis.tolist(), "target_pelvis_m": target.tolist(),
        "vision_xy_offset_m": list(args.vision_xy_offset), "side_standoff_y_m": args.side_standoff_y,
        "approach_side": side, "z_mode": "estimated" if args.pose_use_estimated_z else "legacy_fixed",
        "hand_orientation_mode": "legacy_side_grasp_wrist",
    }


def quaternion_distance(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.all(np.isfinite([a, b])):
        raise ValueError("Invalid body IMU quaternion")
    a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
    if a_norm < 1e-8 or b_norm < 1e-8:
        raise ValueError("Zero body IMU quaternion")
    dot = abs(float(np.dot(a / a_norm, b / b_norm)))
    return 2 * math.acos(min(1.0, dot))


class BodyMonitor:
    """Read-only DDS: reject changed body/waist or stale state before motion.

    IMU and leg/waist joints cannot detect every external displacement or an
    object being moved. Stationary scene/body is still a prerequisite.
    """
    def __init__(self, args):
        self.args = args
        self.subscriber = None
        self.lock = threading.Lock()
        self.latest = None
        self.reference = None
        self.max_angle = 0.0
        self.max_delta = 0.0
        self.error = None

    def start(self, network_interface):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(0, network_interface)
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self.on_state, 10)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self.lock:
                if self.latest is not None:
                    return
            time.sleep(0.02)
        raise RuntimeError("No robot state available before camera capture")

    def on_state(self, msg):
        with self.lock:
            try:
                current = {
                    "quaternion": list(msg.imu_state.quaternion),
                    "body_q": [float(msg.motor_state[i].q) for i in range(15)],
                    "received_monotonic": time.monotonic(),
                }
                quaternion_distance(current["quaternion"], current["quaternion"])
                if not np.all(np.isfinite(current["body_q"])):
                    raise ValueError("Non-finite robot body joints")
                if self.reference is not None:
                    self.max_angle = max(self.max_angle, quaternion_distance(
                        self.reference["quaternion"], current["quaternion"]))
                    self.max_delta = max(self.max_delta, float(np.max(np.abs(
                        np.asarray(self.reference["body_q"]) - current["body_q"]))))
                self.latest = current
            except Exception as exc:
                self.error = str(exc)

    def arm(self):
        with self.lock:
            if self.latest is None or time.monotonic() - self.latest["received_monotonic"] > 0.5:
                raise RuntimeError("Robot state is stale before capture")
            self.reference = dict(self.latest)

    def check(self):
        with self.lock:
            if self.error:
                raise RuntimeError("Robot-state monitor failed: " + self.error)
            if self.latest is None or time.monotonic() - self.latest["received_monotonic"] > 0.5:
                raise RuntimeError("Robot state became stale while waiting for the pose server")
            if self.max_angle > self.args.pose_max_body_angle:
                raise RuntimeError("Robot body orientation changed after capture; recapture before moving")
            if self.max_delta > self.args.pose_max_body_joint_delta:
                raise RuntimeError("Leg/waist joints changed after capture; recapture before moving")
            return {"max_body_angle_rad": self.max_angle, "max_body_joint_delta_rad": self.max_delta}

    def close(self):
        if self.subscriber is not None:
            self.subscriber.Close()
            self.subscriber = None


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def stop_signal(signum, frame):
    # An SSH terminal hangup must unwind the original sequence's finally block
    # (arm_sdk release), rather than terminate Python without cleanup. Its
    # progress prints must also survive the disconnected terminal.
    if signum == signal.SIGHUP:
        descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(descriptor, 1)
            os.dup2(descriptor, 2)
        finally:
            os.close(descriptor)
    raise KeyboardInterrupt


class PoseVision:
    def __init__(self, args, legacy):
        self.args, self.legacy = args, legacy
        self.monitor: Optional[BodyMonitor] = None
        self.capture_started = None
        self.run_dir = None
        self.plan = None

    def check_fresh(self):
        if self.capture_started is None:
            raise RuntimeError("No live pose capture has been made")
        age = time.monotonic() - self.capture_started
        if age > self.args.pose_max_age:
            raise RuntimeError("Pose is %.1fs old (limit %.1fs); rerun to capture a new frame" % (
                age, self.args.pose_max_age))
        status = {"capture_age_s": age}
        if self.monitor:
            status.update(self.monitor.check())
        return status

    def __call__(self, args, pre_target):
        token = read_api_key(args)
        # Validate a supplied extrinsic file before opening the camera/server.
        camera_transform(args)
        if not args.pose_only:
            self.monitor = BodyMonitor(args)
            self.monitor.start(self.legacy.detect_interface(args.network_interface))
            self.monitor.arm()
        capture_id = uuid.uuid4().hex
        self.run_dir = args.pose_output_dir.expanduser() / (
            datetime.now().strftime("%Y%m%d_%H%M%S_") + capture_id[:8])
        self.run_dir.mkdir(parents=True, exist_ok=False)
        print("pose_artifact_directory:", self.run_dir, flush=True)
        self.capture_started = time.monotonic()  # Conservative, includes camera warm-up/SSH.
        image_bytes, capture = capture_frame(args)
        camera = capture["camera"]
        (self.run_dir / "frame.png").write_bytes(image_bytes)
        save_json(self.run_dir / "camera.json", camera)
        capture.update({"capture_id": capture_id, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "capture_source": args.pose_capture_source})
        save_json(self.run_dir / "capture.json", capture)
        print("pose_capture:", capture_id, camera["size_wh"], camera["serial_number"], flush=True)
        print("pose_server_request:", args.pose_server_url, "(one request, no automatic retries)", flush=True)
        started = time.monotonic()
        result = request_pose(args, image_bytes, camera, capture_id, capture["capture_timestamp_ns"], token)
        if not isinstance(result, dict):
            raise ValueError("Pose server returned a non-object response")
        overlay = result.pop("visualization_jpeg_base64", None)
        save_json(self.run_dir / "pose_result.json", result)
        camera_object = validate_pose_result(
            result, image_bytes, camera, capture_id, capture["capture_timestamp_ns"])
        if overlay:
            (self.run_dir / "pose_overlay.jpg").write_bytes(base64.b64decode(overlay, validate=True))
        freshness = self.check_fresh()
        target, self.plan = grasp_target(args, pre_target, camera_object)
        self.legacy.validate_target(target)
        self.plan.update({"capture_id": capture_id, "server_url": args.pose_server_url,
                          "server_roundtrip_s": time.monotonic() - started, "freshness": freshness,
                          "robot_posture_monitored": self.monitor is not None,
                          "T_camera_object": camera_object.tolist(),
                          "original_sequence_sha256": hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest()})
        save_json(self.run_dir / "plan.json", self.plan)
        print("pose_T_camera_object_m:\n", np.array2string(camera_object, precision=6), flush=True)
        print("pose_grasp_point_pelvis_m:", self.plan["grasp_point_pelvis_m"], flush=True)
        print("pose_final_target_pelvis_m:", target.tolist(), flush=True)
        print("pose_z_mode:", self.plan["z_mode"], "hand_orientation: legacy_side_grasp_wrist", flush=True)
        print("pose_server_roundtrip_s: %.2f" % self.plan["server_roundtrip_s"], flush=True)
        return target

    def before_motion(self, args, announce):
        freshness = self.check_fresh()
        self.plan["pre_motion_freshness"] = freshness
        save_json(self.run_dir / "plan.json", self.plan)
        announce(args)

    def close(self):
        if self.monitor:
            self.monitor.close()


def main(argv=None):
    vision = None
    signal_handlers = {}
    try:
        argv = list(sys.argv[1:] if argv is None else argv)
        args, legacy = parse_arguments(argv)
        if args.pose_run_on_robot:
            return run_on_robot(args, argv)
        for signum in (signal.SIGHUP, signal.SIGTERM):
            signal_handlers[signum] = signal.signal(signum, stop_signal)
        vision = PoseVision(args, legacy)
        if args.pose_only:
            vision(args, np.asarray(args.pregrasp_target, dtype=float))
            print("Pose-only complete; no DDS participant, arm or hand commands were created.")
            return 0
        original_announce = legacy.announce_start
        legacy.parse_args = lambda: args
        legacy.compute_vision_grasp_target = vision
        legacy.announce_start = lambda parsed: vision.before_motion(parsed, original_announce)
        return legacy.main()
    except KeyboardInterrupt:
        print("Interrupted; no pose result is reused.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print("Free3Pose grasp failed: " + str(exc), file=sys.stderr)
        return 2
    finally:
        if vision:
            vision.close()
        for signum, handler in signal_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
