#!/usr/bin/env python3
"""Serve ROS RGB-D image topics as /rgbd.npz and /color.mjpg over HTTP."""

import argparse
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image


class FrameStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.color_bgr: Optional[np.ndarray] = None
        self.depth_raw: Optional[np.ndarray] = None
        self.depth_scale = 0.001
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.stamp = 0.0

    def set_color(self, msg: Image) -> None:
        image = image_to_numpy(msg)
        if msg.encoding in ("rgb8", "rgba8"):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "bgra8":
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif msg.encoding == "mono8":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif msg.encoding != "bgr8":
            raise ValueError(f"unsupported color encoding: {msg.encoding}")
        with self.lock:
            self.color_bgr = np.ascontiguousarray(image[:, :, :3])
            self.stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()

    def set_depth(self, msg: Image) -> None:
        depth = image_to_numpy(msg)
        if msg.encoding == "32FC1":
            depth_m = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            depth_raw = np.clip(depth_m / self.depth_scale, 0, 65535).astype(np.uint16)
        elif msg.encoding in ("16UC1", "mono16"):
            depth_raw = depth.astype(np.uint16, copy=False)
        else:
            raise ValueError(f"unsupported depth encoding: {msg.encoding}")
        with self.lock:
            self.depth_raw = np.ascontiguousarray(depth_raw)
            self.stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()

    def set_info(self, msg: CameraInfo) -> None:
        with self.lock:
            self.fx = float(msg.K[0])
            self.fy = float(msg.K[4])
            self.cx = float(msg.K[2])
            self.cy = float(msg.K[5])

    def snapshot(self):
        with self.lock:
            if self.color_bgr is None or self.depth_raw is None:
                return None
            fx = self.fx if self.fx is not None else 0.0
            fy = self.fy if self.fy is not None else 0.0
            cx = self.cx if self.cx is not None else 0.0
            cy = self.cy if self.cy is not None else 0.0
            return (
                self.color_bgr.copy(),
                self.depth_raw.copy(),
                float(self.depth_scale),
                float(fx),
                float(fy),
                float(cx),
                float(cy),
                float(self.stamp),
            )


def image_to_numpy(msg: Image) -> np.ndarray:
    dtype, channels = encoding_dtype_channels(msg.encoding)
    array = np.frombuffer(msg.data, dtype=dtype)
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    itemsize = np.dtype(dtype).itemsize
    row_values = step // itemsize
    if channels == 1:
        shaped = array.reshape((height, row_values))[:, :width]
    else:
        shaped = array.reshape((height, row_values // channels, channels))[:, :width, :]
    if msg.is_bigendian:
        shaped = shaped.byteswap().newbyteorder()
    return np.array(shaped, copy=True)


def encoding_dtype_channels(encoding: str):
    mapping = {
        "bgr8": (np.uint8, 3),
        "rgb8": (np.uint8, 3),
        "bgra8": (np.uint8, 4),
        "rgba8": (np.uint8, 4),
        "mono8": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        "16UC1": (np.uint16, 1),
        "32FC1": (np.float32, 1),
    }
    if encoding not in mapping:
        raise ValueError(f"unsupported image encoding: {encoding}")
    return mapping[encoding]


def make_handler(store: FrameStore, jpeg_quality: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                snapshot = store.snapshot()
                self.send_response(200 if snapshot is not None else 503)
                self.end_headers()
                self.wfile.write(b"ok\n" if snapshot is not None else b"waiting_for_frames\n")
                return
            if self.path == "/rgbd.npz":
                self.send_rgbd()
                return
            if self.path == "/color.mjpg":
                self.send_mjpg()
                return
            self.send_response(404)
            self.end_headers()

        def send_rgbd(self) -> None:
            snapshot = store.snapshot()
            if snapshot is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"waiting_for_frames\n")
                return
            color_bgr, depth_raw, depth_scale, fx, fy, cx, cy, stamp = snapshot
            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                color_bgr=color_bgr,
                depth_raw=depth_raw,
                depth_scale=np.asarray(depth_scale, dtype=np.float64),
                fx=np.asarray(fx, dtype=np.float64),
                fy=np.asarray(fy, dtype=np.float64),
                cx=np.asarray(cx, dtype=np.float64),
                cy=np.asarray(cy, dtype=np.float64),
                stamp=np.asarray(stamp, dtype=np.float64),
            )
            payload = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_mjpg(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while not rospy.is_shutdown():
                snapshot = store.snapshot()
                if snapshot is None:
                    time.sleep(0.05)
                    continue
                color_bgr = snapshot[0]
                ok, encoded = cv2.imencode(
                    ".jpg",
                    color_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not ok:
                    time.sleep(0.05)
                    continue
                frame = encoded.tobytes()
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(0.03)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--color-topic", default="/d455/color/image_raw")
    parser.add_argument("--depth-topic", default="/d455/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/d455/color/camera_info")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = FrameStore()
    store.depth_scale = float(args.depth_scale)
    rospy.init_node("g1_ros_rgbd_http_server", anonymous=True, disable_signals=True)
    rospy.Subscriber(args.color_topic, Image, store.set_color, queue_size=1)
    rospy.Subscriber(args.depth_topic, Image, store.set_depth, queue_size=1)
    rospy.Subscriber(args.camera_info_topic, CameraInfo, store.set_info, queue_size=1)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.jpeg_quality))
    print(f"serving ROS RGB-D HTTP on {args.host}:{args.port}", flush=True)
    print(f"color_topic={args.color_topic}", flush=True)
    print(f"depth_topic={args.depth_topic}", flush=True)
    print(f"camera_info_topic={args.camera_info_topic}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
