#!/usr/bin/env python3
"""Persistent YOLO-World RGB-D detection server for G1.

Start this once on the robot. It keeps the YOLO-World model alive in memory,
so repeated detections avoid model reload cost.
"""

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "yolov8s-world.pt"
DEFAULT_SERIAL = "243322071448"
DEFAULT_RGBD_NPZ_URL = "http://10.113.53.104:18082/rgbd.npz"
DEFAULT_CONF = 0.10
DEFAULT_IMGSZ = 640


def resolve_model_path(model_path):
    candidates = [
        Path.cwd() / model_path,
        Path(__file__).resolve().parent / model_path,
        ROOT / model_path,
        Path(model_path),
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path(model_path)


def best_detection(result, object_name):
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
        raise RuntimeError(f"no detection for object={object_name!r}")
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


class DetectorState:
    def __init__(self, args):
        self.args = args
        self.model_path = resolve_model_path(args.model)
        self.object_name = None
        self.request_count = 0
        self.lock = threading.Lock()

        from ultralytics import YOLOWorld

        t0 = time.monotonic()
        self.model = YOLOWorld(str(self.model_path))
        self.model_load_s = time.monotonic() - t0

        self.pipeline = None
        self.align = None
        self.pipeline_start_s = 0.0
        self.depth_scale = None
        self.fx = self.fy = self.cx = self.cy = None
        self.rs = None

        if args.source == "realsense":
            self.start_realsense_pipeline()

        self.set_object(args.object)
        if not args.no_warmup_predict:
            try:
                cap = self.capture_frame()
                self.predict(cap["color_bgr"], args.conf, args.imgsz)
            except Exception as exc:
                print(f"warmup_predict_failed: {exc}", flush=True)

    def set_object(self, object_name):
        if object_name == self.object_name:
            return
        self.model.set_classes([object_name])
        self.object_name = object_name

    def start_realsense_pipeline(self):
        if self.pipeline is not None:
            return

        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.args.serial:
            config.enable_device(self.args.serial)
        config.enable_stream(rs.stream.depth, self.args.width, self.args.height, rs.format.z16, self.args.fps)
        config.enable_stream(rs.stream.color, self.args.width, self.args.height, rs.format.bgr8, self.args.fps)

        t0 = time.monotonic()
        profile = self.pipeline.start(config)
        self.pipeline_start_s = time.monotonic() - t0
        self.align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        for _ in range(max(0, self.args.camera_warmup_frames)):
            self.capture_realsense_frame()

    def capture_realsense_frame(self):
        self.start_realsense_pipeline()
        frames = self.align.process(self.pipeline.wait_for_frames(self.args.frame_timeout_ms))
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("missing depth or color frame")

        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        self.fx = float(intr.fx)
        self.fy = float(intr.fy)
        self.cx = float(intr.ppx)
        self.cy = float(intr.ppy)
        return {
            "source": "realsense",
            "color_bgr": np.asanyarray(color_frame.get_data()).copy(),
            "depth_raw": np.asanyarray(depth_frame.get_data()).copy(),
            "depth_scale": self.depth_scale,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }

    def capture_frame(self):
        if self.args.source == "http-rgbd":
            try:
                with urllib.request.urlopen(self.args.rgbd_npz_url, timeout=self.args.http_timeout) as response:
                    payload = response.read()
                data = np.load(io.BytesIO(payload))
                self.depth_scale = float(data["depth_scale"])
                self.fx = float(data["fx"])
                self.fy = float(data["fy"])
                self.cx = float(data["cx"])
                self.cy = float(data["cy"])
                return {
                    "source": "http_rgbd_npz",
                    "color_bgr": np.asarray(data["color_bgr"]).copy(),
                    "depth_raw": np.asarray(data["depth_raw"]).copy(),
                    "depth_scale": self.depth_scale,
                    "fx": self.fx,
                    "fy": self.fy,
                    "cx": self.cx,
                    "cy": self.cy,
                    "stamp": float(data["stamp"]) if "stamp" in data.files else None,
                    "rgbd_npz_url": self.args.rgbd_npz_url,
                }
            except Exception as exc:
                if self.args.fallback_source == "realsense":
                    capture = self.capture_realsense_frame()
                    capture["http_rgbd_error"] = str(exc)
                    capture["rgbd_npz_url"] = self.args.rgbd_npz_url
                    return capture
                raise RuntimeError(f"failed to read HTTP RGB-D npz: {self.args.rgbd_npz_url}: {exc}") from exc

        return self.capture_realsense_frame()

    def predict(self, image_bgr, conf, imgsz):
        results = self.model.predict(
            source=image_bgr,
            conf=conf,
            imgsz=imgsz,
            stream=False,
            verbose=False,
        )
        if not results:
            raise RuntimeError("YOLO returned no result")
        return results[0]

    def detect(self, object_name, conf, imgsz):
        with self.lock:
            self.request_count += 1
            self.set_object(object_name)

            total_t0 = time.monotonic()
            capture_t0 = time.monotonic()
            capture = self.capture_frame()
            capture_s = time.monotonic() - capture_t0

            predict_t0 = time.monotonic()
            result = self.predict(capture["color_bgr"], conf, imgsz)
            predict_s = time.monotonic() - predict_t0

            det = best_detection(result, object_name)
            point_camera, depth_stats = depth_point_for_detection(capture, det)
            total_s = time.monotonic() - total_t0

            return {
                "ok": True,
                "object": object_name,
                "model": str(self.model_path),
                "source": capture.get("source", self.args.source),
                "request_count": self.request_count,
                "detection": det,
                "depth_stats": depth_stats,
                "point_camera_m": point_camera.tolist(),
                "intrinsics": {
                    "fx": capture["fx"],
                    "fy": capture["fy"],
                    "cx": capture["cx"],
                    "cy": capture["cy"],
                    "depth_scale": capture["depth_scale"],
                },
                "image_shape": list(capture["color_bgr"].shape),
                "stamp": capture.get("stamp"),
                "rgbd_npz_url": capture.get("rgbd_npz_url"),
                "http_rgbd_error": capture.get("http_rgbd_error"),
                "timing_s": {
                    "capture": capture_s,
                    "predict": predict_s,
                    "total": total_s,
                },
            }

    def health(self):
        return {
            "ok": True,
            "model": str(self.model_path),
            "source": self.args.source,
            "fallback_source": self.args.fallback_source,
            "model_load_s": self.model_load_s,
            "pipeline_start_s": self.pipeline_start_s,
            "object": self.object_name,
            "request_count": self.request_count,
            "camera": {
                "serial": self.args.serial,
                "rgbd_npz_url": self.args.rgbd_npz_url,
                "width": self.args.width,
                "height": self.args.height,
                "fps": self.args.fps,
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                "depth_scale": self.depth_scale,
            },
        }


def json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(state, default_conf, default_imgsz):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args), flush=True)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                json_response(self, 200, state.health())
                return
            json_response(self, 404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/detect":
                json_response(self, 404, {"ok": False, "error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(size).decode("utf-8") if size else "{}"
                req = json.loads(body or "{}")
                object_name = str(req.get("object") or state.object_name or "bottle")
                conf = float(req.get("conf", default_conf))
                imgsz = int(req.get("imgsz", default_imgsz))
                payload = state.detect(object_name, conf, imgsz)
                json_response(self, 200, payload)
            except Exception as exc:
                json_response(self, 500, {"ok": False, "error": str(exc)})

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Persistent YOLO-World RGB-D detection server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--object", default="bottle")
    parser.add_argument("--source", choices=("http-rgbd", "realsense"), default="http-rgbd")
    parser.add_argument(
        "--fallback-source",
        choices=("none", "realsense"),
        default="realsense",
        help="Fallback capture source when --source http-rgbd cannot read --rgbd-npz-url.",
    )
    parser.add_argument("--rgbd-npz-url", default=DEFAULT_RGBD_NPZ_URL)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--no-warmup-predict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    state = DetectorState(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, args.conf, args.imgsz))
    print(
        json.dumps(
            {
                "server": f"http://{args.host}:{args.port}",
                "health": f"http://{args.host}:{args.port}/health",
                "model": str(state.model_path),
                "source": args.source,
                "fallback_source": args.fallback_source,
                "rgbd_npz_url": args.rgbd_npz_url,
                "model_load_s": state.model_load_s,
                "pipeline_start_s": state.pipeline_start_s,
                "object": state.object_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        if state.pipeline is not None:
            state.pipeline.stop()


if __name__ == "__main__":
    main()
