#!/usr/bin/env python3
"""Estimate a dynamic G1 placement target from an observed cardboard box.

The placement target is expressed in the G1 pelvis frame.  A fresh RGB-D
capture is taken, an open-vocabulary detector finds the box, and the depth at
the inner part of the detection is projected through the existing camera to
pelvis transform.  The caller can then add a small pelvis-frame offset to put
the hand above the foam pad instead of on the observed surface.

This module deliberately does not publish DDS commands or move the robot.  It
only performs perception and target calculation.
"""

import argparse
import os
import base64
import io
import json
import sys
import time
import urllib.request
import urllib.error
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GRASP_DIR = THIS_DIR.parents[0] / "grasp"
if str(GRASP_DIR) not in sys.path:
    sys.path.insert(0, str(GRASP_DIR))

from g1_portable_vision_grasp import (  # noqa: E402
    camera_to_pelvis,
    detect_object,
    load_model,
    run_capture,
    run_local_capture,
    run_navigation_rgbd_capture,
)


DEFAULT_BOX_OBJECT = "cardboard box"
# HTTP fallback when direct camera capture is unavailable or busy.
DEFAULT_NAV_RGBD_URL = "http://127.0.0.1:18082/rgbd.npz"
DEFAULT_NAV_RGBD_TIMEOUT = 10.0
# The torso-mounted D435 is pitched down.  The older real-arm grasp scripts
# use 0.8 rad for this same camera, while 0.35 rad makes a nearby floor point
# look much farther forward than it is.
DEFAULT_CAMERA_PITCH = 0.8
# This is an object-to-hand/palm offset, reused from the tested bottle grasp
# path.  The detected point is where the bottle should enter the box; the hand
# stays behind and above that point instead of needing to enter the box.
# Raise the hand before release so the bottle has the requested vertical drop.
# The X offset is slightly less conservative than the original tested value,
# bringing the hand closer to the box while preserving a reachable right-arm
# target; the detected bottle point remains at the box center because
# placement_region defaults to "center".
DEFAULT_PLACE_OFFSET = (-0.30, -0.04, 0.60)
DEFAULT_APPROACH_OFFSET = (0.0, -0.10, 0.0)
DEFAULT_INNER_FRACTION = 0.45
DEFAULT_PLACEMENT_REGION = "center"
DEFAULT_ARM_SAFE_BOUNDS = (0.10, 0.55, -0.55, 0.20, -0.10, 0.55)

# Fractions are measured inside the 2-D detector box.  The near side is lower
# in the RGB image for this forward-facing camera; the right-arm variant also
# moves toward the image-right side, which maps toward pelvis -Y.
PLACEMENT_REGION_ANCHORS = {
    "center": (0.50, 0.50),
    "near": (0.50, 0.72),
    "near-right": (0.72, 0.72),
}


def add_box_vision_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared dynamic-box options to a placement command parser."""

    parser.add_argument(
        "--box-vision",
        action="store_true",
        help="Detect the receiving cardboard box and derive placement targets from a fresh RGB-D frame.",
    )
    parser.add_argument(
        "--box-object",
        default=DEFAULT_BOX_OBJECT,
        help="Open-vocabulary detector text for the receiving box, e.g. 'cardboard box' or 'box'.",
    )
    parser.add_argument(
        "--box-vision-host",
        default="unitree@10.113.53.104",
        help="SSH host used by the optional direct-RealSense capture path.",
    )
    parser.add_argument("--box-vision-password", default=os.environ.get("ROBOT_SSH_PASSWORD", ""), help=argparse.SUPPRESS)
    parser.add_argument("--box-detector-server-url", default=None,
                        help="Persistent detector /detect URL; omit for local model loading.")
    parser.add_argument("--box-detector-timeout", type=float, default=30.0)
    parser.add_argument(
        "--box-vision-source",
        choices=("auto", "navigation-rgbd", "direct-realsense"),
        default="auto",
        help="auto: try local RealSense first, then HTTP RGB-D on capture failure.",
    )
    parser.add_argument(
        "--box-rgbd-url",
        default=DEFAULT_NAV_RGBD_URL,
        help="Navigation RGB-D HTTP endpoint returning an aligned .npz frame.",
    )
    parser.add_argument(
        "--box-rgbd-timeout",
        type=float,
        default=DEFAULT_NAV_RGBD_TIMEOUT,
        help="HTTP timeout in seconds for one navigation RGB-D frame.",
    )
    parser.add_argument(
        "--box-camera-pitch",
        type=float,
        default=DEFAULT_CAMERA_PITCH,
        help="Torso D435 pitch in radians for the RGB-D-to-pelvis transform; default matches the tested real-arm grasp path.",
    )
    parser.add_argument(
        "--box-place-offset",
        nargs=3,
        type=float,
        default=list(DEFAULT_PLACE_OFFSET),
        metavar=("X", "Y", "Z"),
        help="Pelvis-frame offset from the desired bottle point in the box to the hand/palm target. Default reuses the tested bottle grasp offset.",
    )
    parser.add_argument(
        "--box-approach-offset",
        nargs=3,
        type=float,
        default=list(DEFAULT_APPROACH_OFFSET),
        metavar=("X", "Y", "Z"),
        help="Pelvis-frame offset from the dynamic place target to the preplace approach target.",
    )
    parser.add_argument(
        "--box-inner-fraction",
        type=float,
        default=DEFAULT_INNER_FRACTION,
        help="Fraction of the 2-D box detection sampled around the selected placement region for robust depth.",
    )
    parser.add_argument(
        "--box-placement-region",
        choices=tuple(PLACEMENT_REGION_ANCHORS),
        default=DEFAULT_PLACEMENT_REGION,
        help="Where to place inside the detected box: center, near side, or near-right side.",
    )
    parser.add_argument(
        "--box-arm-safe-bounds",
        nargs=6,
        type=float,
        default=list(DEFAULT_ARM_SAFE_BOUNDS),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Conservative pelvis-frame arm bounds used to reject unreachable dynamic box targets.",
    )
    parser.add_argument(
        "--allow-box-outside-arm-workspace",
        action="store_true",
        help="Allow reporting/planning a box target outside the conservative arm bounds.",
    )


def point_in_bounds(point: np.ndarray, bounds) -> bool:
    """Return whether a pelvis-frame point is inside xmin/xmax/ymin/ymax/zmin/zmax."""

    values = np.asarray(bounds, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("arm safe bounds must contain six finite values")
    xmin, xmax, ymin, ymax, zmin, zmax = values
    if not (xmin <= xmax and ymin <= ymax and zmin <= zmax):
        raise ValueError("arm safe bounds must be ordered min then max")
    point = np.asarray(point, dtype=float)
    return bool(
        point.shape == (3,)
        and np.all(np.isfinite(point))
        and xmin <= point[0] <= xmax
        and ymin <= point[1] <= ymax
        and zmin <= point[2] <= zmax
    )


def suggested_base_shift_to_bounds(point: np.ndarray, bounds) -> np.ndarray:
    """Suggest a small planar base shift that would bring a point into bounds.

    This is diagnostic output only.  It is not a locomotion command.  The
    sign follows the pelvis convention: +X forward and +Y robot-left.
    """

    values = np.asarray(bounds, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("arm safe bounds must contain six finite values")
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point must contain three finite values")
    xmin, xmax, ymin, ymax, _zmin, _zmax = values
    shift = np.zeros(3, dtype=float)
    if point[0] > xmax:
        shift[0] = point[0] - xmax
    elif point[0] < xmin:
        shift[0] = point[0] - xmin
    if point[1] > ymax:
        shift[1] = point[1] - ymax
    elif point[1] < ymin:
        shift[1] = point[1] - ymin
    return shift


def depth_point_for_box(
    capture: Dict[str, object],
    detection: Dict[str, object],
    inner_fraction: float,
    placement_region: str = DEFAULT_PLACEMENT_REGION,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Project a robust depth point from a selected inner part of a box detection."""

    if not 0.05 <= float(inner_fraction) <= 1.0:
        raise ValueError("box inner fraction must be in [0.05, 1.0]")
    try:
        anchor_u, anchor_v = PLACEMENT_REGION_ANCHORS[str(placement_region)]
    except KeyError as exc:
        choices = ", ".join(PLACEMENT_REGION_ANCHORS)
        raise ValueError(f"unknown box placement region {placement_region!r}; choose {choices}") from exc

    depth_raw = np.asarray(capture["depth_raw"])
    if depth_raw.ndim != 2:
        raise RuntimeError(f"expected a 2-D depth image, got shape={depth_raw.shape}")
    depth_scale = float(capture["depth_scale"])
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise RuntimeError(f"invalid depth scale: {depth_scale!r}")

    height, width = depth_raw.shape
    x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
    sample_x = x1 + anchor_u * (x2 - x1)
    sample_y = y1 + anchor_v * (y2 - y1)
    crop_width = max(6.0, (x2 - x1) * float(inner_fraction))
    crop_height = max(6.0, (y2 - y1) * float(inner_fraction))
    x0 = max(0, int(round(sample_x - 0.5 * crop_width)))
    x3 = min(width, int(round(sample_x + 0.5 * crop_width)))
    y0 = max(0, int(round(sample_y - 0.5 * crop_height)))
    y3 = min(height, int(round(sample_y + 0.5 * crop_height)))
    if x3 <= x0 or y3 <= y0:
        raise RuntimeError("box detection inner depth crop is empty")

    depth_m = depth_raw[y0:y3, x0:x3].astype(np.float32) * depth_scale
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.05)]
    if valid.size == 0:
        raise RuntimeError("no valid depth samples in the inner cardboard-box detection")

    z = float(np.median(valid))
    fx = float(capture["fx"])
    fy = float(capture["fy"])
    cx = float(capture["cx"])
    cy = float(capture["cy"])
    if min(fx, fy) <= 0.0 or not all(np.isfinite(value) for value in (fx, fy, cx, cy)):
        raise RuntimeError("invalid RGB-D intrinsics")

    point_camera = np.asarray(
        [
            (sample_x - cx) * z / fx,
            (sample_y - cy) * z / fy,
            z,
        ],
        dtype=float,
    )
    stats = {
        "depth_crop_xyxy": [x0, y0, x3, y3],
        "inner_fraction": float(inner_fraction),
        "placement_region": str(placement_region),
        "placement_anchor_uv": [float(anchor_u), float(anchor_v)],
        "sample_pixel_xy": [float(sample_x), float(sample_y)],
        "valid_depth_samples": int(valid.size),
        "depth_median_m": z,
        "depth_percentile_10_m": float(np.percentile(valid, 10)),
        "depth_percentile_90_m": float(np.percentile(valid, 90)),
        "depth_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
    }
    return point_camera, stats


def capture_box_rgbd(args: Namespace) -> Dict[str, object]:
    """Prefer direct capture, falling back to aligned HTTP RGB-D."""
    source = getattr(args, "box_vision_source", "auto")
    url = getattr(args, "box_rgbd_url", DEFAULT_NAV_RGBD_URL)
    timeout = getattr(args, "box_rgbd_timeout", DEFAULT_NAV_RGBD_TIMEOUT)
    if source == "auto":
        try:
            print("capture: trying local RealSense", flush=True)
            return run_local_capture()
        except Exception as direct_error:
            print(f"capture: local RealSense failed: {direct_error}; trying HTTP {url}", flush=True)
            try:
                return run_navigation_rgbd_capture(url, timeout=timeout)
            except Exception as http_error:
                raise RuntimeError(
                    f"RGB-D capture failed; local: {direct_error}; HTTP: {http_error}"
                ) from http_error
    if source == "navigation-rgbd":
        return run_navigation_rgbd_capture(url, timeout=timeout)
    if source == "direct-realsense":
        return run_capture(Namespace(host=args.box_vision_host, password=args.box_vision_password))
    raise ValueError(f"Unknown box vision source: {source}")


def detect_box_remote(args, capture):
    # Send the exact RGB frame paired with the depth used below.
    buffer = io.BytesIO()
    np.save(buffer, capture["color_bgr"], allow_pickle=False)
    payload = json.dumps({
        "object": args.box_object,
        "image_npy": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }).encode("utf-8")
    request = urllib.request.Request(args.box_detector_server_url, data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.box_detector_timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error", body)
        except (ValueError, AttributeError):
            detail = body
        raise RuntimeError(
            f"Detector {args.box_detector_server_url} HTTP {exc.code}: {detail}"
        ) from exc
    if not result.get("ok") or result.get("source") != "request_image":
        raise RuntimeError("Detector must support image_npy; restart the updated detection server")
    if result.get("image_shape") != list(capture["color_bgr"].shape):
        raise RuntimeError("Detector image shape does not match captured RGB-D")
    detection = result["detection"]
    if detection.get("class_name") != args.box_object:
        raise RuntimeError("Detector returned an unexpected object class")
    bbox = np.asarray(detection["bbox_xyxy"], dtype=float)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise RuntimeError("Detector returned an invalid bounding box")
    return detection, result["model"]


def estimate_box_place_targets(args: Namespace) -> Dict[str, object]:
    """Capture the scene and return dynamic place/preplace targets."""

    started = time.monotonic()
    capture = capture_box_rgbd(args)
    captured = time.monotonic()
    if getattr(args, "box_detector_server_url", None):
        detection, model_path = detect_box_remote(args, capture)
    else:
        model, model_path = load_model(args.box_object)
        detection = detect_object(model, capture["color_bgr"], args.box_object)
    detected = time.monotonic()
    print(f"box_capture_seconds: {captured - started:.3f}", flush=True)
    print(f"box_detection_seconds: {detected - captured:.3f}", flush=True)
    point_camera, depth_stats = depth_point_for_box(
        capture,
        detection,
        args.box_inner_fraction,
        args.box_placement_region,
    )
    point_pelvis = camera_to_pelvis(point_camera, args.box_camera_pitch)
    place_offset = np.asarray(args.box_place_offset, dtype=float)
    approach_offset = np.asarray(args.box_approach_offset, dtype=float)
    if place_offset.shape != (3,) or approach_offset.shape != (3,):
        raise ValueError("box placement offsets must contain exactly three values")
    if not np.all(np.isfinite(place_offset)) or not np.all(np.isfinite(approach_offset)):
        raise ValueError("box placement offsets must be finite")

    place_target = point_pelvis + place_offset
    approach_target = place_target + approach_offset
    place_workspace_ok = point_in_bounds(place_target, args.box_arm_safe_bounds)
    approach_workspace_ok = point_in_bounds(approach_target, args.box_arm_safe_bounds)
    suggested_base_shift = suggested_base_shift_to_bounds(place_target, args.box_arm_safe_bounds)
    return {
        "box_object": str(args.box_object),
        "placement_region": str(args.box_placement_region),
        "model": str(model_path),
        "detection": detection,
        "point_camera_m": point_camera.tolist(),
        "point_pelvis_m": point_pelvis.tolist(),
        "place_offset_pelvis_m": place_offset.tolist(),
        "place_target_pelvis_m": place_target.tolist(),
        "approach_offset_pelvis_m": approach_offset.tolist(),
        "approach_target_pelvis_m": approach_target.tolist(),
        "arm_safe_bounds_pelvis_m": [float(value) for value in args.box_arm_safe_bounds],
        "place_workspace_ok": place_workspace_ok,
        "approach_workspace_ok": approach_workspace_ok,
        "suggested_base_shift_to_arm_workspace_m": suggested_base_shift.tolist(),
        "depth_stats": depth_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect a cardboard box and print a dynamic G1 placement target.")
    add_box_vision_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimate = estimate_box_place_targets(args)
    print(json.dumps(estimate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
