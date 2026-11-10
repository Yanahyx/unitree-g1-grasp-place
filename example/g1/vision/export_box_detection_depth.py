#!/usr/bin/env python3
"""Export RGB-D images with the existing cardboard-box detector overlay.

This script is intentionally perception-only: it captures the existing
RealSense stream, runs the same YOLO-World box detector used by the placement
pipeline, and saves RGB/depth visualizations.  It does not publish DDS
commands or move the robot.
"""

import argparse
import os
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GRASP_DIR = THIS_DIR.parents[0] / "grasp"
if str(GRASP_DIR) not in sys.path:
    sys.path.insert(0, str(GRASP_DIR))

from g1_box_place_target import depth_point_for_box  # noqa: E402
from g1_portable_vision_grasp import (  # noqa: E402
    camera_to_pelvis,
    detect_object,
    load_model,
    run_capture,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save RGB-D frames and cardboard-box detection overlays."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.8)
    parser.add_argument("--object", default="cardboard box")
    parser.add_argument("--camera-pitch", type=float, default=0.8)
    parser.add_argument("--inner-fraction", type=float, default=0.45)
    parser.add_argument("--placement-region", default="center")
    parser.add_argument("--host", default="unitree@10.113.53.104")
    parser.add_argument("--password", default=os.environ.get("ROBOT_SSH_PASSWORD", ""))
    parser.add_argument("--depth-min-m", type=float, default=0.20)
    parser.add_argument("--depth-max-m", type=float, default=2.50)
    return parser.parse_args()


def point_tuple(values):
    return tuple(int(round(float(value))) for value in values)


def draw_cross(image, xy, color, size=12, thickness=2):
    x, y = point_tuple(xy)
    cv2.drawMarker(
        image,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )


def draw_label(image, text, origin, color=(255, 255, 255)):
    x, y = point_tuple(origin)
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def annotate_image(image_bgr, detection, depth_stats, point_camera, point_pelvis):
    image = np.asarray(image_bgr).copy()
    x1, y1, x2, y2 = point_tuple(detection["bbox_xyxy"])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 3)
    draw_cross(image, detection["center_xy"], (0, 0, 255), size=18, thickness=3)
    draw_cross(image, depth_stats["sample_pixel_xy"], (255, 0, 0), size=18, thickness=3)
    draw_label(
        image,
        f"{detection['class_name']} {detection['confidence']:.3f}",
        (x1, max(24, y1 - 10)),
        (0, 255, 0),
    )
    draw_label(image, "red=det center  blue=depth sample", (12, 28), (255, 255, 255))
    draw_label(
        image,
        f"depth {depth_stats['depth_median_m']:.3f} m",
        (12, 54),
        (255, 220, 0),
    )
    draw_label(
        image,
        "camera xyz " + " ".join(f"{value:.3f}" for value in point_camera),
        (12, 80),
        (255, 220, 0),
    )
    draw_label(
        image,
        "pelvis xyz " + " ".join(f"{value:.3f}" for value in point_pelvis),
        (12, 106),
        (255, 220, 0),
    )
    return image


def depth_colormap(capture, detection, depth_stats, min_m, max_m):
    depth_raw = np.asarray(capture["depth_raw"])
    depth_m = depth_raw.astype(np.float32) * float(capture["depth_scale"])
    clipped = np.clip(depth_m, float(min_m), float(max_m))
    normalized = ((clipped - float(min_m)) / (float(max_m) - float(min_m)) * 255.0).astype(
        np.uint8
    )
    result = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    valid = np.isfinite(depth_m) & (depth_m > 0.05)
    result[~valid] = (0, 0, 0)

    x1, y1, x2, y2 = point_tuple(detection["bbox_xyxy"])
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    crop = point_tuple(depth_stats["depth_crop_xyxy"])
    cv2.rectangle(result, (crop[0], crop[1]), (crop[2], crop[3]), (255, 0, 0), 2)
    draw_cross(result, detection["center_xy"], (0, 0, 255), size=18, thickness=3)
    draw_cross(result, depth_stats["sample_pixel_xy"], (255, 0, 0), size=18, thickness=3)
    draw_label(result, f"depth colormap {min_m:.2f}-{max_m:.2f} m", (12, 28))
    draw_label(result, "green=box  blue=crop  red=center", (12, 54))
    draw_label(
        result,
        f"median {depth_stats['depth_median_m']:.3f} m",
        (12, 80),
        (255, 220, 0),
    )
    return result, depth_m


def save_frame(output_dir, index, capture, detection, depth_stats, point_camera, point_pelvis, args):
    color_bgr = np.asarray(capture["color_bgr"])
    annotated = annotate_image(color_bgr, detection, depth_stats, point_camera, point_pelvis)
    depth_vis, depth_m = depth_colormap(
        capture,
        detection,
        depth_stats,
        args.depth_min_m,
        args.depth_max_m,
    )
    side_by_side = np.hstack([annotated, depth_vis])
    stem = f"box_{index:03d}"
    jpeg_options = [cv2.IMWRITE_JPEG_QUALITY, 95]
    if not cv2.imwrite(str(output_dir / f"{stem}_rgb_raw.jpg"), color_bgr, jpeg_options):
        raise RuntimeError("failed to save raw RGB image")
    if not cv2.imwrite(str(output_dir / f"{stem}_rgb_detection.jpg"), annotated, jpeg_options):
        raise RuntimeError("failed to save annotated RGB image")
    if not cv2.imwrite(str(output_dir / f"{stem}_depth_colormap.png"), depth_vis):
        raise RuntimeError("failed to save depth colormap")
    if not cv2.imwrite(str(output_dir / f"{stem}_rgb_depth_side_by_side.jpg"), side_by_side, jpeg_options):
        raise RuntimeError("failed to save RGB/depth comparison")
    np.save(str(output_dir / f"{stem}_depth_m.npy"), depth_m)
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(output_dir / f"{stem}_depth_mm.png"), depth_mm):
        raise RuntimeError("failed to save 16-bit depth image")

    metadata = {
        "frame_index": index,
        "color_shape": list(color_bgr.shape),
        "depth_shape": list(depth_m.shape),
        "depth_scale": float(capture["depth_scale"]),
        "intrinsics": {
            key: float(capture[key]) for key in ("fx", "fy", "cx", "cy")
        },
        "detection": detection,
        "depth_stats": depth_stats,
        "point_camera_m": [float(value) for value in point_camera],
        "point_pelvis_m": [float(value) for value in point_pelvis],
        "camera_pitch_rad": float(args.camera_pitch),
        "depth_colormap_range_m": [float(args.depth_min_m), float(args.depth_max_m)],
    }
    (output_dir / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def main():
    args = parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be at least 1")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_args = argparse.Namespace(host=args.host, password=args.password)

    print("loading existing box detector", flush=True)
    model, model_path = load_model(args.object)
    print(f"model={model_path}", flush=True)
    summary = {
        "object": args.object,
        "model": str(model_path),
        "frames_requested": int(args.frames),
        "frames_saved": 0,
        "errors": [],
    }
    for index in range(1, args.frames + 1):
        if index > 1 and args.interval > 0:
            time.sleep(args.interval)
        try:
            print(f"capturing frame {index}/{args.frames}", flush=True)
            capture = run_capture(capture_args)
            detection = detect_object(model, capture["color_bgr"], args.object)
            point_camera, depth_stats = depth_point_for_box(
                capture,
                detection,
                args.inner_fraction,
                args.placement_region,
            )
            point_pelvis = camera_to_pelvis(point_camera, args.camera_pitch)
            save_frame(
                output_dir,
                index,
                capture,
                detection,
                depth_stats,
                point_camera,
                point_pelvis,
                args,
            )
            summary["frames_saved"] += 1
            print(
                "saved "
                f"confidence={detection['confidence']:.3f} "
                f"depth={depth_stats['depth_median_m']:.3f}m",
                flush=True,
            )
        except Exception as exc:
            message = f"frame {index}: {type(exc).__name__}: {exc}"
            summary["errors"].append(message)
            print(message, file=sys.stderr, flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if summary["frames_saved"] == 0:
        raise SystemExit("no frame was exported")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
