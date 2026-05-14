#!/usr/bin/env python3
"""
Prepares a YOLOv5 training dataset directly from videos with 3 classes:
  0 — person
  1 — empty cart
  2 — non-empty cart

Runs YOLO detection + cart fill classification on sampled frames (every N frames).
Writes YOLO-format images + labels, splits by video into train/val, and
copies a calibration subset for INT8 export.

Usage:
    python prepare_yolo_dataset.py
    python prepare_yolo_dataset.py --data-folder path/to/videos/
    python prepare_yolo_dataset.py --sample-every 10 --conf-thresh 0.8
    python prepare_yolo_dataset.py --model path/to/custom/best.pt
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from engine.classifier import CartClassifier
from engine.config import (
    FILL_WEIGHT_PATH,
    MODEL_PATH,
    QUALITY_WEIGHT_PATH,
    YOLO_IMGSZ,
)

_CLASS_PERSON        = 0
_CLASS_EMPTY_CART    = 1
_CLASS_NONEMPTY_CART = 2
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def _bbox_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def _flush_batch(
    buffer: list[tuple[int, object]],
    model: YOLO,
    classifier: CartClassifier,
    conf_thresh: float,
    img_w: int,
    img_h: int,
) -> list[tuple[int, object, list[str]]]:
    """Run detection + cart classification on a batch of (frame_num, frame_bgr) pairs."""
    if not buffer:
        return []

    frame_nums = [n for n, _ in buffer]
    frames_bgr = [f for _, f in buffer]

    predictions = model.predict(
        source=frames_bgr,
        imgsz=YOLO_IMGSZ,
        conf=conf_thresh,
        verbose=False,
    )

    results = []
    for frame_num, frame_bgr, pred in zip(frame_nums, frames_bgr, predictions):
        if pred.boxes is None or len(pred.boxes) == 0:
            continue

        label_lines = []
        cart_boxes = []

        for box, cls_idx in zip(
            pred.boxes.xyxy.cpu().tolist(),
            pred.boxes.cls.cpu().tolist(),
        ):
            label = model.names[int(cls_idx)]
            if label == "person":
                cx, cy, w, h = _bbox_to_yolo(*box, img_w, img_h)
                label_lines.append(f"{_CLASS_PERSON} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            elif label == "cart":
                cart_boxes.append(box)

        if cart_boxes:
            int_bboxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in cart_boxes]
            cls_results = classifier.classify_batch(frame_bgr, int_bboxes, list(range(len(int_bboxes))))
            for i, box in enumerate(cart_boxes):
                fill = cls_results.get(i, {}).get("fill", "unclassified")
                if fill == "empty":
                    class_id = _CLASS_EMPTY_CART
                elif fill in ("partial", "full"):
                    class_id = _CLASS_NONEMPTY_CART
                else:
                    continue  # skip unclear / unclassified carts
                cx, cy, w, h = _bbox_to_yolo(*box, img_w, img_h)
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if label_lines:
            results.append((frame_num, frame_bgr, label_lines))

    return results


def _iter_video_samples(
    video_path: Path,
    model: YOLO,
    classifier: CartClassifier,
    sample_every: int,
    conf_thresh: float,
    infer_batch_size: int,
) -> list[tuple[int, object, list[str]]]:
    """
    Read a video, run YOLO + cart classification on every Nth frame, return
    (frame_num, frame_bgr, label_lines) tuples that have at least one detection.
    """
    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buffer: list[tuple[int, object]] = []
    all_results: list[tuple[int, object, list[str]]] = []
    frame_num = 0

    print(f"    {total} frames  →  ~{total // sample_every} sampled", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_num += 1

        if frame_num % sample_every != 0:
            continue

        buffer.append((frame_num, frame.copy()))

        if len(buffer) >= infer_batch_size:
            all_results.extend(_flush_batch(buffer, model, classifier, conf_thresh, img_w, img_h))
            buffer = []

    all_results.extend(_flush_batch(buffer, model, classifier, conf_thresh, img_w, img_h))
    cap.release()
    return all_results


def _write_split(
    split: str,
    video_paths: list[Path],
    output_dir: Path,
    model: YOLO,
    classifier: CartClassifier,
    sample_every: int,
    conf_thresh: float,
    infer_batch_size: int,
) -> tuple[dict, list[Path]]:
    img_dir   = output_dir / "images" / split
    label_dir = output_dir / "labels" / split

    stats = {"frames": 0, "persons": 0, "empty_carts": 0, "nonempty_carts": 0}
    saved_image_paths: list[Path] = []

    for video_path in video_paths:
        print(f"  [{split}] {video_path.name}")
        samples = _iter_video_samples(
            video_path, model, classifier, sample_every, conf_thresh, infer_batch_size
        )

        for frame_num, frame_bgr, label_lines in samples:
            name     = f"{video_path.stem}_f{frame_num:06d}"
            img_path = img_dir   / f"{name}.jpg"
            lbl_path = label_dir / f"{name}.txt"

            cv2.imwrite(str(img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lbl_path.write_text("\n".join(label_lines))

            stats["frames"]        += 1
            stats["persons"]       += sum(1 for l in label_lines if l.startswith(f"{_CLASS_PERSON} "))
            stats["empty_carts"]   += sum(1 for l in label_lines if l.startswith(f"{_CLASS_EMPTY_CART} "))
            stats["nonempty_carts"] += sum(1 for l in label_lines if l.startswith(f"{_CLASS_NONEMPTY_CART} "))
            saved_image_paths.append(img_path)

        print(f"    → {len(samples)} labeled frames saved")

    return stats, saved_image_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a 3-class YOLOv5 dataset from sampled video frames."
    )
    parser.add_argument(
        "--data-folder", default="sample_videos",
        help="Folder containing video files (default: sample_videos)",
    )
    parser.add_argument(
        "--output", default="datasets/gk-retail",
        help="Output dataset root (default: datasets/gk-retail)",
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Detection model weights (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--quality-weights", default=QUALITY_WEIGHT_PATH,
        help=f"Cart quality classifier weights (default: {QUALITY_WEIGHT_PATH})",
    )
    parser.add_argument(
        "--fill-weights", default=FILL_WEIGHT_PATH,
        help=f"Cart fill/bag classifier weights (default: {FILL_WEIGHT_PATH})",
    )
    parser.add_argument(
        "--sample-every", type=int, default=5,
        help="Run detection on every Nth frame (default: 5)",
    )
    parser.add_argument(
        "--conf-thresh", type=float, default=0.75,
        help="Minimum detection confidence to keep as a label (default: 0.75)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Fraction of videos held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--calib-frames", type=int, default=300,
        help="Frames to copy into calibration/ for INT8 export (default: 300)",
    )
    parser.add_argument(
        "--infer-batch", type=int, default=8,
        help="Frames per YOLO inference batch (default: 8)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="'auto', 'cuda', or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible train/val split (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    data_folder = Path(args.data_folder)
    output_dir  = Path(args.output)

    video_paths = sorted(
        p for p in data_folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        print(f"[ERROR] No video files found in '{data_folder}'")
        return

    print(f"Found {len(video_paths)} video(s) in '{data_folder}'")
    print(f"Loading detection model: {args.model}")
    model = YOLO(args.model)
    model.to(device)

    classifier = CartClassifier(device)
    if Path(args.quality_weights).exists():
        classifier.load_quality(args.quality_weights)
    else:
        print(f"[WARN] Quality weights not found: {args.quality_weights} — carts will be skipped")
    if Path(args.fill_weights).exists():
        classifier.load_fill(args.fill_weights)
    else:
        print(f"[WARN] Fill weights not found: {args.fill_weights} — carts will be skipped")

    print(f"Device: {device}  |  sample_every={args.sample_every}  conf={args.conf_thresh}\n")

    # Split by video — not frame — to avoid temporal leakage across train/val
    shuffled = list(video_paths)
    random.shuffle(shuffled)
    n_val      = max(1, round(len(shuffled) * args.val_split))
    val_paths   = shuffled[:n_val]
    train_paths = shuffled[n_val:]
    print(f"Split: {len(train_paths)} train / {len(val_paths)} val video(s)\n")

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "calibration").mkdir(parents=True, exist_ok=True)

    train_stats, train_images = _write_split(
        "train", train_paths, output_dir, model, classifier,
        args.sample_every, args.conf_thresh, args.infer_batch,
    )
    val_stats, _ = _write_split(
        "val", val_paths, output_dir, model, classifier,
        args.sample_every, args.conf_thresh, args.infer_batch,
    )

    # Calibration subset from train images only
    n_calib = min(args.calib_frames, len(train_images))
    for src in random.sample(train_images, n_calib):
        shutil.copy(src, output_dir / "images" / "calibration" / src.name)

    # dataset.yaml — absolute path so yolov5/train.py works from any cwd
    abs_output = output_dir.resolve()
    yaml_path  = output_dir / "dataset.yaml"
    yaml_path.write_text(
        f"# GK POPS — YOLOv5 detection dataset\n"
        f"path: {abs_output.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: 3\n"
        f"names: ['person', 'empty_cart', 'non_empty_cart']\n"
    )

    print("\n--- Dataset Summary ---")
    for split, s in [("train", train_stats), ("val", val_stats)]:
        print(
            f"  {split:5s}  {s['frames']:5d} frames  |"
            f"  {s['persons']:5d} person  |"
            f"  {s['empty_carts']:5d} empty cart  |"
            f"  {s['nonempty_carts']:5d} non-empty cart"
        )
    print(f"  calib  {n_calib:5d} frames")
    print(f"\nOutput: {abs_output}")
    print(f"YAML:   {yaml_path.resolve()}")


if __name__ == "__main__":
    main()
