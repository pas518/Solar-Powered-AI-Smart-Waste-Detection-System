"""
Batch Processor — run waste detection on image files or video files.
Outputs annotated images and a CSV/JSON count report.
"""

import cv2
import os
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from waste_detector import WasteDetector


def process_image(detector: WasteDetector, img_path: str, out_dir: str) -> dict:
    """Detect waste in a single image file, save annotated output."""
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  [SKIP] Cannot read: {img_path}")
        return {}

    annotated, detections, counts = detector.detect(frame)
    detector.update_session_counts(counts)

    stem = Path(img_path).stem
    out_path = os.path.join(out_dir, f"{stem}_detected.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"  [IMAGE] {stem}  →  BIO:{counts.get('biodegradable',0)}"
          f"  REC:{counts.get('recyclable',0)}  HAZ:{counts.get('hazardous',0)}")
    return {"file": img_path, "counts": counts, "detections": detections}


def process_video(detector: WasteDetector, video_path: str, out_dir: str,
                  every_n: int = 5) -> dict:
    """
    Process a video file; sample every N frames.
    Writes an annotated output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem = Path(video_path).stem
    out_path = os.path.join(out_dir, f"{stem}_detected.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    all_counts = {"biodegradable": 0, "recyclable": 0, "hazardous": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            annotated, _, counts = detector.detect(frame)
            for k in all_counts:
                all_counts[k] = max(all_counts[k], counts.get(k, 0))
            writer.write(annotated)
        else:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    detector.update_session_counts(all_counts)
    print(f"  [VIDEO] {stem}  →  BIO:{all_counts['biodegradable']}"
          f"  REC:{all_counts['recyclable']}  HAZ:{all_counts['hazardous']}")
    return {"file": video_path, "counts": all_counts}


def run_batch(input_path: str,
              output_dir: str = "batch_output",
              model_path: str = "yolov8n.pt",
              confidence: float = 0.40):
    """
    Process all images/videos in input_path (file or directory).
    Saves annotated outputs + CSV + JSON report.
    """
    os.makedirs(output_dir, exist_ok=True)
    detector = WasteDetector(model_path=model_path, confidence=confidence)

    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    files = []
    p = Path(input_path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.iterdir())
    else:
        print(f"❌ Input path not found: {input_path}")
        return

    all_results = []
    print(f"\n[Batch] Processing {len(files)} file(s)...\n")

    for fpath in files:
        ext = fpath.suffix.lower()
        if ext in IMAGE_EXT:
            r = process_image(detector, str(fpath), output_dir)
        elif ext in VIDEO_EXT:
            r = process_video(detector, str(fpath), output_dir)
        else:
            continue
        if r:
            all_results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = detector.get_session_summary()
    print(f"\n{'='*50}")
    print(f"  BATCH SUMMARY")
    print(f"  Files processed : {len(all_results)}")
    print(f"  Biodegradable   : {summary['biodegradable']}")
    print(f"  Recyclable      : {summary['recyclable']}")
    print(f"  Hazardous       : {summary['hazardous']}")
    print(f"  Total items     : {summary['total']}")
    print(f"{'='*50}\n")

    # Save CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"batch_report_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "biodegradable", "recyclable", "hazardous"])
        for r in all_results:
            c = r.get("counts", {})
            writer.writerow([r["file"],
                             c.get("biodegradable", 0),
                             c.get("recyclable", 0),
                             c.get("hazardous", 0)])
        writer.writerow(["TOTAL",
                         summary["biodegradable"],
                         summary["recyclable"],
                         summary["hazardous"]])
    print(f"  CSV report : {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, f"batch_report_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "files": all_results}, f, indent=2)
    print(f"  JSON report: {json_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch waste detection on images/videos")
    parser.add_argument("input", help="Image file, video file, or folder")
    parser.add_argument("--output", default="batch_output")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.40)
    args = parser.parse_args()
    run_batch(args.input, args.output, args.model, args.conf)
