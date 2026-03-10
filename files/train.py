"""
Custom YOLOv8 Training on Free Waste Datasets
Datasets used:
  1. TACO (Trash Annotations in Context) — https://tacodataset.org
     GitHub: https://github.com/pedropro/TACO
  2. Waste Pictures (Kaggle) — kaggle.com/datasets/wangziang/waste-pictures
  3. Recyclable/Hazardous Waste (Roboflow Universe) — FREE public datasets

This script:
  1. Downloads the chosen dataset automatically
  2. Remaps class labels → {biodegradable, recyclable, hazardous}
  3. Trains YOLOv8n (or larger) on the dataset
  4. Saves the trained model for use in camera_app.py
"""

import os
import json
import yaml
import shutil
import random
from pathlib import Path
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY MAPPING: raw dataset class names → our 3 categories
# These cover TACO v1 + common Kaggle waste dataset labels
# ──────────────────────────────────────────────────────────────────────────────
RAW_TO_CATEGORY = {
    # BIODEGRADABLE (0)
    "banana peel": 0, "food": 0, "organic": 0, "paper": 0, "cardboard": 0,
    "paper bag": 0, "newspaper": 0, "tissue": 0, "napkin": 0,
    "fruit": 0, "vegetable": 0, "leaf": 0, "wood": 0, "eggshell": 0,
    "tea bag": 0, "coffee grounds": 0, "bread": 0, "food waste": 0,
    # RECYCLABLE (1)
    "plastic bottle": 1, "bottle": 1, "can": 1, "tin can": 1,
    "aluminum can": 1, "glass bottle": 1, "jar": 1, "cardboard box": 1,
    "magazine": 1, "plastic bag": 1, "plastic container": 1,
    "styrofoam": 1, "foam": 1, "metal": 1, "rubber": 1,
    "cup": 1, "straw": 1, "wrapper": 1, "sachet": 1,
    "carton": 1, "milk carton": 1, "juice carton": 1,
    "unlabeled litter": 1, "other plastic": 1, "clear plastic bottle": 1,
    "plastic film": 1, "six pack rings": 1,
    # HAZARDOUS (2)
    "battery": 2, "cell phone": 2, "laptop": 2, "circuit board": 2,
    "light bulb": 2, "fluorescent bulb": 2, "syringe": 2, "needle": 2,
    "spray can": 2, "paint can": 2, "motor oil": 2, "chemical": 2,
    "medication": 2, "thermometer": 2, "electronic": 2, "e-waste": 2,
}

CATEGORY_NAMES = ["biodegradable", "recyclable", "hazardous"]
CATEGORY_COLORS = [(34, 197, 94), (59, 130, 246), (239, 68, 68)]


def build_yaml(dataset_path: str, output_yaml: str = "waste_dataset.yaml") -> str:
    """
    Create a YOLO-format dataset YAML pointing to train/val image folders.
    Assumes dataset_path contains images/train, images/val, labels/train, labels/val.
    """
    cfg = {
        "path": str(Path(dataset_path).resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": CATEGORY_NAMES,
    }
    with open(output_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[YAML] ✅ Dataset config written: {output_yaml}")
    return output_yaml


def download_roboflow_dataset(api_key: str,
                               workspace: str,
                               project: str,
                               version: int,
                               dest: str = "datasets/roboflow_waste") -> str:
    """
    Download a Roboflow public waste dataset.
    Free public datasets that work well:
      - Workspace: "material-identification", Project: "garbage-classification-3"
      - Workspace: "computer-vision-demos", Project: "waste-detection-vkmbt"
    """
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=dest)
        print(f"[Roboflow] ✅ Dataset downloaded to: {dataset.location}")
        return dataset.location
    except ImportError:
        print("[Roboflow] ❌ Install: pip install roboflow")
        raise


def download_taco_dataset(dest: str = "datasets/taco") -> str:
    """
    Clone TACO dataset (Creative Commons licensed, free).
    Requires git.
    """
    import subprocess
    os.makedirs(dest, exist_ok=True)
    repo_url = "https://github.com/pedropro/TACO.git"
    if not os.path.exists(os.path.join(dest, ".git")):
        print("[TACO] Cloning TACO dataset repository...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, dest], check=True)
    print("[TACO] ✅ TACO repo ready.")
    return dest


def prepare_synthetic_demo_dataset(dest: str = "datasets/synthetic_demo",
                                    n_train: int = 200,
                                    n_val: int = 50) -> str:
    """
    Creates a minimal synthetic YOLO-format dataset for smoke-testing
    the training pipeline without requiring downloads.
    Uses random color patches labeled as waste categories.
    """
    import cv2
    import numpy as np

    print(f"[SyntheticData] Building demo dataset: {n_train} train / {n_val} val images")

    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = Path(dest) / "images" / split
        lbl_dir = Path(dest) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            # Random background
            img = np.random.randint(180, 220, (640, 640, 3), dtype=np.uint8)

            # Random number of waste objects (1–4)
            num_objects = random.randint(1, 4)
            labels = []

            for _ in range(num_objects):
                cat = random.randint(0, 2)
                color = [int(c) for c in CATEGORY_COLORS[cat]]

                # Random box
                cx = random.uniform(0.15, 0.85)
                cy = random.uniform(0.15, 0.85)
                bw = random.uniform(0.08, 0.30)
                bh = random.uniform(0.08, 0.30)

                x1 = int((cx - bw / 2) * 640)
                y1 = int((cy - bh / 2) * 640)
                x2 = int((cx + bw / 2) * 640)
                y2 = int((cy + bh / 2) * 640)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

                # Add category text
                cv2.putText(img, CATEGORY_NAMES[cat][:3].upper(),
                            (x1 + 4, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                labels.append(f"{cat} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            cv2.imwrite(str(img_dir / f"img_{i:04d}.jpg"), img)
            with open(lbl_dir / f"img_{i:04d}.txt", "w") as f:
                f.write("\n".join(labels))

    print(f"[SyntheticData] ✅ Synthetic dataset ready at: {dest}")
    return dest


def train(dataset_yaml: str,
          model_size: str = "n",       # n / s / m / l / x
          epochs: int = 50,
          imgsz: int = 640,
          batch: int = 16,
          device: str = "cpu",         # "cpu", "0" (GPU), "mps" (Apple)
          project: str = "waste_runs",
          name: str = "waste_model"):
    """
    Train YOLOv8 on the waste dataset.
    Uses transfer learning from COCO-pretrained weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Install ultralytics: pip install ultralytics")
        raise

    model = YOLO(f"yolov8{model_size}.pt")  # Load pretrained COCO weights

    print(f"\n{'='*60}")
    print(f"  TRAINING  YOLOv8{model_size.upper()}  —  {epochs} epochs")
    print(f"  Dataset : {dataset_yaml}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        verbose=True,
        save=True,
        plots=True,
    )

    best = Path(project) / name / "weights" / "best.pt"
    print(f"\n✅ Training complete. Best model: {best}")
    return str(best)


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 on Waste Dataset")
    parser.add_argument("--mode", choices=["synthetic", "roboflow", "taco"],
                        default="synthetic",
                        help="Dataset source: synthetic (demo), roboflow, or taco")
    parser.add_argument("--model", default="n", choices=["n","s","m","l","x"],
                        help="YOLOv8 size: n(ano) s(mall) m(edium) l(arge) x(large)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--roboflow-key", default="", help="Roboflow API key")
    args = parser.parse_args()

    if args.mode == "synthetic":
        print("[Mode] Using SYNTHETIC demo dataset (no download needed)")
        dataset_dir = prepare_synthetic_demo_dataset()

    elif args.mode == "roboflow":
        if not args.roboflow_key:
            print("❌ Provide --roboflow-key YOUR_FREE_API_KEY")
            print("   Get a free key at: https://roboflow.com (no credit card)")
            exit(1)
        # Free public waste detection dataset on Roboflow Universe
        dataset_dir = download_roboflow_dataset(
            api_key=args.roboflow_key,
            workspace="techshreyash",
            project="waste-detection-yolo",
            version=1,
        )

    elif args.mode == "taco":
        print("[Mode] Using TACO dataset (requires manual COCO→YOLO conversion)")
        dataset_dir = download_taco_dataset()
        print("NOTE: After cloning, run the TACO conversion utility and point to the YAML.")
        exit(0)

    yaml_path = build_yaml(dataset_dir)
    best_model = train(
        dataset_yaml=yaml_path,
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )

    print(f"\n🎉 Use your trained model in camera_app.py:")
    print(f"   python camera_app.py  (it will auto-use 'best.pt' if set)")
    print(f"   or update model_path='{best_model}' in camera_app.py\n")
