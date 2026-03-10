# ♻️ AI Waste Management System
### Real-time waste detection & classification using YOLOv8

---

## Overview

This system uses **YOLOv8** (You Only Look Once v8) to detect waste objects from a live camera feed or image/video files and classify them into:

| Category | Color | Examples |
|---|---|---|
| 🟢 **Biodegradable** | Green | Food scraps, paper, cardboard, leaves |
| 🔵 **Recyclable** | Blue | Plastic bottles, cans, glass, cartons |
| 🔴 **Hazardous** | Red | Batteries, electronics, chemicals, syringes |

Live counts are displayed per frame and as running session totals.

---

## Project Structure

```
waste_management/
│
├── waste_detector.py     ← Core detection engine (YOLOv8 + category classifier)
├── camera_app.py         ← Live camera mode with HUD overlay
├── batch_process.py      ← Process image files / video files in bulk
├── train.py              ← Train custom model on waste datasets
├── requirements.txt      ← Python dependencies
└── README.md             ← This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> YOLOv8 (`ultralytics`) will **automatically download** the pretrained model weights (~6 MB for yolov8n) on first run.

### 2. Run live camera detection

```bash
python camera_app.py
```

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save screenshot |
| `R` | Reset session counts |

### 3. Process images or videos (batch mode)

```bash
# Single image
python batch_process.py my_image.jpg

# Entire folder
python batch_process.py ./my_photos/ --output ./results/

# Video file
python batch_process.py waste_video.mp4 --conf 0.45
```

---

## Training a Custom Model

### Option A — Synthetic demo dataset (no download, instant)

```bash
python train.py --mode synthetic --epochs 30
```
Use this to test the training pipeline locally.

---

### Option B — Roboflow Free Dataset (best quality, recommended)

1. Go to [roboflow.com](https://roboflow.com) → Create a free account
2. Get your **free API key** from the dashboard
3. Run:

```bash
python train.py --mode roboflow --roboflow-key YOUR_API_KEY --model s --epochs 50
```

Free waste datasets on Roboflow Universe:
- `waste-detection-yolo` (multi-class TACO-based)
- `garbage-classification-3`
- `recyclable-waste-types`

Search at: [universe.roboflow.com](https://universe.roboflow.com/?q=waste+detection)

---

### Option C — TACO Dataset (full research dataset, free)

TACO (Trash Annotations in Context) is a large open dataset:
- **GitHub:** https://github.com/pedropro/TACO
- **License:** Creative Commons
- **Size:** 1,500+ images, 4,784 annotations across 60 categories

```bash
python train.py --mode taco
# Follow the printed instructions to convert COCO → YOLO format
```

---

### Option D — Open Images Dataset (Google, free)

Use [FiftyOne](https://voxel51.com/fiftyone/) to download waste-related classes from Google's Open Images:

```bash
pip install fiftyone
python -c "
import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset(
    'open-images-v7',
    split='train',
    label_types=['detections'],
    classes=['Bottle', 'Can', 'Paper bag', 'Plastic bag'],
    max_samples=2000,
)
dataset.export('datasets/open_images_waste', dataset_type='yolov5')
"
```

---

## Using a Trained Model

After training, use your best model in the camera app:

```python
# In camera_app.py, change:
run_camera(model_path="waste_runs/waste_model/weights/best.pt")
```

---

## Configuration

### Confidence Threshold

In `camera_app.py`:
```python
run_camera(confidence=0.45)   # Higher = fewer false positives
```

### Camera Index

```python
run_camera(camera_index=1)    # 0 = default, 1+ for external cameras
```

### Model Size vs Speed

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | 6 MB | ⚡⚡⚡ Fast | Good |
| `yolov8s.pt` | 22 MB | ⚡⚡ Medium | Better |
| `yolov8m.pt` | 52 MB | ⚡ Slower | Best |

---

## Adding New Waste Classes

Edit `waste_detector.py`:
```python
WASTE_CATEGORIES = {
    "biodegradable": {
        "items": ["your_new_item", ...]  # Add here
    },
    ...
}
```

---

## Output Files

- **Screenshots:** `screenshots/waste_YYYYMMDD_HHMMSS.jpg`
- **Session report:** `screenshots/session_report_YYYYMMDD_HHMMSS.json`
- **Batch CSV:** `batch_output/batch_report_YYYYMMDD_HHMMSS.csv`

---

## Hardware Requirements

| Mode | Minimum | Recommended |
|------|---------|-------------|
| Detection (CPU) | 4 GB RAM, any CPU | 8 GB RAM, modern CPU |
| Detection (GPU) | NVIDIA GTX 1060 | NVIDIA RTX 3060+ |
| Training | 8 GB RAM, GPU | 16 GB RAM, NVIDIA GPU |

> CPU-only inference works fine for real-time detection at ~10–20 FPS with yolov8n.

---

## License

YOLOv8: [AGPL-3.0](https://github.com/ultralytics/ultralytics)  
TACO Dataset: [Creative Commons CC BY 4.0](https://tacodataset.org)
