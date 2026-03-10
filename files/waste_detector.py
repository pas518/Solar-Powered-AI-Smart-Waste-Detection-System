"""
Waste Management AI Detection System
Uses YOLOv8 to detect and classify waste into:
  - Biodegradable
  - Recyclable
  - Hazardous
"""

import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
import json
import os

# ─── Waste Category Mapping ───────────────────────────────────────────────────
# Maps TACO/COCO waste class names → our 3 categories
WASTE_CATEGORIES = {
    # BIODEGRADABLE
    "biodegradable": {
        "color": (34, 197, 94),      # Green
        "items": [
            "food", "fruit", "vegetable", "banana", "apple", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "sandwich", "bread", "egg", "meat", "fish", "leaf", "plant",
            "flower", "wood", "paper bag", "cardboard", "newspaper",
            "tissue", "napkin", "paper",
        ]
    },
    # RECYCLABLE
    "recyclable": {
        "color": (59, 130, 246),     # Blue
        "items": [
            "bottle", "plastic bottle", "water bottle", "wine glass", "cup",
            "can", "tin can", "aluminum can", "soda can", "beer can",
            "glass bottle", "jar", "cardboard box", "box", "carton",
            "magazine", "book", "notebook", "metal", "steel", "aluminum",
            "plastic bag", "plastic container", "tray", "foam", "styrofoam",
            "bubble wrap", "plastic", "rubber", "tire",
        ]
    },
    # HAZARDOUS
    "hazardous": {
        "color": (239, 68, 68),      # Red
        "items": [
            "battery", "cell phone", "mobile phone", "laptop", "computer",
            "keyboard", "mouse", "remote", "tv", "monitor", "tablet",
            "circuit board", "wire", "cable", "light bulb", "fluorescent",
            "needle", "syringe", "medicine", "pill", "spray can",
            "paint", "chemical", "oil", "motor oil", "pesticide",
            "bleach", "detergent", "acid", "solvent", "thermometer",
        ]
    }
}

# Build reverse lookup: item_name → category
ITEM_TO_CATEGORY = {}
for cat, data in WASTE_CATEGORIES.items():
    for item in data["items"]:
        ITEM_TO_CATEGORY[item.lower()] = cat

# COCO class names (used by YOLOv8 pretrained model)
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]

def classify_waste(label: str) -> str:
    """Classify a detected object label into a waste category."""
    label_lower = label.lower().strip()
    
    # Direct match
    if label_lower in ITEM_TO_CATEGORY:
        return ITEM_TO_CATEGORY[label_lower]
    
    # Partial match
    for key, cat in ITEM_TO_CATEGORY.items():
        if key in label_lower or label_lower in key:
            return cat
    
    # Keyword heuristics
    hazardous_kw = ["battery","electric","electronic","chemical","medical","toxic","phone","laptop","bulb","needle"]
    bio_kw = ["food","organic","fruit","vegetable","leaf","plant","wood","paper","bread","egg"]
    
    for kw in hazardous_kw:
        if kw in label_lower:
            return "hazardous"
    for kw in bio_kw:
        if kw in label_lower:
            return "biodegradable"
    
    # Default: recyclable (most common waste)
    return "recyclable"


class WasteDetector:
    """
    Real-time waste detection system using YOLOv8.
    Classifies detections into biodegradable / recyclable / hazardous.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.40):
        self.confidence = confidence
        self.model = None
        self.model_path = model_path
        self.session_counts = defaultdict(int)   # category → count
        self.detection_log = []                  # history of detections
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model (downloads automatically on first run)."""
        try:
            from ultralytics import YOLO
            print(f"[WasteDetector] Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("[WasteDetector] ✅ Model loaded successfully.")
        except ImportError:
            print("[WasteDetector] ❌ ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as e:
            print(f"[WasteDetector] ❌ Failed to load model: {e}")
            raise

    def detect(self, frame: np.ndarray):
        """
        Run detection on a single frame.
        Returns:
            annotated_frame: BGR image with bounding boxes drawn
            results_data: list of detection dicts
            category_counts: dict {category: count_in_this_frame}
        """
        if self.model is None:
            return frame, [], {}

        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        detections = []
        frame_counts = defaultdict(int)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
            category = classify_waste(label)

            detections.append({
                "label": label,
                "category": category,
                "confidence": round(conf_score, 3),
                "bbox": [x1, y1, x2, y2],
                "timestamp": datetime.now().isoformat()
            })
            frame_counts[category] += 1

        annotated = self._draw_boxes(frame.copy(), detections)
        return annotated, detections, dict(frame_counts)

    def _draw_boxes(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw annotated bounding boxes on frame."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cat = det["category"]
            color = WASTE_CATEGORIES[cat]["color"]
            label = det["label"]
            conf = det["confidence"]

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            text = f"{label} [{cat[:3].upper()}] {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            # Label text
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def update_session_counts(self, frame_counts: dict):
        """Accumulate per-frame counts into session totals."""
        for cat, cnt in frame_counts.items():
            self.session_counts[cat] = max(self.session_counts[cat], cnt)

    def get_session_summary(self) -> dict:
        return {
            "biodegradable": self.session_counts.get("biodegradable", 0),
            "recyclable": self.session_counts.get("recyclable", 0),
            "hazardous": self.session_counts.get("hazardous", 0),
            "total": sum(self.session_counts.values()),
        }

    def reset_session(self):
        self.session_counts.clear()
        self.detection_log.clear()
        print("[WasteDetector] Session reset.")
