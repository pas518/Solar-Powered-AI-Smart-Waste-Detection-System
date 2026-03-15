# AI Smart Waste Bin — Complete Project

Automated waste sorting: Camera → YOLOv8/HOG → KMeans → BIO/REC/HAZ bins → MQTT IoT

## Quick Start

```bash
# Train classifier (synthetic mode, no images needed)
python model/train_clustering.py --synthetic

# Run full system demo
python software/main.py --demo --no-uart

# Live camera mode (on Raspberry Pi)
python software/main.py --show
```

## AI Pipeline
Camera → Preprocess → YOLOv8n detection → MobileNetV2/HOG features → PCA → KMeans(k=3) → DBSCAN anomaly

## Structure
firmware/esp32/  — MicroPython: stepper, servos, sensors, bag loader, cutter
software/        — Python: camera, AI classifier, bin controller, push-out, MQTT
model/           — KMeans training + data augmentation

See full docs in docs/system_diagram.png
