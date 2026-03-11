# Solar-Powered-AI-Smart-Waste-Detection-System
AI-powered Smart Waste Management System that uses computer vision to detect and classify waste types using a camera. The system is powered by solar energy and aims to improve recycling efficiency and reduce environmental pollution.
# 🗑️ AI Smart Waste Bin v4
### Auto Bag Push-Out · New Bag Auto-Load · Conveyor Belt · Unsupervised AI Classification

> **Zero human contact. Solar powered. Self-managing smart waste bin with AI-driven waste classification, automatic bag ejection, and new bag auto-loading.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Hardware Components](#hardware-components)
- [AI Model](#ai-model)
- [Waste Categories](#waste-categories)
- [Dataset](#dataset)
- [Circuit & Power](#circuit--power)
- [Software Stack](#software-stack)
- [File Structure](#file-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Simulation](#simulation)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **AI Smart Waste Bin v4** is a fully autonomous waste sorting system designed for public spaces. A user throws a single sealed waste bag into the top slot — the bin does everything else. An auto-cutter slices the bag open, a conveyor belt carries each item past an AI camera, unsupervised machine learning classifies each piece of waste, and a 3-way servo diverter routes it into the correct compartment (Organic / Recyclable / Hazardous). When any compartment reaches 90% full, a servo-driven piston automatically pushes the sealed bag out through a side door flap, and a new bag unrolls and opens by itself — ready for the next cycle.

The entire system runs on a 20W solar panel with a 10Ah LiFePO4 battery and sends live fill data and bag-change alerts to a cloud IoT dashboard over WiFi.

---

## Key Features

| Feature | Description |
|---|---|
| **Single Bag Input** | User throws one sealed bag in — no sorting needed |
| **Auto Bag Cutter** | Rotary stainless-steel blade slices the bag open in < 2 seconds |
| **AI Camera + Conveyor** | Inside-lid camera scans each item at 30fps on the 0.3 m/s belt |
| **Unsupervised AI** | K-Means + DBSCAN + CNN classifies waste without manual labelling |
| **3-Way Servo Diverter** | Routes each item into Organic, Recyclable, or Hazardous bin |
| **Auto Bag Push-Out** ⭐ NEW | At 90% full, a piston seals and ejects the bag through a side flap |
| **New Bag Auto-Load** ⭐ NEW | Stepper motor unrolls and opens the next colour-coded bag instantly |
| **Auto-Brush Cleaning** | Brush sweeps bin walls clean after every bag change |
| **Solar Powered** | 20W panel + 10Ah LiFePO4 — 14 hours night operation |
| **IoT Dashboard** | WiFi sends live bin fill % and bag-change alerts to cloud |
| **Zero Human Contact** | From bag insertion to bag ejection — fully hands-free |

---

## How It Works

### Full Cycle

```
Throw sealed bag in
       ↓
Auto-cutter slices bag open (< 2s)
       ↓
Items fall onto conveyor belt (0.3 m/s)
       ↓
AI camera scans each item (30fps, 224×224)
       ↓
CNN extracts features → K-Means + DBSCAN clusters → class assigned
       ↓
3-way servo diverter routes item to correct bin
       ↓
[ORGANIC]    [RECYCLABLE]    [HAZARDOUS]
       ↓
Bin reaches 90% full trigger
       ↓
Auto-brush seals full bag from inside
       ↓
Servo piston pushes sealed bag out through side door flap
       ↓
Side flap closes automatically
       ↓
New colour-coded bag unrolls from cartridge
       ↓
Small arm spreads new bag open — ready for next waste
       ↓
Auto-brush cleans bin walls ✓
```

### Paste Amount Control

For semi-liquid or paste-like waste (food slurry, wet materials), ultrasonic fill sensors monitor the compartment level in real time. The system tracks cumulative weight and volume estimates to prevent overfilling and bag bursting. The sealed bag technique contains leakage from the moment of input.

### Brushing & Seal System

Before each bag push-out, the auto-brush sweeps downward from the top of the compartment, folding waste inward and forming a tight seal at the bag neck. A dual sensor confirms the seal is complete before the piston activates — preventing spills during ejection.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLOUD IoT DASHBOARD                      │
│              (WiFi · bin fill % · bag alerts)               │
└────────────────────────┬────────────────────────────────────┘
                         │ WiFi / MQTT
┌────────────────────────▼────────────────────────────────────┐
│                   RASPBERRY PI 4 (Main MCU)                  │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │  TF Lite AI  │  │  Servo Control  │  │  WiFi / MQTT   │  │
│  │  K-Means     │  │  Belt PWM       │  │  IoT Client    │  │
│  │  DBSCAN+CNN  │  │  Piston Driver  │  │  Alerts        │  │
│  └──────┬───────┘  └────────┬────────┘  └────────────────┘  │
└─────────│──────────────────│─────────────────────────────────┘
          │                  │
┌─────────▼──────────┐  ┌────▼──────────────────────────────┐
│   CAMERA MODULE    │  │           ESP32 (Low-level)        │
│  30fps · 224×224   │  │  NEMA17 stepper belt · PWM        │
│  OpenCV capture    │  │  Auto-cutter DC motor              │
└────────────────────┘  │  3-way servo diverter              │
                        │  Bag-load stepper motor            │
┌───────────────────┐   │  Auto-brush motor                  │
│  SOLAR + BATTERY  │   │  Fill-level ultrasonic sensors     │
│  20W panel        │   │  Side-door flap servo              │
│  MPPT controller  │   └────────────────────────────────────┘
│  10Ah LiFePO4     │
│  5V converter     │
└───────────────────┘
```

---

## Hardware Components

### Mechanical

| Component | Spec | Purpose |
|---|---|---|
| Rotary Cutter Blade | Stainless steel, DC motor | Slices sealed bag at entry |
| Conveyor Belt | 0.3 m/s rubber belt | Carries waste past camera |
| NEMA 17 Stepper Motor | ESP32 PWM controlled | Drives conveyor belt |
| 3-Way Servo Diverter | 180° servo, 3-position | Routes waste to correct bin |
| Push Piston | Servo-driven linear actuator | Ejects full bag through side flap |
| Side Door Flap | Spring-loaded servo hinge | Opens for ejection, auto-closes |
| Bag Roll Cartridge | Per compartment (3×) | Stores ~30 colour-coded bags |
| Bag-Load Stepper | Small stepper + spreader arm | Unrolls and opens new bag |
| Auto-Brush | DC motor, top-down sweep | Seals bag + cleans walls |
| Compartments | 3 sealed chambers | Organic / Recyclable / Hazardous |

### Electronics

| Component | Spec | Purpose |
|---|---|---|
| Raspberry Pi 4 | 4GB RAM | Main compute — AI inference, WiFi |
| ESP32 | Dual-core 240MHz | Low-level motor & sensor control |
| Camera Module | 30fps, 224×224 resolution | Item scanning on belt |
| Ultrasonic Sensors ×3 | HC-SR04 per compartment | Fill level detection |
| Solar Panel | 20W monocrystalline | Primary power source |
| MPPT Controller | 12V input | Maximises solar harvest |
| LiFePO4 Battery | 10Ah | Energy storage — 14hr night runtime |
| 5V Buck Converter | Input: 12V, Output: 5V | Powers electronics from battery |
| Servo Driver Board | PCA9685 I2C | Controls all servos |

---

## AI Model

### Pipeline

```
Camera Frame (224×224 RGB)
        ↓
Preprocessing (normalise, resize, denoise)
        ↓
CNN Feature Extraction (MobileNetV2 backbone, TF Lite)
        ↓
128-dimensional feature vector
        ↓
K-Means Clustering (k=3 initial groups)
        ↓
DBSCAN Refinement (noise filtering, outlier detection)
        ↓
Class Assignment: ORGANIC | RECYCLABLE | HAZARDOUS
        ↓
Servo Diverter Signal → correct bin
```

### Why Unsupervised?

The model uses **unsupervised learning** (K-Means + DBSCAN) rather than supervised classification for two key reasons:

1. **No manual labelling required** — waste types vary by region, season, and demographics. A supervised model trained on fixed categories would need constant retraining.
2. **Self-improving** — as the belt camera captures more images, the model refines cluster boundaries automatically. The longer it runs, the more accurate it gets.

The CNN backbone (MobileNetV2, TF Lite) is pre-trained on ImageNet for feature extraction only — it is never fine-tuned with labelled waste data.

### Model Details

| Property | Value |
|---|---|
| Backbone | MobileNetV2 (TF Lite, quantised) |
| Feature vector | 128-dimensional |
| Clustering | K-Means (k=3) + DBSCAN |
| Inference time | < 2 seconds per item |
| Runtime | TensorFlow Lite on Raspberry Pi 4 |
| Self-improvement | Clusters update from new belt images every 24 hours |

---

## Waste Categories

| Category | Colour Code | Examples | Destination |
|---|---|---|---|
| 🟢 **ORGANIC / BIO** | Green bag | Food scraps, peels, leaves, vegetable waste | Compost plant |
| 🔵 **RECYCLABLE** | Blue bag | Plastic, paper, cardboard, cartons | Recycling centre |
| 🟡 **METAL / GLASS** | Yellow bag | Cans, tins, glass bottles | Smelting / processing |
| 🔴 **HAZARDOUS** | Red bag | Batteries, syringes, chemicals, electronics | Special disposal |

> The bin currently sorts into 3 physical compartments (Organic, Recyclable, Hazardous). Metal/Glass is treated as Recyclable at the bin level and separated downstream at the recycling centre.

---

## Dataset

The AI model is trained and validated on a combined dataset of ~50,000 images after augmentation.

| Source | Images | Classes | Notes |
|---|---|---|---|
| **TrashNet** | 2,527 | 6 | Cardboard, glass, metal, paper, plastic, trash |
| **TACO** | 1,500 | 60 | Trash Annotations in Context — real-world diverse images |
| **Custom Belt Camera** | ~10,000 raw | 3 | Captured live from the bin's own conveyor belt |
| **After Augmentation** | ~50,000 | 3 | Flip, rotate, crop, brightness, noise augmentation |

### Data Augmentation Techniques

- Random horizontal and vertical flip
- Rotation (±30°)
- Random crop and resize
- Brightness and contrast jitter
- Gaussian noise injection
- Colour channel shuffle

---

## Circuit & Power

### Power Budget

| Component | Current Draw | Voltage |
|---|---|---|
| Raspberry Pi 4 | ~700 mA idle / 1.2 A load | 5V |
| ESP32 | ~80 mA | 3.3V |
| Camera Module | ~250 mA | 5V |
| NEMA 17 Stepper | ~1.5 A active | 12V |
| Servos (×4 total) | ~600 mA peak | 5V |
| DC Cutter Motor | ~500 mA active | 12V |
| Ultrasonic Sensors | ~15 mA × 3 | 5V |
| **Total (peak)** | **~5 A** | — |

### Solar + Battery Calculation

- **20W solar panel** → ~1.6 A at 12V in full sun
- **MPPT controller** → maximises harvest in partial shade
- **10Ah LiFePO4** at 12V = 120 Wh storage
- **Average consumption** ~15W → ~8 hours on battery alone
- **With solar top-up** → **14+ hours** night operation

### Wiring Overview

```
Solar Panel (12V)
      ↓
MPPT Controller
      ↓
LiFePO4 Battery (12V, 10Ah)
      ├── Buck Converter 12V→5V ──► Raspberry Pi 4
      ├── Buck Converter 12V→5V ──► ESP32 + Sensors
      ├── Buck Converter 12V→5V ──► Servos via PCA9685
      └── Direct 12V ─────────────► NEMA17 Stepper Driver (A4988)
                                  ► DC Cutter Motor Driver (L298N)
```

---

## Software Stack

| Layer | Technology |
|---|---|
| **Main Compute OS** | Raspberry Pi OS Lite (64-bit) |
| **AI / ML** | TensorFlow Lite, OpenCV, scikit-learn |
| **Camera Capture** | OpenCV (Python) |
| **Low-level Control** | MicroPython on ESP32 |
| **RPi ↔ ESP32 Comm** | UART serial (115200 baud) |
| **IoT Dashboard** | MQTT → Node-RED → cloud broker |
| **WiFi** | Raspberry Pi 4 built-in, MQTT protocol |
| **Simulation / Docs** | HTML5 Canvas + CSS animation |

### Key Python Libraries

```
tensorflow-lite      # AI inference
opencv-python        # Camera capture & preprocessing
scikit-learn         # K-Means, DBSCAN clustering
numpy                # Array operations
paho-mqtt            # IoT communication
RPi.GPIO             # GPIO pin control
pyserial             # UART to ESP32
```

---

## File Structure

```
ai-smart-waste-bin/
│
├── README.md                           # This file
│
├── simulation/
│   └── ai_smart_waste_bin.html         # Animated HTML simulation (1280×720)
│
├── hardware/
│   ├── circuit_diagram.png             # Full wiring diagram
│   ├── bom.csv                         # Bill of materials
│   └── 3d_models/                      # STL files for 3D-printed parts
│       ├── bin_body.stl
│       ├── cutter_housing.stl
│       ├── bag_cartridge.stl
│       └── side_flap.stl
│
├── firmware/
│   └── esp32/
│       ├── main.py                     # MicroPython entry point
│       ├── stepper.py                  # NEMA17 belt control
│       ├── servo_controller.py         # Diverter + piston + flap servos
│       ├── bag_loader.py               # Bag roll + auto-open logic
│       ├── cutter.py                   # DC motor cutter control
│       ├── brush.py                    # Auto-brush sweep logic
│       └── sensors.py                  # Ultrasonic fill sensors
│
├── software/
│   ├── main.py                         # Main Raspberry Pi entry point
│   ├── camera.py                       # OpenCV camera capture
│   ├── ai_classifier.py                # TF Lite + K-Means + DBSCAN pipeline
│   ├── bin_controller.py               # Compartment logic, fill tracking
│   ├── push_out_controller.py          # 90% trigger → piston → bag ejection
│   ├── iot_client.py                   # MQTT publish to dashboard
│   └── config.py                       # Pin assignments, thresholds, settings
│
├── model/
│   ├── mobilenetv2_features.tflite     # Pre-trained CNN feature extractor
│   ├── kmeans_clusters.pkl             # Saved K-Means cluster state
│   ├── train_clustering.py             # Re-train / update cluster script
│   └── augment_dataset.py              # Data augmentation pipeline
│
└── docs/
    ├── system_diagram.png
    ├── full_cycle_flowchart.pdf
    └── presentation_slide.html
```

---

## Setup & Installation

### Prerequisites

- Raspberry Pi 4 (4GB recommended) with Raspberry Pi OS Lite
- ESP32 development board
- Python 3.9+
- MicroPython flashed on ESP32

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-smart-waste-bin.git
cd ai-smart-waste-bin
```

### 2. Install Python Dependencies (Raspberry Pi)

```bash
pip install tensorflow opencv-python scikit-learn numpy paho-mqtt pyserial RPi.GPIO
```

### 3. Flash ESP32 Firmware

```bash
# Flash MicroPython to ESP32
esptool.py --chip esp32 erase_flash
esptool.py --chip esp32 write_flash -z 0x1000 micropython.bin

# Upload firmware files
ampy --port /dev/ttyUSB0 put firmware/esp32/main.py
ampy --port /dev/ttyUSB0 put firmware/esp32/stepper.py
ampy --port /dev/ttyUSB0 put firmware/esp32/servo_controller.py
ampy --port /dev/ttyUSB0 put firmware/esp32/bag_loader.py
ampy --port /dev/ttyUSB0 put firmware/esp32/cutter.py
ampy --port /dev/ttyUSB0 put firmware/esp32/brush.py
ampy --port /dev/ttyUSB0 put firmware/esp32/sensors.py
```

### 4. Configure Settings

Edit `software/config.py` to match your wiring:

```python
# UART to ESP32
SERIAL_PORT = '/dev/ttyS0'
SERIAL_BAUD = 115200

# Fill thresholds
FILL_TRIGGER_PCT  = 90    # % fill before push-out activates
BAG_LOW_ALERT_PCT = 10    # % remaining bags before app alert

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH  = 224
FRAME_HEIGHT = 224
FPS          = 30

# MQTT IoT
MQTT_BROKER = 'your-broker.io'
MQTT_PORT   = 1883
MQTT_TOPIC  = 'smartbin/status'
```

### 5. Run the System

```bash
cd software
python main.py
```

### 6. View the Simulation

Open `simulation/ai_smart_waste_bin.html` in any modern browser.
The animated slide is fixed at **1280×720px (16:9)** — ready for PowerPoint or display screens.

---

## Usage

### Normal Operation

1. Place the bin in a public area with the solar panel facing the sun.
2. Power on — the system boots and enters standby mode automatically.
3. User throws a **sealed waste bag** into the single top opening.
4. The bin handles everything automatically.
5. Ejected bags collect in the labelled trays beneath each compartment.
6. The city dashboard receives live fill % and alerts via WiFi.

### Bag Roll Replacement

1. Open the bottom access panel.
2. Load a new roll of colour-coded bags into the cartridge.
3. Thread the first bag into the auto-loader mechanism.
4. Close the panel — the system detects the new roll automatically.

> **Bag roll capacity:** ~30 bags per cartridge. The app sends an alert when fewer than 3 bags remain.

### Manual Override

In case of jam or sensor error:

```bash
# Connect via SSH on local network
ssh pi@smartbin.local

# Run manual override tool
python software/manual_override.py
# Options: open flap | run belt | trigger piston | reset sensors
```

---

## Simulation

The animated HTML simulation (`simulation/ai_smart_waste_bin.html`) runs entirely in-browser with no dependencies and demonstrates:

- Bag falling into the input slot
- Auto-cutter activating
- Camera scan beam sweeping across belt
- Conveyor belt moving with items (🥬 📦 🔋)
- Waste items arcing from belt and falling into correct bins
- AI classify tags flashing (🧠 BIO / 🧠 REC / 🧠 HAZ)
- Fill percentage incrementing in real time
- Bag roll spinners and new-bag-open animation
- Push piston extending and retracting
- Full cycle step highlighter (Bag In → Cut → Belt → AI Scan → Sorted → ... → Clean ✓)

**To embed in PowerPoint:** Insert → Object → Web Browser, or screenshot the slide at 1280×720px.

---

## Future Improvements

- [ ] Weight sensors per compartment for more precise fill estimation
- [ ] Odour sensor (MQ series) to help classify organic vs hazardous
- [ ] Compactor mechanism to compress contents and extend bag life
- [ ] Offline AI fallback — rule-based sorting when RPi is busy
- [ ] Multi-language voice feedback for public users
- [ ] QR code printed on ejected bags for downstream traceability
- [ ] Edge TPU (Coral Dev Board) for faster inference (< 0.5s)
- [ ] 4G/LTE modem for locations without WiFi infrastructure
- [ ] Carbon footprint tracker displayed on IoT dashboard
- [ ] Supervised fine-tuning mode — allow operators to label edge cases

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

- [TrashNet Dataset](https://github.com/garythung/trashnet) — Gary Thung & Mindy Yang
- [TACO Dataset](http://tacodataset.org/) — Pedro F. Proença & Pedro Simões
- [TensorFlow Lite](https://www.tensorflow.org/lite) — Google
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Sandler et al.
- [scikit-learn](https://scikit-learn.org/) — K-Means & DBSCAN implementation

---

<div align="center">

**Built for cleaner cities · Powered by the sun · Sorted by AI**

`Solar` · `IoT` · `TF Lite` · `K-Means` · `DBSCAN` · `RPi4` · `ESP32` · `Zero Human Contact`

</div>
