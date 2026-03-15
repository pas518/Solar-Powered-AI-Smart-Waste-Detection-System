"""
config.py — System Configuration
=================================
Single source of truth for all pin assignments, thresholds, and settings.
Edit this file to match your hardware wiring.
"""

from __future__ import annotations
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
MODEL_DIR   = BASE_DIR / "model"
LOG_DIR     = BASE_DIR / "logs"
DATA_DIR    = BASE_DIR / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TFLITE_MODEL  = str(MODEL_DIR / "mobilenetv2_features.tflite")
KMEANS_MODEL  = str(MODEL_DIR / "kmeans_clusters.pkl")
YOLO_WEIGHTS  = "yolov8n.pt"  # downloads automatically if ultralytics installed

# ══════════════════════════════════════════════════════════════════════════════
#  Raspberry Pi → ESP32 UART
# ══════════════════════════════════════════════════════════════════════════════
UART_PORT   = "/dev/ttyUSB0"   # or /dev/ttyAMA0 for GPIO UART
UART_BAUD   = 115200
UART_TIMEOUT= 0.5

# ══════════════════════════════════════════════════════════════════════════════
#  Camera
# ══════════════════════════════════════════════════════════════════════════════
CAMERA_INDEX    = 0          # 0=USB, -1=CSI (depends on driver)
CAMERA_WIDTH    = 1280
CAMERA_HEIGHT   = 720
CAMERA_FPS      = 30
CAMERA_WARMUP   = 2.5        # seconds for auto-exposure to settle
USE_GSTREAMER   = False      # True for RPi CSI via nvargus/v4l2

# Belt ROI: fraction of frame  (x1%, y1%, x2%, y2%)
BELT_ROI        = (0.10, 0.20, 0.90, 0.80)
MOTION_MIN_AREA = 3000       # px² — minimum foreground blob to trigger scan

# ══════════════════════════════════════════════════════════════════════════════
#  AI Classification Thresholds
# ══════════════════════════════════════════════════════════════════════════════
CONF_THRESHOLD      = 0.55   # minimum confidence to accept classification
ANOMALY_DISTANCE    = 36.0   # DBSCAN cluster distance for outlier flag
SCAN_DEBOUNCE_MS    = 800    # ms between consecutive classifications of same item
BELT_CAMERA_ZONE_X  = 0.34  # fraction of belt width where camera scans

# ══════════════════════════════════════════════════════════════════════════════
#  ESP32 Pin Assignments (MicroPython)
# ══════════════════════════════════════════════════════════════════════════════

# ── Conveyor Belt (NEMA17 stepper via A4988/DRV8825) ──
STEPPER_STEP_PIN   = 14
STEPPER_DIR_PIN    = 27
STEPPER_EN_PIN     = 26
STEPPER_STEPS_REV  = 200     # steps per revolution (1.8° stepper)
STEPPER_MICROSTEP  = 8       # microstepping factor
BELT_SPEED_RPS     = 1.0     # revolutions per second at normal speed
BELT_SLOW_RPS      = 0.3     # speed in camera zone

# ── Servo Pins (PWM, 50Hz) ──
SERVO_DIVERTER_PIN  = 13     # 3-way diverter servo
SERVO_FLAP_PIN      = 12     # top entry flap servo
SERVO_CLIP_PIN      = 25     # bag clip servo

# Diverter servo angles for each category
SERVO_DIVERTER_NEUTRAL = 90  # degrees
SERVO_DIVERTER_BIO     = 45  # left → biodegradable bin
SERVO_DIVERTER_REC     = 90  # centre → recyclable bin
SERVO_DIVERTER_HAZ     = 135 # right → hazardous bin

# ── Pneumatic Piston (relay or motor driver) ──
PISTON_EXTEND_PIN  = 33
PISTON_RETRACT_PIN = 32
PISTON_SEAL_PIN    = 35      # heat seal element
PISTON_EXTEND_MS   = 900     # ms to fully extend
PISTON_HOLD_MS     = 600     # ms to hold
PISTON_SEAL_MS     = 500     # ms to seal bag
PISTON_RETRACT_MS  = 750     # ms to retract

# ── Bag Cutter (DC motor via L298N) ──
CUTTER_IN1_PIN     = 5
CUTTER_IN2_PIN     = 18
CUTTER_EN_PIN      = 19      # PWM speed control
CUTTER_DUTY_CYCLE  = 0.90    # 90% duty = ~1150 RPM

# ── Bag Loader (stepper motor for roll feed) ──
BAGLOAD_STEP_PIN   = 2
BAGLOAD_DIR_PIN    = 4
BAGLOAD_STEPS_PER_BAG = 400  # steps to advance one bag length

# ── Auto-Brush (DC motor for belt cleaning sweep) ──
BRUSH_IN1_PIN      = 21
BRUSH_IN2_PIN      = 22
BRUSH_SWEEP_MS     = 1200    # one full sweep duration

# ── Ultrasonic Fill Sensors (HC-SR04, one per bin) ──
ULTRASONIC_BIO_TRIG  = 15
ULTRASONIC_BIO_ECHO  = 16
ULTRASONIC_REC_TRIG  = 17
ULTRASONIC_REC_ECHO  = 23
ULTRASONIC_HAZ_TRIG  = 34
ULTRASONIC_HAZ_ECHO  = 36

ULTRASONIC_TIMEOUT   = 0.03  # seconds
BIN_EMPTY_CM         = 40.0  # full empty bin depth in cm
BIN_FULL_THRESHOLD   = 0.90  # trigger push-out at 90% full

# ── Gas Sensors (analog via ADC) ──
MQ135_ADC_PIN     = 39       # CO2 / NH3 / organic vapour
MQ2_ADC_PIN       = 34       # LPG / propane / hydrogen
MQ7_ADC_PIN       = 35       # CO
MQ9_ADC_PIN       = 32       # CO / combustible gas

# ── Load Cell (HX711) ──
HX711_DATA_PIN    = 36
HX711_CLK_PIN     = 39
HX711_SCALE_FACTOR= 2280.0   # calibration scale

# ── Metal / Inductive Sensor ──
METAL_DETECT_PIN  = 37       # digital input (NPN NO)

# ── Moisture Sensor ──
MOISTURE_ADC_PIN  = 38

# ── Solar / Battery Monitoring ──
SOLAR_VOLTAGE_ADC  = 33      # voltage divider → ADC
BATTERY_VOLTAGE_ADC= 35
BATTERY_CURRENT_ADC= 34      # shunt resistor → current sense
BATTERY_SHUNT_OHM  = 0.01    # 10 mΩ shunt
SOC_FULL_V         = 3.65    # LiFePO4 full voltage per cell
SOC_EMPTY_V        = 2.80    # LiFePO4 empty voltage per cell

# ── I2C (SSD1306 OLED + BMP280) ──
I2C_SCL_PIN       = 22
I2C_SDA_PIN       = 21
OLED_ADDRESS      = 0x3C
BMP280_ADDRESS    = 0x76

# ══════════════════════════════════════════════════════════════════════════════
#  MQTT / IoT Settings
# ══════════════════════════════════════════════════════════════════════════════
MQTT_BROKER     = "mqtt.example.com"   # replace with your broker
MQTT_PORT       = 1883
MQTT_USERNAME   = ""
MQTT_PASSWORD   = ""
MQTT_CLIENT_ID  = "waste_bin_001"
MQTT_QOS        = 1
MQTT_KEEPALIVE  = 60

MQTT_TOPICS = {
    "detection":   "waste_bin/detection",
    "bin_levels":  "waste_bin/bins",
    "sensors":     "waste_bin/sensors",
    "status":      "waste_bin/status",
    "alert":       "waste_bin/alert",
    "telemetry":   "waste_bin/telemetry",
}

TELEMETRY_INTERVAL_S = 10    # publish telemetry every N seconds
ALERT_FULL_THRESHOLD  = 0.90 # publish alert when bin reaches 90%

# ══════════════════════════════════════════════════════════════════════════════
#  System Behaviour
# ══════════════════════════════════════════════════════════════════════════════
AUTO_SPAWN_INTERVAL_S  = 8.0    # seconds between auto-spawned items (demo mode)
WATCHDOG_TIMEOUT_S     = 30.0   # reboot ESP32 if unresponsive for this long
LOG_LEVEL              = "INFO"
LOG_ROTATE_MB          = 10     # rotate log when > 10 MB
LOG_BACKUP_COUNT       = 3

# ── Category display ──────────────────────────────────────────────────────────
CATEGORY_NAMES  = {0: "Biodegradable", 1: "Recyclable", 2: "Hazardous"}
CATEGORY_COLORS = {0: (0, 255, 136),   1: (34, 170, 255), 2: (255, 96, 32)}  # BGR
CATEGORY_SHORT  = {0: "BIO",           1: "REC",           2: "HAZ"}
