"""
main.py — AI Smart Waste Bin — Raspberry Pi Main Entry Point
=============================================================
Orchestrates the complete system:

  Camera → Preprocess → AI Classifier → Bin Controller
      → Push-Out Controller → IoT Client → MQTT Dashboard

Run: python main.py [--demo] [--show] [--no-uart]
"""

from __future__ import annotations
import argparse, logging, logging.handlers, signal, sys, time, json
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

root_log = logging.getLogger()
root_log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
root_log.addHandler(sh)

fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "system.log", maxBytes=10 * 1024 * 1024, backupCount=3
)
fh.setFormatter(fmt)
root_log.addHandler(fh)

log = logging.getLogger("main")

# ── Local modules ─────────────────────────────────────────────────────────────
from camera            import CameraCapture
from ai_classifier     import WasteClassificationPipeline, CATEGORIES
from bin_controller    import BinController, ClassificationResult
from push_out_controller import PushOutController
from iot_client        import IoTClient
from config            import (
    TFLITE_MODEL, KMEANS_MODEL, YOLO_WEIGHTS,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    BELT_ROI, MOTION_MIN_AREA,
    CONF_THRESHOLD, SCAN_DEBOUNCE_MS,
    UART_PORT, UART_BAUD,
)

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  UART Handler (Raspberry Pi → ESP32)
# ══════════════════════════════════════════════════════════════════════════════

class UARTHandler:
    def __init__(self, port: str, baud: int):
        try:
            import serial  # type: ignore
            self._ser = serial.Serial(port, baud, timeout=0.5)
            self._ok  = True
            log.info("UART open: %s @ %d", port, baud)
        except Exception as e:
            log.warning("UART unavailable (%s) — hardware commands disabled", e)
            self._ser = None
            self._ok  = False

    def write(self, data: bytes):
        if self._ser and self._ok:
            self._ser.write(data)

    def readline(self) -> str:
        if self._ser and self._ok:
            try:
                return self._ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                pass
        return ""

    def close(self):
        if self._ser:
            self._ser.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════

class WasteBinApp:

    def __init__(self, args):
        self.args   = args
        self._alive = True
        self._last_classify_t: dict[int, float] = {}  # bbox_hash → timestamp

        # ── Components ──
        log.info("Initialising components…")

        self.uart = None if args.no_uart else UARTHandler(UART_PORT, UART_BAUD)

        self.camera = CameraCapture(
            device    = CAMERA_INDEX,
            width     = CAMERA_WIDTH,
            height    = CAMERA_HEIGHT,
            fps       = CAMERA_FPS,
            roi       = BELT_ROI,
        )

        self.pipeline = WasteClassificationPipeline(
            tflite_path  = TFLITE_MODEL,
            yolo_weights = YOLO_WEIGHTS,
        )

        self.push_ctrl = PushOutController(
            uart_handler   = self.uart,
            on_step_change = self._on_push_step,
            on_complete    = self._on_push_complete,
        )

        self.bin_ctrl = BinController(
            on_item_added  = self._on_item_added,
            on_bin_full    = self._on_bin_full,
            on_bag_ejected = self._on_bag_ejected,
        )

        self.iot = IoTClient()

        # Session stats
        self._frame_count = 0
        self._classify_count = 0
        self._start_time  = time.time()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_item_added(self, bin_state, result):
        self.iot.publish_detection({
            "cat_id":    result.cat_id,
            "category":  result.category,
            "short":     result.short,
            "confidence": result.confidence,
            "anomaly":   result.anomaly,
            "det_label": result.item_label,
        })
        log.info("✓ %s → [%s]  conf=%.0f%%  fill=%.0f%%",
                 result.item_label or result.category,
                 bin_state.short, result.confidence * 100,
                 bin_state.fill_pct * 100)

    def _on_bin_full(self, bin_state):
        log.warning("⚠ Bin [%s] FULL — triggering push-out", bin_state.short)
        self.iot.publish_alert("bin_full",
                               f"{bin_state.name} bin is full",
                               severity="warning")
        self.push_ctrl.trigger(bin_state.cat_id)

    def _on_bag_ejected(self, bin_state):
        log.info("✓ Bag ejected from [%s] — bag #%d", bin_state.short, bin_state.bag_count)

    def _on_push_step(self, cat_id, step, progress):
        if self.uart:
            pass  # UART command already sent by push_ctrl
        self.iot.publish_telemetry({
            "push_step":     step.name,
            "push_progress": round(progress, 2),
            "push_cat_id":   cat_id,
        })

    def _on_push_complete(self, cat_id):
        self.bin_ctrl.reset_bin(cat_id)

    # ── Debounce ──────────────────────────────────────────────────────────────
    def _is_debounced(self, bbox: tuple) -> bool:
        """Prevent classifying same item multiple times."""
        key = hash(bbox)
        now = time.time()
        last = self._last_classify_t.get(key, 0)
        if now - last < SCAN_DEBOUNCE_MS / 1000:
            return True
        self._last_classify_t[key] = now
        # Clean old entries
        if len(self._last_classify_t) > 200:
            oldest_key = min(self._last_classify_t, key=self._last_classify_t.get)
            del self._last_classify_t[oldest_key]
        return False

    # ── Demo mode (no camera) ─────────────────────────────────────────────────
    def _run_demo(self):
        """Simulate items arriving on belt every ~5 seconds."""
        import random
        DEMO_ITEMS = [
            ("Banana peel",   0), ("Cardboard box", 0), ("Apple core",  0),
            ("Plastic bottle",1), ("Glass jar",      1), ("Aluminium can",1),
            ("Battery",       2), ("Mobile phone",   2), ("Circuit board",2),
        ]
        log.info("Running in DEMO mode — items spawn every 5 seconds")
        log.info("Press Ctrl+C to stop\n")
        i = 0
        try:
            while self._alive:
                item_lbl, cat_id = DEMO_ITEMS[i % len(DEMO_ITEMS)]
                i += 1
                conf = 0.72 + random.random() * 0.25
                result = ClassificationResult(
                    cat_id    = cat_id,
                    category  = CATEGORIES[cat_id],
                    short     = ["BIO","REC","HAZ"][cat_id],
                    confidence= conf,
                    anomaly   = False,
                    item_label= item_lbl,
                )
                self.bin_ctrl.add_item(result)
                self._classify_count += 1
                self.bin_ctrl.print_status()
                time.sleep(5)
        except KeyboardInterrupt:
            pass

    # ── Live camera mode ──────────────────────────────────────────────────────
    def _run_live(self):
        log.info("Starting live camera classification")
        if not self.camera.open():
            log.error("Camera failed — falling back to demo mode")
            self._run_demo()
            return

        self.camera.start()
        log.info("Camera live — press Ctrl+C to stop\n")

        try:
            while self._alive:
                frame = self.camera.read(timeout=0.1)
                if frame is None:
                    continue

                self._frame_count += 1

                # Motion gate — skip empty frames
                if not self.camera.has_motion(frame, min_area=MOTION_MIN_AREA):
                    continue

                # Preprocess
                proc = self.camera.preprocess(frame)

                # AI inference
                results = self.pipeline.run(proc)

                for r in results:
                    if r["confidence"] < CONF_THRESHOLD:
                        continue
                    if r["cat_id"] < 0:
                        continue
                    if self._is_debounced(r["bbox"]):
                        continue

                    # Add to bin
                    cls_result = ClassificationResult(
                        cat_id    = r["cat_id"],
                        category  = r["category"],
                        short     = r["short"],
                        confidence= r["confidence"],
                        anomaly   = r.get("anomaly", False),
                        item_label= r.get("det_label", ""),
                    )
                    self.bin_ctrl.add_item(cls_result)
                    self._classify_count += 1

                    if r.get("anomaly"):
                        self.iot.publish_alert(
                            "anomaly",
                            f"Unknown material detected (conf={r['confidence']:.0%})",
                            severity="info"
                        )

                # Display
                if self.args.show:
                    annotated = self.pipeline.annotate(frame, results)
                    annotated = self.camera.draw_roi(annotated)
                    fps_txt   = f"FPS:{self.pipeline.avg_fps:.0f}  Classified:{self._classify_count}"
                    cv2.putText(annotated, fps_txt, (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 229, 255), 1, cv2.LINE_AA)
                    cv2.imshow("AI Waste Bin", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._alive = False

                # Periodic status print
                if self._frame_count % 300 == 0:
                    self.bin_ctrl.print_status()
                    fps = self._frame_count / (time.time() - self._start_time)
                    log.info("FPS: %.1f  |  Classified: %d", fps, self._classify_count)

        except KeyboardInterrupt:
            pass
        finally:
            self.camera.close()
            cv2.destroyAllWindows()

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        log.info("═══ AI Smart Waste Bin — System Start ═══")

        # Connect IoT
        self.iot.connect()
        self.iot.start_auto_telemetry(
            data_fn=lambda: {
                "fps":      round(self.pipeline.avg_fps, 1),
                "uptime_s": round(time.time() - self._start_time, 0),
                "classified": self._classify_count,
                **self.bin_ctrl.get_status(),
            }
        )

        # Graceful shutdown
        signal.signal(signal.SIGINT,  lambda s, f: self._shutdown())
        signal.signal(signal.SIGTERM, lambda s, f: self._shutdown())

        if self.args.demo:
            self._run_demo()
        else:
            self._run_live()

        self._shutdown()

    def _shutdown(self):
        log.info("Shutting down…")
        self._alive = False
        self.bin_ctrl.print_status()
        self.iot.publish_telemetry({
            "event":      "shutdown",
            "uptime_s":   round(time.time() - self._start_time, 0),
            "classified": self._classify_count,
        })
        time.sleep(0.5)
        self.iot.disconnect()
        if self.uart:
            self.uart.write(b"BELT:STOP\n")
            self.uart.close()
        log.info("Clean shutdown complete.")
        sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI Smart Waste Bin — Main Controller")
    ap.add_argument("--demo",     action="store_true",
                    help="Run without camera (simulate items)")
    ap.add_argument("--show",     action="store_true",
                    help="Show live annotated video window")
    ap.add_argument("--no-uart",  dest="no_uart", action="store_true",
                    help="Disable ESP32 UART (software only)")
    ap.add_argument("--retrain",  action="store_true",
                    help="Retrain classifier before starting")
    args = ap.parse_args()

    if args.retrain:
        log.info("Retraining classifier first…")
        import subprocess
        ret = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "model" / "train_clustering.py"),
             "--synthetic"],
            capture_output=True, text=True
        )
        print(ret.stdout)
        if ret.returncode != 0:
            log.error("Training failed:\n%s", ret.stderr)

    app = WasteBinApp(args)
    app.run()
