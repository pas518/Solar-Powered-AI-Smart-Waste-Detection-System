"""
camera.py — Camera Capture & Preprocessing Module
===================================================
Handles RPi CSI / USB camera capture with:
  • Auto-exposure warmup
  • Belt region-of-interest (ROI) extraction
  • Adaptive pre-processing for consistent classification
  • Frame buffer with drop-frame protection at 30 FPS
  • Motion detection gate (only classify when item is in FOV)
"""

from __future__ import annotations
import threading, time, logging, queue
from pathlib import Path
import cv2
import numpy as np

log = logging.getLogger("camera")

# ── Default belt ROI (percentage of frame) ────────────────────────────────────
# Adjust in config.py to match physical camera mounting position
DEFAULT_ROI = (0.10, 0.20, 0.90, 0.80)  # (x1%, y1%, x2%, y2%)


class CameraCapture:
    """
    Thread-safe camera capture with preprocessing pipeline.

    Parameters
    ----------
    device      : int or str — camera index or GStreamer pipeline string
    width, height, fps : capture resolution & rate
    roi         : (x1%, y1%, x2%, y2%) region of interest on belt
    warmup_secs : seconds to allow auto-exposure to settle
    buffer_size : max frames in internal queue
    """

    def __init__(
        self,
        device=0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        roi: tuple = DEFAULT_ROI,
        warmup_secs: float = 2.0,
        buffer_size: int = 3,
        use_gstreamer: bool = False,
    ):
        self.device       = device
        self.width        = width
        self.height       = height
        self.fps          = fps
        self.roi          = roi
        self.warmup_secs  = warmup_secs
        self.buffer_size  = buffer_size
        self.use_gstreamer = use_gstreamer

        self._cap: cv2.VideoCapture | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._running  = False
        self._thread   = None
        self._frame_id = 0
        self._bg_model = None  # MOG2 background subtractor for motion gate

        self._stats = {"captured": 0, "dropped": 0, "errors": 0}

    # ── Open / close ──────────────────────────────────────────────────────────
    def open(self) -> bool:
        pipeline = self._build_pipeline() if self.use_gstreamer else self.device
        self._cap = cv2.VideoCapture(pipeline)

        if not self._cap.isOpened():
            log.error("Cannot open camera: %s", self.device)
            return False

        if not self.use_gstreamer:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
            # Enable auto-exposure, auto-white-balance
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_f = self._cap.get(cv2.CAP_PROP_FPS)
        log.info("Camera opened: %dx%d @ %.1f fps (requested %dx%d @ %d)",
                 actual_w, actual_h, actual_f, self.width, self.height, self.fps)

        # Warmup: discard frames while auto-exposure settles
        log.info("Warming up for %.1f seconds…", self.warmup_secs)
        t_end = time.time() + self.warmup_secs
        while time.time() < t_end:
            self._cap.read()

        # Background subtractor
        self._bg_model = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )
        return True

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        log.info("Camera closed. Stats: %s", self._stats)

    # ── GStreamer pipeline for RPi CSI ────────────────────────────────────────
    def _build_pipeline(self) -> str:
        return (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM),width={self.width},height={self.height},"
            f"format=NV12,framerate={self.fps}/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink"
        )

    # ── Background capture thread ─────────────────────────────────────────────
    def start(self):
        if not self._cap or not self._cap.isOpened():
            if not self.open():
                return
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info("Capture thread started")

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self._stats["errors"] += 1
                time.sleep(0.01)
                continue

            self._frame_id       += 1
            self._stats["captured"] += 1

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                # Drop oldest, put newest
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(frame)
                self._stats["dropped"] += 1

    # ── Read ──────────────────────────────────────────────────────────────────
    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        """Get the latest frame from queue. Returns None if not available."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_blocking(self) -> np.ndarray:
        """Block until a frame is available."""
        while True:
            frame = self.read(timeout=0.2)
            if frame is not None:
                return frame

    # ── Motion gate ───────────────────────────────────────────────────────────
    def has_motion(self, frame: np.ndarray, min_area: int = 3000) -> bool:
        """
        Returns True if a significant foreground object is in the belt ROI.
        Used to avoid classifying empty belt frames.
        """
        roi_frame = self.extract_roi(frame)
        fg_mask   = self._bg_model.apply(roi_frame)
        # Remove noise with morphological opening
        kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean     = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_area   = cv2.countNonZero(clean)
        return fg_area > min_area

    # ── ROI extraction ────────────────────────────────────────────────────────
    def extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop to the belt region of interest."""
        h, w = frame.shape[:2]
        x1 = int(self.roi[0] * w)
        y1 = int(self.roi[1] * h)
        x2 = int(self.roi[2] * w)
        y2 = int(self.roi[3] * h)
        return frame[y1:y2, x1:x2]

    # ── Preprocessing pipeline ────────────────────────────────────────────────
    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """
        Normalise illumination and enhance contrast for consistent AI input.

        Steps:
          1. CLAHE on L channel (adaptive histogram equalisation)
          2. Gaussian denoising
          3. Sharpening unsharp mask
        """
        # Convert to LAB for CLAHE on luminance only
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq  = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        out   = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Gentle Gaussian denoise
        out = cv2.GaussianBlur(out, (3, 3), 0)

        # Unsharp mask for edge enhancement
        blur  = cv2.GaussianBlur(out, (5, 5), 1.0)
        out   = cv2.addWeighted(out, 1.5, blur, -0.5, 0)

        return out

    # ── Draw ROI overlay ──────────────────────────────────────────────────────
    def draw_roi(self, frame: np.ndarray, colour=(0, 229, 255)) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = int(self.roi[0] * w); y1 = int(self.roi[1] * h)
        x2 = int(self.roi[2] * w); y2 = int(self.roi[3] * h)
        out = frame.copy()
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, "BELT ROI", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Snapshot utility
# ══════════════════════════════════════════════════════════════════════════════

def capture_snapshot(
    output_dir: str = "captures",
    device: int = 0,
    n: int = 1,
    label: str = "unknown"
) -> list[str]:
    """
    Capture N snapshots from the camera and save to output_dir/label/.
    Used for building labelled training datasets.
    """
    out_path = Path(output_dir) / label
    out_path.mkdir(parents=True, exist_ok=True)

    cam = CameraCapture(device=device, warmup_secs=1.5)
    cam.open()

    saved = []
    for i in range(n):
        frame = cam.read_blocking()
        ts    = int(time.time() * 1000)
        fp    = out_path / f"{label}_{ts}_{i:03d}.jpg"
        cv2.imwrite(str(fp), frame)
        saved.append(str(fp))
        log.info("Saved: %s", fp)
        time.sleep(0.2)

    cam.close()
    return saved


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Camera test & snapshot tool")
    ap.add_argument("--device",  default=0, type=int)
    ap.add_argument("--width",   default=1280, type=int)
    ap.add_argument("--height",  default=720,  type=int)
    ap.add_argument("--fps",     default=30,   type=int)
    ap.add_argument("--show",    action="store_true")
    ap.add_argument("--snap",    default=0, type=int,
                    help="Capture N snapshots then exit")
    ap.add_argument("--label",   default="unknown")
    args = ap.parse_args()

    if args.snap > 0:
        paths = capture_snapshot(n=args.snap, label=args.label, device=args.device)
        print(f"Saved {len(paths)} snapshots")
    else:
        cam = CameraCapture(
            device=args.device, width=args.width, height=args.height, fps=args.fps
        )
        cam.start()
        log.info("Streaming — press Q to quit")
        try:
            while True:
                frame = cam.read()
                if frame is None:
                    continue
                proc = cam.preprocess(frame)
                roi  = cam.extract_roi(proc)
                disp = cam.draw_roi(frame)
                motion = cam.has_motion(frame)
                cv2.putText(disp, f"Motion: {motion}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if motion else (128, 128, 128), 1)
                if args.show:
                    cv2.imshow("Camera", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cam.close()
            cv2.destroyAllWindows()
