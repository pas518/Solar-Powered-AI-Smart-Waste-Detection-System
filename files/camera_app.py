"""
Waste Management Camera Application
Real-time camera feed with YOLOv8 waste detection overlay.
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from collections import deque
from waste_detector import WasteDetector, WASTE_CATEGORIES


# ─── HUD / Overlay Configuration ─────────────────────────────────────────────
HUD_BG_COLOR = (15, 15, 25)          # Dark panel background
HUD_TEXT_COLOR = (220, 220, 230)     # Light text
HUD_WIDTH = 280                       # Right-side panel width
FPS_BUFFER = deque(maxlen=30)        # Rolling FPS window


def draw_hud(frame: np.ndarray, counts: dict, fps: float,
             frame_counts: dict, total_frames: int) -> np.ndarray:
    """
    Draws a right-side HUD panel showing:
    - Live FPS
    - Per-category counts (frame & session)
    - Color-coded category bars
    """
    h, w = frame.shape[:2]
    panel_x = w - HUD_WIDTH

    # Semi-transparent panel overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    y = 20
    # ── Title ──
    cv2.putText(frame, "WASTE DETECTOR", (panel_x + 10, y + 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 220, 255), 1, cv2.LINE_AA)
    y += 32
    cv2.line(frame, (panel_x + 8, y), (w - 8, y), (60, 60, 80), 1)
    y += 14

    # ── FPS ──
    fps_color = (80, 240, 80) if fps >= 20 else (240, 180, 40) if fps >= 10 else (240, 60, 60)
    cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 10, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 1, cv2.LINE_AA)
    y += 28

    # ── Frame counter ──
    cv2.putText(frame, f"Frames: {total_frames}", (panel_x + 10, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, HUD_TEXT_COLOR, 1, cv2.LINE_AA)
    y += 28
    cv2.line(frame, (panel_x + 8, y), (w - 8, y), (60, 60, 80), 1)
    y += 14

    # ── Category counts ──
    cv2.putText(frame, "THIS FRAME", (panel_x + 10, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 170), 1, cv2.LINE_AA)
    y += 22

    categories = [
        ("biodegradable", "BIO"),
        ("recyclable",    "REC"),
        ("hazardous",     "HAZ"),
    ]

    for cat, short in categories:
        color = WASTE_CATEGORIES[cat]["color"]
        cnt = frame_counts.get(cat, 0)

        # Colored dot
        cv2.circle(frame, (panel_x + 18, y + 8), 6, color, -1)

        # Category label
        cv2.putText(frame, f"{cat.capitalize()}", (panel_x + 32, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, HUD_TEXT_COLOR, 1, cv2.LINE_AA)

        # Count badge
        badge_txt = str(cnt)
        (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bx = w - 30 - bw
        cv2.rectangle(frame, (bx - 4, y - 2), (bx + bw + 4, y + bh + 4), color, -1)
        cv2.putText(frame, badge_txt, (bx, y + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30

    y += 4
    cv2.line(frame, (panel_x + 8, y), (w - 8, y), (60, 60, 80), 1)
    y += 14

    # ── Session totals ──
    cv2.putText(frame, "SESSION TOTALS", (panel_x + 10, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 170), 1, cv2.LINE_AA)
    y += 22

    total = sum(counts.values())
    for cat, short in categories:
        color = WASTE_CATEGORIES[cat]["color"]
        cnt = counts.get(cat, 0)
        pct = (cnt / total * 100) if total > 0 else 0

        cv2.circle(frame, (panel_x + 18, y + 8), 6, color, -1)
        cv2.putText(frame, f"{cat.capitalize()}: {cnt}", (panel_x + 32, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, HUD_TEXT_COLOR, 1, cv2.LINE_AA)

        # Mini progress bar
        bar_w = HUD_WIDTH - 50
        bar_x = panel_x + 25
        bar_y = y + 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 5), (50, 50, 60), -1)
        filled = int(bar_w * pct / 100)
        if filled > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 5), color, -1)
        y += 34

    y += 4
    cv2.line(frame, (panel_x + 8, y), (w - 8, y), (60, 60, 80), 1)
    y += 14

    # ── Total ──
    cv2.putText(frame, f"TOTAL ITEMS: {total}", (panel_x + 10, y + 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 220, 80), 1, cv2.LINE_AA)
    y += 34

    # ── Hotkeys ──
    cv2.line(frame, (panel_x + 8, y), (w - 8, y), (60, 60, 80), 1)
    y += 12
    hotkeys = [("[Q]", "Quit"), ("[S]", "Screenshot"), ("[R]", "Reset")]
    for key, action in hotkeys:
        cv2.putText(frame, f"{key} {action}", (panel_x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 140), 1, cv2.LINE_AA)
        y += 18

    return frame


def draw_top_bar(frame: np.ndarray, timestamp: str) -> np.ndarray:
    """Draw a slim top status bar."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - HUD_WIDTH, 28), (10, 10, 20), -1)
    cv2.putText(frame, f"AI Waste Management System  |  {timestamp}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 180, 255), 1, cv2.LINE_AA)
    return frame


def run_camera(camera_index: int = 0,
               model_path: str = "yolov8n.pt",
               confidence: float = 0.40,
               save_screenshots: bool = True,
               screenshot_dir: str = "screenshots"):
    """
    Main loop: opens camera, runs detection, displays annotated feed.
    
    Keys:
        Q  — quit
        S  — save screenshot
        R  — reset session counts
    """
    os.makedirs(screenshot_dir, exist_ok=True)

    print("=" * 60)
    print("  WASTE MANAGEMENT AI  —  Starting up...")
    print("=" * 60)

    detector = WasteDetector(model_path=model_path, confidence=confidence)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Camera] ❌ Cannot open camera index {camera_index}")
        print("         Try changing camera_index (0, 1, 2...)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"[Camera] ✅ Camera {camera_index} opened.")
    print("[Camera] Press Q to quit, S for screenshot, R to reset counts.\n")

    frame_idx = 0
    current_frame_counts = {}

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("[Camera] ❌ Frame read failed. Exiting.")
            break

        # Detection
        annotated, detections, frame_counts = detector.detect(frame)
        detector.update_session_counts(frame_counts)
        current_frame_counts = frame_counts

        # FPS
        elapsed = time.perf_counter() - t0
        FPS_BUFFER.append(1.0 / max(elapsed, 1e-6))
        fps = sum(FPS_BUFFER) / len(FPS_BUFFER)

        frame_idx += 1
        timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

        # Draw HUD
        session = detector.get_session_summary()
        annotated = draw_hud(annotated, session, fps, current_frame_counts, frame_idx)
        annotated = draw_top_bar(annotated, timestamp)

        cv2.imshow("AI Waste Management System", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n[Camera] Shutting down...")
            break
        elif key == ord('s'):
            path = os.path.join(screenshot_dir, f"waste_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(path, annotated)
            print(f"[Camera] 📸 Screenshot saved: {path}")
        elif key == ord('r'):
            detector.reset_session()
            print("[Camera] 🔄 Session counts reset.")

    cap.release()
    cv2.destroyAllWindows()

    # Save session report
    summary = detector.get_session_summary()
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Biodegradable : {summary['biodegradable']}")
    print(f"  Recyclable    : {summary['recyclable']}")
    print(f"  Hazardous     : {summary['hazardous']}")
    print(f"  TOTAL         : {summary['total']}")
    print("=" * 60)

    report = {
        "session_end": datetime.now().isoformat(),
        "total_frames": frame_idx,
        "summary": summary,
    }
    report_path = os.path.join(screenshot_dir, f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved : {report_path}\n")


if __name__ == "__main__":
    run_camera(
        camera_index=0,          # 0 = default webcam; change if needed
        model_path="yolov8n.pt", # auto-downloads on first run
        confidence=0.40,
        save_screenshots=True,
    )
