"""
bin_controller.py — Bin Compartment Logic & Fill Tracking
===========================================================
Manages the three waste bins (BIO / REC / HAZ):
  • Fill level tracking from ultrasonic sensors (or weight)
  • Triggers push-out sequence at configurable threshold
  • Maintains per-session and all-time item counts
  • Publishes bin state changes via callback
"""

from __future__ import annotations
import time, logging, threading
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger("bin_controller")

try:
    from config import (
        BIN_FULL_THRESHOLD, BIN_EMPTY_CM, CATEGORY_NAMES, CATEGORY_SHORT
    )
except ImportError:
    BIN_FULL_THRESHOLD = 0.90
    BIN_EMPTY_CM       = 40.0
    CATEGORY_NAMES     = {0: "Biodegradable", 1: "Recyclable", 2: "Hazardous"}
    CATEGORY_SHORT     = {0: "BIO", 1: "REC", 2: "HAZ"}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BinState:
    cat_id:    int
    name:      str
    short:     str
    count:     int   = 0
    fill_pct:  float = 0.0      # 0.0–1.0
    fill_cm:   float = 0.0      # ultrasonic reading
    weight_g:  float = 0.0      # load cell reading
    is_full:   bool  = False
    last_item: str   = ""
    last_ts:   float = field(default_factory=time.time)
    bag_count: int   = 0        # bags ejected for this bin this session

    def to_dict(self) -> dict:
        return {
            "cat_id":    self.cat_id,
            "name":      self.name,
            "short":     self.short,
            "count":     self.count,
            "fill_pct":  round(self.fill_pct * 100, 1),
            "fill_cm":   round(self.fill_cm, 1),
            "weight_g":  round(self.weight_g, 1),
            "is_full":   self.is_full,
            "last_item": self.last_item,
            "bag_count": self.bag_count,
        }


@dataclass
class ClassificationResult:
    cat_id:     int
    category:   str
    short:      str
    confidence: float
    anomaly:    bool
    item_label: str = ""
    bbox:       tuple = ()
    timestamp:  float = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════════════════
#  BinController
# ══════════════════════════════════════════════════════════════════════════════

class BinController:
    """
    Central bin management controller.

    Callbacks
    ---------
    on_item_added(bin_state, result)  — called when item classified & routed
    on_bin_full(bin_state)             — called when fill ≥ threshold
    on_bag_ejected(bin_state)          — called after bag push-out completes
    """

    def __init__(
        self,
        full_threshold: float = BIN_FULL_THRESHOLD,
        on_item_added:  Optional[Callable] = None,
        on_bin_full:    Optional[Callable] = None,
        on_bag_ejected: Optional[Callable] = None,
    ):
        self.full_threshold  = full_threshold
        self.on_item_added   = on_item_added
        self.on_bin_full     = on_bin_full
        self.on_bag_ejected  = on_bag_ejected

        # Initialise 3 bins
        self.bins: dict[int, BinState] = {
            cat_id: BinState(
                cat_id=cat_id,
                name=CATEGORY_NAMES[cat_id],
                short=CATEGORY_SHORT[cat_id],
            )
            for cat_id in range(3)
        }

        # Session totals
        self.total_items     = 0
        self.session_start   = time.time()
        self._lock           = threading.Lock()

    # ── Update from sensor reading ────────────────────────────────────────────
    def update_fill_sensor(self, cat_id: int, distance_cm: float):
        """
        Update fill level from HC-SR04 ultrasonic reading.
        distance_cm is the distance from sensor to waste surface.
        Smaller distance = more waste = higher fill.
        """
        if cat_id not in self.bins:
            return
        with self._lock:
            b = self.bins[cat_id]
            b.fill_cm  = distance_cm
            b.fill_pct = max(0.0, min(1.0, 1.0 - (distance_cm / BIN_EMPTY_CM)))
            was_full   = b.is_full
            b.is_full  = b.fill_pct >= self.full_threshold
            if b.is_full and not was_full:
                log.warning("Bin %s FULL (%.0f%%)", b.short, b.fill_pct * 100)
                if self.on_bin_full:
                    self.on_bin_full(b)

    def update_weight_sensor(self, cat_id: int, weight_g: float):
        """Update weight reading from HX711 load cell."""
        if cat_id not in self.bins:
            return
        with self._lock:
            self.bins[cat_id].weight_g = weight_g

    # ── Register classified item ──────────────────────────────────────────────
    def add_item(self, result: ClassificationResult, fill_increment: float = 0.03):
        """
        Register a newly classified waste item landing in its bin.

        fill_increment : fractional fill increase per item
                         (override with ultrasonic reading for accuracy)
        """
        cat_id = result.cat_id
        if cat_id not in self.bins:
            log.error("Unknown cat_id: %d", cat_id)
            return

        with self._lock:
            b          = self.bins[cat_id]
            b.count   += 1
            b.last_item = result.item_label
            b.last_ts   = result.timestamp
            self.total_items += 1

            # Increment fill if sensor data not available
            if b.fill_cm == 0.0:
                b.fill_pct = min(1.0, b.fill_pct + fill_increment)

            was_full = b.is_full
            b.is_full = b.fill_pct >= self.full_threshold

            log.info("[%s] Item #%d added: %s (conf=%.2f%%)  fill=%.0f%%",
                     b.short, b.count, result.item_label or result.category,
                     result.confidence * 100, b.fill_pct * 100)

            if self.on_item_added:
                self.on_item_added(b, result)

            if b.is_full and not was_full:
                log.warning("Bin %s FULL — triggering push-out", b.short)
                if self.on_bin_full:
                    self.on_bin_full(b)

    # ── Register bag ejection ─────────────────────────────────────────────────
    def reset_bin(self, cat_id: int):
        """Called after successful bag ejection — resets fill level."""
        if cat_id not in self.bins:
            return
        with self._lock:
            b          = self.bins[cat_id]
            b.fill_pct = 0.0
            b.fill_cm  = BIN_EMPTY_CM
            b.is_full  = False
            b.bag_count += 1
            log.info("Bin %s reset after ejection (bag #%d)", b.short, b.bag_count)
            if self.on_bag_ejected:
                self.on_bag_ejected(b)

    # ── Status report ─────────────────────────────────────────────────────────
    def get_status(self) -> dict:
        elapsed = time.time() - self.session_start
        return {
            "total_items":    self.total_items,
            "session_mins":   round(elapsed / 60, 1),
            "bins":           {b.short: b.to_dict() for b in self.bins.values()},
        }

    def get_full_bins(self) -> list[BinState]:
        """Returns list of bins that are full and need push-out."""
        with self._lock:
            return [b for b in self.bins.values() if b.is_full]

    def print_status(self):
        st = self.get_status()
        print(f"\n{'═'*50}")
        print(f"  Session: {st['session_mins']:.1f} min  |  Total items: {st['total_items']}")
        print(f"{'─'*50}")
        for short, b in st["bins"].items():
            bar_n  = int(b["fill_pct"] / 5)
            bar    = "█" * bar_n + "░" * (20 - bar_n)
            full   = " ⚠ FULL" if b["is_full"] else ""
            print(f"  [{short}] {bar} {b['fill_pct']:5.1f}%  {b['count']:3d} items{full}")
        print(f"{'═'*50}\n")
