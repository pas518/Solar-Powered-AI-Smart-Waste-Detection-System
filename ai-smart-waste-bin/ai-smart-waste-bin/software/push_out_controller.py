"""
push_out_controller.py — Bag Push-Out & New Bag Load Controller
=================================================================
Executes the 4-step bag ejection sequence when a bin reaches 90%:

  Step 1 — EXTEND  : Pneumatic piston extends, pushes filled bag outward
  Step 2 — HOLD    : Piston holds position while heat seal activates
  Step 3 — SEAL    : Heat seal element activates (500 ms)
  Step 4 — RETRACT : Piston retracts, bag loader advances new bag

Hardware path (ESP32 via UART commands):
  Pi → UART → ESP32 → GPIO → Relay/H-bridge → Piston solenoid valve

Simulation path (no hardware):
  State machine with configurable timings, callbacks fire on each step
"""

from __future__ import annotations
import time, logging, threading
from enum import Enum, auto
from typing import Callable, Optional

log = logging.getLogger("push_out")

try:
    from config import (
        PISTON_EXTEND_MS, PISTON_HOLD_MS,
        PISTON_SEAL_MS, PISTON_RETRACT_MS,
        BAGLOAD_STEPS_PER_BAG,
    )
except ImportError:
    PISTON_EXTEND_MS   = 900
    PISTON_HOLD_MS     = 600
    PISTON_SEAL_MS     = 500
    PISTON_RETRACT_MS  = 750
    BAGLOAD_STEPS_PER_BAG = 400


class PushStep(Enum):
    IDLE    = auto()
    EXTEND  = auto()
    HOLD    = auto()
    SEAL    = auto()
    RETRACT = auto()
    DONE    = auto()


STEP_DURATIONS = {
    PushStep.EXTEND:  PISTON_EXTEND_MS  / 1000,
    PushStep.HOLD:    PISTON_HOLD_MS    / 1000,
    PushStep.SEAL:    PISTON_SEAL_MS    / 1000,
    PushStep.RETRACT: PISTON_RETRACT_MS / 1000,
}

STEP_ORDER = [
    PushStep.EXTEND,
    PushStep.HOLD,
    PushStep.SEAL,
    PushStep.RETRACT,
    PushStep.DONE,
]


class PushOutController:
    """
    Non-blocking push-out state machine.

    Callbacks
    ---------
    on_step_change(cat_id, step, progress)  — fires on each state transition
    on_complete(cat_id)                      — fires when ejection completes
    on_error(cat_id, message)               — fires on fault detection
    """

    def __init__(
        self,
        uart_handler=None,
        on_step_change: Optional[Callable] = None,
        on_complete:    Optional[Callable] = None,
        on_error:       Optional[Callable] = None,
    ):
        self.uart            = uart_handler
        self.on_step_change  = on_step_change
        self.on_complete     = on_complete
        self.on_error        = on_error

        self._step      = PushStep.IDLE
        self._cat_id    = -1
        self._t_step    = 0.0
        self._thread    = None
        self._running   = False
        self._eject_cnt = 0
        self._lock      = threading.Lock()

    # ── Trigger ────────────────────────────────────────────────────────────────
    def trigger(self, cat_id: int) -> bool:
        """
        Start bag ejection for the given bin.
        Returns False if a sequence is already in progress.
        """
        with self._lock:
            if self._step != PushStep.IDLE:
                log.warning("Push-out busy (%s) — cannot start for cat %d",
                            self._step.name, cat_id)
                return False
            self._cat_id = cat_id
            self._step   = PushStep.EXTEND

        log.info("Push-out TRIGGERED for cat %d — starting sequence", cat_id)
        self._thread = threading.Thread(target=self._run_sequence, daemon=True)
        self._thread.start()
        return True

    # ── State machine ─────────────────────────────────────────────────────────
    def _run_sequence(self):
        self._running = True
        cat_id        = self._cat_id
        total_steps   = len(STEP_ORDER) - 1  # exclude DONE from count

        for i, step in enumerate(STEP_ORDER):
            if step == PushStep.DONE:
                break
            with self._lock:
                self._step = step

            duration = STEP_DURATIONS[step]
            progress = (i + 1) / total_steps

            log.info("[PUSH] Step %d/%d: %s  (%.0f ms)",
                     i + 1, total_steps, step.name, duration * 1000)

            # Send hardware command
            self._send_command(step, cat_id)

            if self.on_step_change:
                self.on_step_change(cat_id, step, progress)

            time.sleep(duration)

        # Sequence complete
        with self._lock:
            self._step = PushStep.DONE
            self._eject_cnt += 1
            eject_n = self._eject_cnt

        log.info("[PUSH] Sequence COMPLETE — ejection #%d", eject_n)

        # Advance new bag
        self._advance_bag()

        if self.on_complete:
            self.on_complete(cat_id)

        # Back to idle
        with self._lock:
            self._step   = PushStep.IDLE
            self._cat_id = -1

        self._running = False

    # ── Hardware commands (via UART to ESP32) ─────────────────────────────────
    def _send_command(self, step: PushStep, cat_id: int):
        if self.uart is None:
            return
        cmd_map = {
            PushStep.EXTEND:  f"PISTON:EXTEND:{cat_id}\n",
            PushStep.HOLD:    f"PISTON:HOLD:{cat_id}\n",
            PushStep.SEAL:    f"SEAL:ON:{cat_id}\n",
            PushStep.RETRACT: f"PISTON:RETRACT:{cat_id}\n",
        }
        cmd = cmd_map.get(step)
        if cmd:
            try:
                self.uart.write(cmd.encode())
                log.debug("UART → %s", cmd.strip())
            except Exception as e:
                log.error("UART write failed: %s", e)
                if self.on_error:
                    self.on_error(cat_id, str(e))

    def _advance_bag(self):
        """Send command to advance the bag roll."""
        if self.uart is None:
            return
        cmd = f"BAGLOAD:ADVANCE:{BAGLOAD_STEPS_PER_BAG}\n"
        try:
            self.uart.write(cmd.encode())
            log.info("New bag advanced (%d steps)", BAGLOAD_STEPS_PER_BAG)
        except Exception as e:
            log.error("Bag advance failed: %s", e)

    # ── Status ─────────────────────────────────────────────────────────────────
    @property
    def is_busy(self) -> bool:
        return self._step != PushStep.IDLE

    @property
    def current_step(self) -> PushStep:
        return self._step

    @property
    def step_number(self) -> int:
        try:
            return STEP_ORDER.index(self._step)
        except ValueError:
            return 0

    def get_status(self) -> dict:
        return {
            "step":       self._step.name,
            "step_num":   self.step_number,
            "busy":       self.is_busy,
            "cat_id":     self._cat_id,
            "ejections":  self._eject_cnt,
        }
