"""
stepper.py — NEMA17 Stepper Motor Controller (MicroPython)
===========================================================
Drives the conveyor belt NEMA17 via A4988 or DRV8825 driver.
Supports microstepping, acceleration ramp, and async step generation.
"""

from machine import Pin
import uasyncio as asyncio
import time


class StepperMotor:
    """
    NEMA17 stepper driver for A4988 / DRV8825.

    Parameters
    ----------
    step, dir, en : GPIO pin numbers
    steps_rev     : full steps per revolution (200 for 1.8° motor)
    microstep     : microstepping factor (1/2/4/8/16/32)
    """

    def __init__(
        self,
        step: int, dir: int, en: int,
        steps_rev: int = 200,
        microstep: int = 8,
    ):
        self._step_pin = Pin(step, Pin.OUT, value=0)
        self._dir_pin  = Pin(dir,  Pin.OUT, value=0)
        self._en_pin   = Pin(en,   Pin.OUT, value=1)  # HIGH = disabled on A4988

        self.steps_rev  = steps_rev
        self.microstep  = microstep
        self.total_steps = steps_rev * microstep   # micro-steps per revolution

        # Speed state
        self._period_us  = 2000   # microseconds between steps (default ~0.5 RPS)
        self._running    = False
        self._task       = None

    # ── Enable / disable ──────────────────────────────────────────────────────
    def enable(self):
        """Enable driver (LOW on ENABLE pin)."""
        self._en_pin.value(0)

    def disable(self):
        """Disable driver — coils de-energised (saves power when belt idle)."""
        self._en_pin.value(1)

    # ── Speed ──────────────────────────────────────────────────────────────────
    def set_speed(self, rps: float):
        """
        Set belt speed in revolutions per second.
        rps=1.0 ≈ normal belt speed.  rps=0.3 = camera slow zone.
        """
        if rps <= 0:
            self.stop()
            return
        steps_per_sec    = self.total_steps * rps
        self._period_us  = max(100, int(1_000_000 / steps_per_sec))

    def set_rpm(self, rpm: float):
        self.set_speed(rpm / 60.0)

    # ── Direction ─────────────────────────────────────────────────────────────
    def set_direction(self, forward: bool = True):
        self._dir_pin.value(0 if forward else 1)

    # ── Step generation ───────────────────────────────────────────────────────
    def step_once(self):
        """Generate one step pulse."""
        self._step_pin.value(1)
        time.sleep_us(2)
        self._step_pin.value(0)
        time.sleep_us(2)

    def step_n(self, n: int, delay_us: int | None = None):
        """Generate N steps synchronously (blocks)."""
        d = delay_us or self._period_us
        for _ in range(n):
            self._step_pin.value(1)
            time.sleep_us(2)
            self._step_pin.value(0)
            time.sleep_us(d - 2)

    def rotate_deg(self, degrees: float):
        """Rotate by exactly N degrees."""
        steps = int(abs(degrees) / 360.0 * self.total_steps)
        self.set_direction(degrees > 0)
        self.step_n(steps)

    # ── Continuous async run ──────────────────────────────────────────────────
    async def _run_loop(self):
        while self._running:
            self._step_pin.value(1)
            await asyncio.sleep_us(2)
            self._step_pin.value(0)
            await asyncio.sleep_us(self._period_us - 2)

    def start(self):
        """Start continuous belt movement (async)."""
        self.enable()
        self.set_direction(True)
        self._running = True
        self._task    = asyncio.ensure_future(self._run_loop())

    def stop(self):
        """Stop belt movement."""
        self._running = False
        self._step_pin.value(0)

    # ── Acceleration ramp ─────────────────────────────────────────────────────
    def ramp_to(self, target_rps: float, ramp_time_s: float = 0.5, steps: int = 20):
        """
        Linearly ramp speed from current to target_rps over ramp_time_s seconds.
        Call start() before and stop() after if needed.
        """
        start_period = self._period_us
        end_rps      = max(0.01, target_rps)
        end_period   = max(100, int(1_000_000 / (self.total_steps * end_rps)))
        step_delay   = ramp_time_s / steps

        for i in range(steps + 1):
            t = i / steps
            self._period_us = int(start_period + (end_period - start_period) * t)
            time.sleep(step_delay)
