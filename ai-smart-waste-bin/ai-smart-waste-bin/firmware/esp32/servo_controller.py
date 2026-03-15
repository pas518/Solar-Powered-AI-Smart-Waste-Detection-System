"""
servo_controller.py — Servo & Actuator Controller (MicroPython)
================================================================
Controls all servo-driven actuators:
  • 3-way diverter servo (SG90 / MG996R)
  • Entry flap servo
  • Bag clip servo
  • Pneumatic piston relay outputs
  • Heat seal relay
"""

from machine import Pin, PWM
import time


class Servo:
    """Generic PWM servo. 50Hz, 500–2500µs pulse width."""

    def __init__(self, pin: int, min_us: int = 500, max_us: int = 2500, freq: int = 50):
        self._pwm    = PWM(Pin(pin), freq=freq)
        self._min_us = min_us
        self._max_us = max_us
        self._angle  = 90
        self._freq   = freq

    def _angle_to_duty(self, angle: float) -> int:
        # duty = us / period_us * 65535
        period_us = 1_000_000 / self._freq
        us = self._min_us + (angle / 180.0) * (self._max_us - self._min_us)
        return int(us / period_us * 65535)

    def set_angle(self, angle: float, smooth: bool = False, steps: int = 20):
        angle = max(0.0, min(180.0, angle))
        if smooth and abs(angle - self._angle) > 3:
            start = self._angle
            for i in range(1, steps + 1):
                mid = start + (angle - start) * (i / steps)
                self._pwm.duty_u16(self._angle_to_duty(mid))
                time.sleep_ms(12)
        else:
            self._pwm.duty_u16(self._angle_to_duty(angle))
        self._angle = angle

    def off(self):
        """Cut PWM signal (save power when servo isn't moving)."""
        self._pwm.duty_u16(0)

    @property
    def angle(self) -> float:
        return self._angle


class ServoController:
    """
    High-level controller for all servo-driven mechanisms.

    Pin assignments from config:
      Diverter : GPIO 13
      Flap     : GPIO 12
      Clip     : GPIO 25
    Piston relays:
      Extend   : GPIO 33
      Retract  : GPIO 32
      Seal     : GPIO 35
    """

    # Servo angle presets
    DIVERTER_ANGLES = {"BIO": 45, "REC": 90, "HAZ": 135, "NEUTRAL": 90}
    FLAP_OPEN       = 80
    FLAP_CLOSED     = 10
    CLIP_OPEN       = 90
    CLIP_CLOSED     = 0

    def __init__(self):
        # Servos
        self.diverter = Servo(pin=13)
        self.flap     = Servo(pin=12)
        self.clip     = Servo(pin=25)

        # Piston relay pins (active-HIGH relay board)
        self._piston_extend  = Pin(33, Pin.OUT, value=0)
        self._piston_retract = Pin(32, Pin.OUT, value=0)
        self._heat_seal      = Pin(35, Pin.OUT, value=0)

        # State
        self._diverter_pos   = "NEUTRAL"

    # ── Diverter ──────────────────────────────────────────────────────────────
    def set_diverter(self, angle: int | None = None, position: str | None = None):
        """
        Route waste to a bin.
          position : "BIO" | "REC" | "HAZ" | "NEUTRAL"
          angle    : direct servo angle (overrides position)
        """
        if position:
            angle = self.DIVERTER_ANGLES.get(position.upper(), 90)
            self._diverter_pos = position.upper()
        elif angle is not None:
            # Snap position label
            for pos, a in self.DIVERTER_ANGLES.items():
                if abs(a - angle) < 10:
                    self._diverter_pos = pos
                    break
        self.diverter.set_angle(angle, smooth=True)

    def divert_bio(self):  self.set_diverter(position="BIO")
    def divert_rec(self):  self.set_diverter(position="REC")
    def divert_haz(self):  self.set_diverter(position="HAZ")
    def divert_neutral(self): self.set_diverter(position="NEUTRAL")

    # ── Flap (entry lid) ──────────────────────────────────────────────────────
    def flap_open(self):
        self.flap.set_angle(self.FLAP_OPEN, smooth=True)

    def flap_close(self):
        self.flap.set_angle(self.FLAP_CLOSED, smooth=True)

    # ── Bag clip ──────────────────────────────────────────────────────────────
    def clip_open(self):
        self.clip.set_angle(self.CLIP_OPEN, smooth=True)

    def clip_close(self):
        self.clip.set_angle(self.CLIP_CLOSED, smooth=True)

    # ── Piston ────────────────────────────────────────────────────────────────
    def piston_extend(self, hold_ms: int = 900):
        """Activate extend relay for hold_ms milliseconds."""
        self._piston_extend.value(1)
        time.sleep_ms(hold_ms)
        # Don't deactivate — piston held until retract command

    def piston_retract(self, hold_ms: int = 750):
        """Deactivate extend, activate retract relay."""
        self._piston_extend.value(0)
        time.sleep_ms(50)
        self._piston_retract.value(1)
        time.sleep_ms(hold_ms)
        self._piston_retract.value(0)

    def piston_idle(self):
        self._piston_extend.value(0)
        self._piston_retract.value(0)

    # ── Heat seal ─────────────────────────────────────────────────────────────
    def heat_seal_on(self):
        self._heat_seal.value(1)

    def heat_seal_off(self):
        self._heat_seal.value(0)

    # ── Home all ──────────────────────────────────────────────────────────────
    def home_all(self):
        """Move all servos to safe home positions on boot."""
        self.divert_neutral()
        self.flap_close()
        self.clip_close()
        self.piston_idle()
        self.heat_seal_off()
        time.sleep_ms(500)
        # Power off servos after homing (reduce heat)
        self.diverter.off()
        self.flap.off()
        self.clip.off()
