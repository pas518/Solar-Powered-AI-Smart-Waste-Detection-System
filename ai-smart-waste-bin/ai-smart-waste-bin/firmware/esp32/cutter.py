"""
cutter.py — Bag Cutter DC Motor Controller (MicroPython)
"""
from machine import Pin, PWM
import time


class BagCutter:
    """
    Controls the rotary bag-cutter DC motor via L298N H-bridge.
    Runs at 90% duty cycle for ~1150 RPM continuous.
    """

    DUTY_NORMAL = int(0.90 * 65535)
    DUTY_SLOW   = int(0.50 * 65535)

    def __init__(self, in1: int = 5, in2: int = 18, en: int = 19):
        self._in1 = Pin(in1, Pin.OUT, value=0)
        self._in2 = Pin(in2, Pin.OUT, value=0)
        self._pwm = PWM(Pin(en), freq=20_000)
        self._pwm.duty_u16(0)
        self._running = False

    def start(self, duty: int | None = None):
        self._in1.value(1)
        self._in2.value(0)
        self._pwm.duty_u16(duty or self.DUTY_NORMAL)
        self._running = True

    def stop(self):
        self._pwm.duty_u16(0)
        self._in1.value(0)
        self._in2.value(0)
        self._running = False

    def brake(self):
        """Short-circuit brake for faster stop."""
        self._pwm.duty_u16(65535)
        self._in1.value(1)
        self._in2.value(1)
        time.sleep_ms(100)
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._running
