"""
brush.py — Auto-Brush Sweep Controller (MicroPython)
"""
from machine import Pin, PWM
import time


class AutoBrush:
    """
    Belt cleaning brush — DC motor that sweeps back and forth.
    Triggered after each bag ejection to clean belt surface.
    """

    DUTY = int(0.70 * 65535)

    def __init__(self, in1: int = 21, in2: int = 22):
        self._in1    = Pin(in1, Pin.OUT, value=0)
        self._in2    = Pin(in2, Pin.OUT, value=0)
        self._sweeps = 0

    def _forward(self, ms: int):
        self._in1.value(1); self._in2.value(0)
        time.sleep_ms(ms)
        self._in1.value(0)

    def _backward(self, ms: int):
        self._in1.value(0); self._in2.value(1)
        time.sleep_ms(ms)
        self._in2.value(0)

    def sweep(self, passes: int = 2, ms_per_pass: int = 600):
        """Execute N back-and-forth sweeps."""
        for _ in range(passes):
            self._forward(ms_per_pass)
            time.sleep_ms(80)
            self._backward(ms_per_pass)
            time.sleep_ms(80)
        self._sweeps += 1

    @property
    def sweep_count(self) -> int:
        return self._sweeps
