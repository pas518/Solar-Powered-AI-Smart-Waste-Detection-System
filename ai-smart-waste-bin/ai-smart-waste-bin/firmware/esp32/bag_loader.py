"""
bag_loader.py — Bag Roll Auto-Load Controller (MicroPython)
"""
from machine import Pin
from stepper import StepperMotor
import time


class BagLoader:
    """
    Controls the bag-roll stepper motor and clip servo.
    Advances a fresh bag into position after each ejection.
    """

    def __init__(self):
        self._motor = StepperMotor(step=2, dir=4, en=26, steps_rev=200, microstep=4)
        self._pos   = 0  # total steps advanced this session
        self._bags  = 0  # bags loaded this session

    def advance(self, steps: int = 400):
        """Feed N steps of bag roll to position one fresh bag."""
        self._motor.enable()
        self._motor.set_direction(True)
        self._motor.step_n(steps, delay_us=1500)
        self._motor.disable()
        self._pos  += steps
        self._bags += 1

    def retract(self, steps: int = 50):
        """Pull back slightly to tension the bag."""
        self._motor.enable()
        self._motor.set_direction(False)
        self._motor.step_n(steps, delay_us=1500)
        self._motor.disable()

    def full_cycle(self):
        """Complete load sequence: advance + retract to tension."""
        self.advance(steps=400)
        time.sleep_ms(100)
        self.retract(steps=40)

    @property
    def bags_loaded(self) -> int:
        return self._bags

    @property
    def steps_used(self) -> int:
        return self._pos
