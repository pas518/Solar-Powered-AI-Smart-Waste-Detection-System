"""
sensors.py — Sensor Bank (MicroPython / ESP32)
================================================
Reads all physical sensors:
  • HC-SR04 ultrasonic (fill levels × 3)
  • MQ-135, MQ-2, MQ-7, MQ-9 gas sensors (ADC)
  • HX711 load cell
  • Metal inductive proximity sensor
  • Moisture sensor (ADC)
  • Solar voltage + battery voltage/current (ADC)
  • BMP280 temperature/pressure (I2C)
"""

from machine import Pin, ADC, SoftI2C
import time, math


# ══════════════════════════════════════════════════════════════════════════════
#  HC-SR04 Ultrasonic
# ══════════════════════════════════════════════════════════════════════════════

class HCSR04:
    def __init__(self, trig: int, echo: int, timeout_us: int = 30_000):
        self._trig     = Pin(trig, Pin.OUT, value=0)
        self._echo     = Pin(echo, Pin.IN)
        self._timeout  = timeout_us

    def distance_cm(self) -> float:
        # Send 10µs pulse
        self._trig.value(0); time.sleep_us(2)
        self._trig.value(1); time.sleep_us(10)
        self._trig.value(0)

        # Measure echo high time
        t0 = time.ticks_us()
        while self._echo.value() == 0:
            if time.ticks_diff(time.ticks_us(), t0) > self._timeout:
                return -1.0
        t1 = time.ticks_us()
        while self._echo.value() == 1:
            if time.ticks_diff(time.ticks_us(), t1) > self._timeout:
                return -1.0
        t2 = time.ticks_us()
        return time.ticks_diff(t2, t1) / 58.0   # cm


# ══════════════════════════════════════════════════════════════════════════════
#  HX711 Load Cell
# ══════════════════════════════════════════════════════════════════════════════

class HX711:
    def __init__(self, data: int, clk: int, scale: float = 2280.0, tare: float = 0.0):
        self._data  = Pin(data, Pin.IN)
        self._clk   = Pin(clk, Pin.OUT, value=0)
        self._scale = scale
        self._tare  = tare

    def _read_raw(self) -> int:
        while self._data.value():
            pass
        count = 0
        for _ in range(24):
            self._clk.value(1); time.sleep_us(1)
            count = (count << 1) | self._data.value()
            self._clk.value(0); time.sleep_us(1)
        # 25th pulse (gain 128)
        self._clk.value(1); time.sleep_us(1)
        self._clk.value(0)
        # Sign extension
        if count & 0x800000:
            count -= 0x1000000
        return count

    def tare(self, samples: int = 10):
        readings = [self._read_raw() for _ in range(samples)]
        self._tare = sum(readings) / len(readings)

    def weight_g(self) -> float:
        raw = self._read_raw()
        return (raw - self._tare) / self._scale


# ══════════════════════════════════════════════════════════════════════════════
#  Gas Sensor (MQ series via ADC)
# ══════════════════════════════════════════════════════════════════════════════

class MQSensor:
    """
    Reads MQ-series gas sensor via 12-bit ADC.
    Returns calibrated ppm value using Rs/R0 ratio method.
    """

    # Sensitivity curve parameters [a, b] for log-log interpolation: log(ppm) = a*log(Rs/R0) + b
    # Values approximate for MQ-135 / MQ-2 from datasheets
    PARAMS = {
        "MQ135": {"a": -0.42, "b": 1.92, "R0": 76.63, "RL": 20.0},
        "MQ2":   {"a": -0.47, "b": 1.85, "R0": 9.83,  "RL": 20.0},
        "MQ7":   {"a": -1.52, "b": 1.70, "R0": 27.50, "RL": 10.0},
        "MQ9":   {"a": -0.39, "b": 1.83, "R0": 11.04, "RL": 10.0},
    }

    def __init__(self, pin: int, model: str = "MQ135", vcc: float = 3.3):
        self._adc   = ADC(Pin(pin))
        self._adc.atten(ADC.ATTN_11DB)   # 0–3.3V range
        self._model = model
        self._vcc   = vcc
        self._p     = self.PARAMS.get(model, self.PARAMS["MQ135"])

    def read_voltage(self) -> float:
        raw = self._adc.read_u16()   # 0–65535
        return (raw / 65535.0) * self._vcc

    def read_rs(self) -> float:
        """Load resistance (Rs) at current gas concentration."""
        v_out = self.read_voltage()
        if v_out < 0.01:
            return float("inf")
        return self._p["RL"] * (self._vcc - v_out) / v_out

    def read_ppm(self) -> float:
        rs   = self.read_rs()
        ratio = rs / self._p["R0"]
        if ratio <= 0:
            return 0.0
        log_ppm = self._p["a"] * math.log10(ratio) + self._p["b"]
        return max(0.0, 10 ** log_ppm)


# ══════════════════════════════════════════════════════════════════════════════
#  BMP280 (I2C)
# ══════════════════════════════════════════════════════════════════════════════

class BMP280Simple:
    """
    Minimal BMP280 reader (temperature + pressure).
    Full calibration compensation included.
    """

    def __init__(self, i2c, addr: int = 0x76):
        self._i2c  = i2c
        self._addr = addr
        self._cal  = self._read_calibration()
        # Set normal mode, 16× oversampling
        i2c.writeto_mem(addr, 0xF4, bytes([0xB7]))
        i2c.writeto_mem(addr, 0xF5, bytes([0x90]))

    def _read_calibration(self):
        raw = self._i2c.readfrom_mem(self._addr, 0x88, 24)
        import ustruct
        vals = ustruct.unpack("<HhhHhhhhhhhh", raw)
        return vals  # T1..T3, P1..P9

    def read(self) -> tuple[float, float]:
        """Returns (temperature_C, pressure_hPa)."""
        data = self._i2c.readfrom_mem(self._addr, 0xF7, 6)
        adc_P = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
        adc_T = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4)

        cal = self._cal
        # Temperature (from BMP280 datasheet)
        v1 = (adc_T / 16384.0 - cal[0] / 1024.0) * cal[1]
        v2 = ((adc_T / 131072.0 - cal[0] / 8192.0) ** 2) * cal[2]
        t_fine = v1 + v2
        temp   = t_fine / 5120.0

        # Pressure
        v1 = t_fine / 2.0 - 64000.0
        v2 = v1 * v1 * cal[8] / 32768.0 + v1 * cal[7] * 2.0
        v2 = v2 / 4.0 + cal[6] * 65536.0
        v1 = (cal[5] * (v1 * v1 * cal[4] / 524288.0 + v1 * cal[3]) / 524288.0 + 1) / 2.0
        if v1 == 0:
            return temp, 0.0
        pres = ((1048576.0 - adc_P) - v2 / 4096.0) * 6250.0 / v1
        v1   = cal[11] * pres * pres / 2147483648.0
        v2   = pres * cal[10] / 32768.0
        pres = (pres + (v1 + v2 + cal[9]) / 16.0) / 100.0

        return round(temp, 2), round(pres, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Full Sensor Bank
# ══════════════════════════════════════════════════════════════════════════════

class SensorBank:
    """
    Aggregates all sensors. Call read_all() to get a complete snapshot.
    """

    # Bin-depth (cm) when completely empty
    BIN_EMPTY_CM = 40.0

    def __init__(self, i2c):
        # Ultrasonic fill sensors
        self._uss = [
            HCSR04(trig=15, echo=16),   # BIO
            HCSR04(trig=17, echo=23),   # REC
            HCSR04(trig=34, echo=36),   # HAZ
        ]

        # Gas sensors
        self._mq135 = MQSensor(pin=39, model="MQ135")
        self._mq2   = MQSensor(pin=34, model="MQ2")
        self._mq7   = MQSensor(pin=35, model="MQ7")
        self._mq9   = MQSensor(pin=32, model="MQ9")

        # Load cell
        self._hx711 = HX711(data=36, clk=39)

        # Metal detector (digital NPN-NO)
        self._metal = Pin(37, Pin.IN, Pin.PULL_UP)

        # Moisture ADC
        self._moisture = ADC(Pin(38))
        self._moisture.atten(ADC.ATTN_11DB)

        # Solar / battery monitoring
        self._sol_v   = ADC(Pin(33)); self._sol_v.atten(ADC.ATTN_11DB)
        self._bat_v   = ADC(Pin(35)); self._bat_v.atten(ADC.ATTN_11DB)
        self._bat_i   = ADC(Pin(34)); self._bat_i.atten(ADC.ATTN_11DB)

        # BMP280
        try:
            self._bmp = BMP280Simple(i2c)
            self._has_bmp = True
        except Exception:
            self._has_bmp = False

    # ── Fill levels ───────────────────────────────────────────────────────────
    def read_fill_levels(self) -> dict:
        result = {}
        cat_names = ["BIO", "REC", "HAZ"]
        for i, us in enumerate(self._uss):
            d = us.distance_cm()
            if d < 0:
                d = self.BIN_EMPTY_CM
            pct = max(0.0, min(1.0, 1.0 - d / self.BIN_EMPTY_CM))
            result[i] = {"cat": cat_names[i], "pct": round(pct, 3), "cm": round(d, 1)}
        return result

    # ── All sensors ───────────────────────────────────────────────────────────
    def read_all(self) -> dict:
        fills = self.read_fill_levels()

        # Moisture (0–100%)
        moist_raw  = self._moisture.read_u16()
        moisture   = round((moist_raw / 65535.0) * 100.0, 1)

        # Metal
        metal_mv   = round((1 - self._metal.value()) * 3300.0, 0)

        # Gas ppm
        mq135_ppm  = round(self._mq135.read_ppm(), 1)
        mq2_ppm    = round(self._mq2.read_ppm(), 1)
        mq7_ppm    = round(self._mq7.read_ppm(), 1)
        mq9_ppm    = round(self._mq9.read_ppm(), 1)

        # Weight
        try:
            weight_g = round(self._hx711.weight_g(), 1)
        except Exception:
            weight_g = 0.0

        # Solar & battery (with resistor divider correction ×6)
        sol_v = round((self._sol_v.read_u16()  / 65535.0) * 3.3 * 6.0, 2)
        bat_v = round((self._bat_v.read_u16()  / 65535.0) * 3.3 * 6.0, 2)
        bat_i_raw = self._bat_i.read_u16() / 65535.0 * 3.3
        bat_i = round(bat_i_raw / 0.01, 2)   # 10mΩ shunt

        # BMP280
        temp_c, pres_hpa = (0.0, 0.0)
        if self._has_bmp:
            try:
                temp_c, pres_hpa = self._bmp.read()
            except Exception:
                pass

        return {
            "fill":       fills,
            "moisture":   moisture,
            "metal_mv":   metal_mv,
            "mq135_ppm":  mq135_ppm,
            "mq2_ppm":    mq2_ppm,
            "mq7_co_ppm": mq7_ppm,
            "mq9_ppm":    mq9_ppm,
            "weight_g":   weight_g,
            "solar_v":    sol_v,
            "battery_v":  bat_v,
            "battery_i":  bat_i,
            "battery_w":  round(bat_v * bat_i, 1),
            "temp_c":     temp_c,
            "pressure":   pres_hpa,
        }
