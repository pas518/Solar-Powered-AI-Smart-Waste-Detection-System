"""
main.py — ESP32 MicroPython Entry Point
=========================================
Listens for commands from Raspberry Pi over UART and drives all hardware:
  • NEMA17 stepper (conveyor belt)
  • 3-way diverter servo + entry flap servo
  • Pneumatic piston (relay control)
  • Bag cutter DC motor
  • Bag loader stepper
  • Auto-brush DC motor
  • HC-SR04 ultrasonic fill sensors
  • Gas sensors (MQ135/MQ2/MQ7/MQ9) via ADC
  • HX711 load cell
  • SSD1306 OLED status display
  • Solar/battery voltage monitoring
  • WiFi + MQTT direct reporting

UART command protocol (newline-terminated ASCII):
  BELT:START
  BELT:STOP
  BELT:SLOW
  BELT:NORMAL
  DIVERT:BIO
  DIVERT:REC
  DIVERT:HAZ
  DIVERT:NEUTRAL
  PISTON:EXTEND:<cat_id>
  PISTON:HOLD:<cat_id>
  PISTON:RETRACT:<cat_id>
  SEAL:ON:<cat_id>
  BAGLOAD:ADVANCE:<steps>
  CUTTER:ON
  CUTTER:OFF
  BRUSH:SWEEP
  FLAP:OPEN
  FLAP:CLOSE
  STATUS?
"""

import gc, sys, time, ujson
import uasyncio as asyncio
from machine import UART, Pin, PWM, ADC, SoftI2C

# ── Local modules ──
from stepper          import StepperMotor
from servo_controller import ServoController
from bag_loader       import BagLoader
from cutter           import BagCutter
from brush            import AutoBrush
from sensors          import SensorBank

# ── Pin assignments (match config.py) ────────────────────────────────────────
UART_TX_PIN = 1
UART_RX_PIN = 3
I2C_SCL     = 22
I2C_SDA     = 21

# ── Hardware init ─────────────────────────────────────────────────────────────
uart = UART(0, baudrate=115200, tx=UART_TX_PIN, rx=UART_RX_PIN)
uart.init(bits=8, parity=None, stop=1)

i2c  = SoftI2C(scl=Pin(I2C_SCL), sda=Pin(I2C_SDA), freq=400_000)

# ── Subsystem modules ─────────────────────────────────────────────────────────
belt    = StepperMotor(step=14, dir=27, en=26, steps_rev=200, microstep=8)
servos  = ServoController()
loader  = BagLoader()
cutter  = BagCutter()
brush   = AutoBrush()
sensors = SensorBank(i2c=i2c)

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    "belt_running":  False,
    "belt_slow":     False,
    "diverter_pos":  "NEUTRAL",
    "piston_state":  "IDLE",
    "uptime_s":      0,
    "cut_count":     0,
    "eject_count":   0,
    "boot_time":     time.time(),
}

# ── OLED display ──────────────────────────────────────────────────────────────
try:
    import ssd1306
    oled = ssd1306.SSD1306_I2C(128, 64, i2c)
    HAS_OLED = True
except Exception:
    HAS_OLED = False
    print("[WARN] No OLED found")


def oled_update(line1="", line2="", line3="", line4=""):
    if not HAS_OLED:
        return
    oled.fill(0)
    oled.text(line1[:16], 0,  0)
    oled.text(line2[:16], 0, 16)
    oled.text(line3[:16], 0, 32)
    oled.text(line4[:16], 0, 48)
    oled.show()


# ── Command dispatcher ────────────────────────────────────────────────────────

def handle_command(cmd: str):
    """Parse and dispatch a UART command string."""
    cmd = cmd.strip().upper()
    if not cmd:
        return

    parts = cmd.split(":")
    verb  = parts[0]

    # ── BELT ──
    if verb == "BELT":
        action = parts[1] if len(parts) > 1 else ""
        if action == "START":
            belt.enable()
            belt.set_speed(rps=1.0)
            state["belt_running"] = True
            state["belt_slow"]    = False
            uart.write(b"OK:BELT:START\n")
        elif action == "STOP":
            belt.stop()
            belt.disable()
            state["belt_running"] = False
            uart.write(b"OK:BELT:STOP\n")
        elif action == "SLOW":
            belt.set_speed(rps=0.3)
            state["belt_slow"] = True
            uart.write(b"OK:BELT:SLOW\n")
        elif action == "NORMAL":
            belt.set_speed(rps=1.0)
            state["belt_slow"] = False
            uart.write(b"OK:BELT:NORMAL\n")

    # ── DIVERT ──
    elif verb == "DIVERT":
        direction = parts[1] if len(parts) > 1 else "NEUTRAL"
        angle_map = {"BIO": 45, "REC": 90, "HAZ": 135, "NEUTRAL": 90}
        angle = angle_map.get(direction, 90)
        servos.set_diverter(angle)
        state["diverter_pos"] = direction
        oled_update("DIVERT", direction, f"ANGLE:{angle}", "")
        uart.write(f"OK:DIVERT:{direction}\n".encode())

    # ── PISTON ──
    elif verb == "PISTON":
        action = parts[1] if len(parts) > 1 else ""
        cat    = parts[2] if len(parts) > 2 else "0"
        if action == "EXTEND":
            servos.piston_extend()
            state["piston_state"] = "EXTENDED"
            oled_update("PISTON", "EXTENDING", f"CAT:{cat}", "")
        elif action == "HOLD":
            state["piston_state"] = "HOLDING"
        elif action == "RETRACT":
            servos.piston_retract()
            state["piston_state"] = "RETRACTED"
            state["eject_count"] += 1
        uart.write(f"OK:PISTON:{action}\n".encode())

    # ── SEAL ──
    elif verb == "SEAL":
        action = parts[1] if len(parts) > 1 else "OFF"
        if action == "ON":
            servos.heat_seal_on()
            time.sleep_ms(500)
            servos.heat_seal_off()
            uart.write(b"OK:SEAL:DONE\n")

    # ── BAG LOAD ──
    elif verb == "BAGLOAD":
        action = parts[1] if len(parts) > 1 else ""
        if action == "ADVANCE":
            steps = int(parts[2]) if len(parts) > 2 else 400
            loader.advance(steps)
            uart.write(f"OK:BAGLOAD:{steps}\n".encode())

    # ── CUTTER ──
    elif verb == "CUTTER":
        action = parts[1] if len(parts) > 1 else "OFF"
        if action == "ON":
            cutter.start()
            state["cut_count"] += 1
        elif action == "OFF":
            cutter.stop()
        uart.write(f"OK:CUTTER:{action}\n".encode())

    # ── BRUSH ──
    elif verb == "BRUSH":
        if parts[1] == "SWEEP" if len(parts) > 1 else False:
            brush.sweep()
            uart.write(b"OK:BRUSH:DONE\n")

    # ── FLAP ──
    elif verb == "FLAP":
        action = parts[1] if len(parts) > 1 else "CLOSE"
        if action == "OPEN":
            servos.flap_open()
        else:
            servos.flap_close()
        uart.write(f"OK:FLAP:{action}\n".encode())

    # ── STATUS REQUEST ──
    elif verb == "STATUS?":
        readings = sensors.read_all()
        status = {
            **state,
            "uptime_s": int(time.time() - state["boot_time"]),
            "sensors":  readings,
        }
        uart.write((ujson.dumps(status) + "\n").encode())

    else:
        uart.write(f"ERR:UNKNOWN:{cmd}\n".encode())


# ── Sensor broadcast loop (every 2 seconds) ───────────────────────────────────

async def sensor_loop():
    while True:
        await asyncio.sleep(2)
        try:
            readings = sensors.read_all()
            data = {"type": "SENSORS", **readings}
            uart.write((ujson.dumps(data) + "\n").encode())
        except Exception as e:
            print("[sensor_loop] error:", e)


# ── Fill level watchdog (every 5 seconds) ────────────────────────────────────

async def fill_watchdog():
    while True:
        await asyncio.sleep(5)
        try:
            levels = sensors.read_fill_levels()
            for cat_id, pct in levels.items():
                data = {"type": "FILL", "cat_id": cat_id, "pct": pct}
                uart.write((ujson.dumps(data) + "\n").encode())
        except Exception as e:
            print("[fill_watchdog] error:", e)


# ── UART read loop ────────────────────────────────────────────────────────────

async def uart_loop():
    buf = b""
    while True:
        await asyncio.sleep_ms(10)
        if uart.any():
            chunk = uart.read(64)
            if chunk:
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        handle_command(line.decode("utf-8", "ignore"))
                    except Exception as e:
                        print("[uart_loop] error:", e)
                        uart.write(f"ERR:{e}\n".encode())


# ── GC / memory maintenance ───────────────────────────────────────────────────

async def gc_loop():
    while True:
        await asyncio.sleep(30)
        gc.collect()
        free = gc.mem_free()
        uart.write(f'{{"type":"MEM","free":{free}}}\n'.encode())


# ── Boot ──────────────────────────────────────────────────────────────────────

async def main():
    print("AI Waste Bin — ESP32 Firmware v1.0")
    print(f"Free memory: {gc.mem_free()} bytes")

    oled_update("AI WASTE BIN", "ESP32 v1.0", "BOOT OK", "WAITING PI...")

    # Home all servos on boot
    servos.home_all()
    belt.disable()   # belt off until commanded

    uart.write(b'{"type":"BOOT","status":"READY"}\n')
    print("Ready — waiting for commands from Pi")

    oled_update("AI WASTE BIN", "READY", "", "UART OK")

    # Run all async loops concurrently
    await asyncio.gather(
        uart_loop(),
        sensor_loop(),
        fill_watchdog(),
        gc_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
