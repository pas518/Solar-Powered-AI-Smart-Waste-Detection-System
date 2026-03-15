"""
iot_client.py — MQTT IoT Client
=================================
Publishes waste classification results and system telemetry to MQTT broker.
Compatible with Home Assistant, Node-RED, AWS IoT, and custom dashboards.

Topics (see config.py for full list):
  waste_bin/detection   — per-item classification result
  waste_bin/bins        — bin fill levels
  waste_bin/sensors     — gas / moisture / weight sensor readings
  waste_bin/telemetry   — system health (battery, uptime, fps)
  waste_bin/alert       — bin full / anomaly / error alerts
  waste_bin/status      — online/offline heartbeat

Graceful offline mode: queues messages and retries on reconnect.
"""

from __future__ import annotations
import json, logging, time, queue, threading
from typing import Optional

log = logging.getLogger("iot_client")

try:
    from config import (
        MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
        MQTT_CLIENT_ID, MQTT_QOS, MQTT_KEEPALIVE,
        MQTT_TOPICS, TELEMETRY_INTERVAL_S,
    )
except ImportError:
    MQTT_BROKER         = "localhost"
    MQTT_PORT           = 1883
    MQTT_USERNAME       = ""
    MQTT_PASSWORD       = ""
    MQTT_CLIENT_ID      = "waste_bin_001"
    MQTT_QOS            = 1
    MQTT_KEEPALIVE      = 60
    MQTT_TOPICS         = {
        "detection":  "waste_bin/detection",
        "bin_levels": "waste_bin/bins",
        "sensors":    "waste_bin/sensors",
        "status":     "waste_bin/status",
        "alert":      "waste_bin/alert",
        "telemetry":  "waste_bin/telemetry",
    }
    TELEMETRY_INTERVAL_S = 10


class IoTClient:
    """
    Non-blocking MQTT client with offline queue.

    Usage
    -----
        iot = IoTClient()
        iot.connect()
        iot.publish_detection(result_dict)
        iot.publish_telemetry({"fps": 28, "soc": 82})
    """

    def __init__(
        self,
        broker:    str = MQTT_BROKER,
        port:      int = MQTT_PORT,
        client_id: str = MQTT_CLIENT_ID,
        username:  str = MQTT_USERNAME,
        password:  str = MQTT_PASSWORD,
    ):
        self.broker    = broker
        self.port      = port
        self.client_id = client_id
        self.username  = username
        self.password  = password

        self._client        = None
        self._connected     = False
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._pub_thread    = None
        self._telem_thread  = None
        self._running       = False
        self._pkt_count     = 0
        self._last_telem    = 0.0
        self._telem_fn      = None   # set by caller for auto telemetry

    # ── Connect ───────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError:
            log.warning("paho-mqtt not installed — IoT client in log-only mode")
            self._connected = False
            self._start_log_only_mode()
            return False

        self._client = mqtt.Client(client_id=self.client_id, clean_session=False)

        if self.username:
            self._client.username_pw_set(self.username, self.password)

        # Last-will message (published by broker if client disconnects unexpectedly)
        self._client.will_set(
            MQTT_TOPICS.get("status", "waste_bin/status"),
            json.dumps({"status": "offline", "client": self.client_id}),
            qos=MQTT_QOS, retain=True
        )

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish    = self._on_publish

        try:
            self._client.connect(self.broker, self.port, keepalive=MQTT_KEEPALIVE)
            self._client.loop_start()
            self._running = True
            self._start_pub_thread()
            log.info("MQTT connecting to %s:%d as %s", self.broker, self.port, self.client_id)
            return True
        except Exception as e:
            log.error("MQTT connect failed: %s — running in offline queue mode", e)
            self._start_log_only_mode()
            return False

    def disconnect(self):
        self._running = False
        if self._client:
            self._publish_status("offline")
            self._client.loop_stop()
            self._client.disconnect()
        log.info("MQTT disconnected. Total packets: %d", self._pkt_count)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            log.info("MQTT connected ✓")
            self._publish_status("online")
        else:
            log.error("MQTT connect refused: rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        log.warning("MQTT disconnected (rc=%d) — will retry", rc)

    def _on_publish(self, client, userdata, mid):
        self._pkt_count += 1

    # ── Publish helpers ───────────────────────────────────────────────────────
    def _enqueue(self, topic: str, payload: dict):
        msg = {"topic": topic, "payload": payload, "ts": time.time()}
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            log.warning("MQTT queue full — dropping oldest message")
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(msg)

    def _start_pub_thread(self):
        self._pub_thread = threading.Thread(target=self._pub_loop, daemon=True)
        self._pub_thread.start()

    def _pub_loop(self):
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if self._connected and self._client:
                try:
                    payload = json.dumps(msg["payload"])
                    self._client.publish(msg["topic"], payload, qos=MQTT_QOS)
                except Exception as e:
                    log.error("MQTT publish error: %s", e)
                    self._queue.put(msg)  # re-queue
                    time.sleep(2.0)
            else:
                # Not connected — re-queue and wait
                self._queue.put(msg)
                time.sleep(3.0)

    def _start_log_only_mode(self):
        """When MQTT is unavailable, just log all messages."""
        self._running = True
        self._pub_thread = threading.Thread(target=self._log_loop, daemon=True)
        self._pub_thread.start()

    def _log_loop(self):
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
                log.debug("[IoT-log] %s → %s", msg["topic"],
                          json.dumps(msg["payload"])[:120])
                self._pkt_count += 1
            except queue.Empty:
                continue

    def _publish_status(self, status: str):
        self._enqueue(MQTT_TOPICS.get("status", "waste_bin/status"), {
            "status":    status,
            "client_id": self.client_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    # ── Public publish methods ────────────────────────────────────────────────
    def publish_detection(self, result: dict):
        """Publish a single waste item classification result."""
        payload = {
            "cat_id":     result.get("cat_id", -1),
            "category":   result.get("category", "unknown"),
            "short":      result.get("short", "?"),
            "confidence": round(result.get("confidence", 0.0), 3),
            "anomaly":    result.get("anomaly", False),
            "label":      result.get("det_label", result.get("label", "")),
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._enqueue(MQTT_TOPICS.get("detection", "waste_bin/detection"), payload)

    def publish_bin_levels(self, bins: dict):
        """Publish current bin fill levels. bins = {short: {fill_pct, count, ...}}"""
        self._enqueue(MQTT_TOPICS.get("bin_levels", "waste_bin/bins"), {
            "bins":      bins,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def publish_sensors(self, sensor_data: dict):
        """Publish raw sensor readings."""
        self._enqueue(MQTT_TOPICS.get("sensors", "waste_bin/sensors"), {
            **sensor_data,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def publish_telemetry(self, data: dict):
        """Publish system health metrics."""
        self._enqueue(MQTT_TOPICS.get("telemetry", "waste_bin/telemetry"), {
            **data,
            "client_id":  self.client_id,
            "packets":    self._pkt_count,
            "queue_size": self._queue.qsize(),
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def publish_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Publish an alert (bin full, anomaly detected, hardware fault)."""
        self._enqueue(MQTT_TOPICS.get("alert", "waste_bin/alert"), {
            "type":      alert_type,
            "message":   message,
            "severity":  severity,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    # ── Periodic auto-telemetry ───────────────────────────────────────────────
    def start_auto_telemetry(self, data_fn: Callable, interval_s: float = TELEMETRY_INTERVAL_S):
        """
        Call data_fn() every interval_s seconds and publish result.
        data_fn should return a dict with telemetry fields.
        """
        self._telem_fn = data_fn
        self._telem_thread = threading.Thread(
            target=self._telem_loop, args=(interval_s,), daemon=True
        )
        self._telem_thread.start()

    def _telem_loop(self, interval: float):
        while self._running:
            time.sleep(interval)
            if self._telem_fn:
                try:
                    data = self._telem_fn()
                    self.publish_telemetry(data)
                except Exception as e:
                    log.error("Telemetry callback error: %s", e)

    # ── Stats ─────────────────────────────────────────────────────────────────
    @property
    def packets_sent(self) -> int:
        return self._pkt_count

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_status(self) -> dict:
        return {
            "connected":  self._connected,
            "broker":     f"{self.broker}:{self.port}",
            "packets":    self._pkt_count,
            "queue_size": self._queue.qsize(),
        }


# ── Type hint ─────────────────────────────────────────────────────────────────
from typing import Callable  # noqa — used in start_auto_telemetry type hint
