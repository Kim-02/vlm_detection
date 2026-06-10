"""
RTSP CCTV action detector.

- FastAPI + native WebSocket
- MJPEG video stream via /video_feed
- TensorRT Qwen2.5-VL for multi-frame action detection
- Semi-transparent overlay showing live analysis result
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from vlm_engine import TensorRTQwenVL


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|rtsp_flags;prefer_tcp|fflags;nobuffer|"
    "flags;low_delay|max_delay;0|reorder_queue_size;0|"
    "analyzeduration;0|probesize;32"
)

templates = Jinja2Templates(directory="templates")

SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS: dict[str, Any] = {
    "rtsp_url": "rtsp://admin:password@192.168.0.65:554/stream1",
    "engine_path": "/media/ds/DATA/engines/qwen25-vl-7b-4k-12batch",
    "llm_inference_bin": "/home/ds/edge_llm/TensorRT-Edge-LLM/build/examples/llm/llm_inference",
    "plugin_path": "",
    "capture_fps": 30,
    "jpeg_quality": 80,
    "frames_per_request": 10,
    "analysis_interval_sec": 1.0,
    "startup_test_prompt": "이미지에 사람이 있는지 확인해줘.",
}
SETTINGS_KEYS = set(DEFAULT_SETTINGS)

settings_lock = threading.RLock()
settings: dict[str, Any] = dict(DEFAULT_SETTINGS)

model_lock = threading.RLock()
model_loader_thread: threading.Thread | None = None

latest_frame_lock = threading.RLock()
latest_frame = None

jpeg_lock = threading.RLock()
jpeg_condition = threading.Condition(jpeg_lock)
latest_jpeg: bytes | None = None
latest_jpeg_seq = 0

stream_lock = threading.RLock()
stream_thread: threading.Thread | None = None
stream_state: dict[str, Any] = {
    "running": False,
    "generation": 0,
    "error": None,
    "rtsp_url": "",
    "stop_event": None,
}

# Circular buffer: stores the last ~4 seconds of frames at 30fps
frame_buffer_lock = threading.RLock()
frame_buffer: collections.deque = collections.deque(maxlen=120)

detection_lock = threading.RLock()
detection_state: dict[str, Any] = {
    "active": False,
    "prompt": "",
    "request_id": 0,
    "running": False,
    "result": {
        "detected": False,
        "description": "",
        "confidence": None,
        "time": None,
    },
}

vlm_call_lock = threading.Lock()


class DetectionStartRequest(BaseModel):
    prompt: str


def now_text() -> str:
    return datetime.now().strftime("%H:%M:%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Settings ──────────────────────────────────────────────────────────────────

def coerce_settings(data: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in SETTINGS_KEYS:
        if key in data:
            cleaned[key] = data[key]

    for key in ("rtsp_url", "engine_path", "llm_inference_bin", "plugin_path", "startup_test_prompt"):
        if key in cleaned:
            cleaned[key] = str(cleaned[key]).strip()

    if "capture_fps" in cleaned:
        try:
            cleaned["capture_fps"] = max(1, min(120, int(float(cleaned["capture_fps"]))))
        except (TypeError, ValueError):
            cleaned["capture_fps"] = DEFAULT_SETTINGS["capture_fps"]

    if "jpeg_quality" in cleaned:
        try:
            cleaned["jpeg_quality"] = max(10, min(100, int(float(cleaned["jpeg_quality"]))))
        except (TypeError, ValueError):
            cleaned["jpeg_quality"] = DEFAULT_SETTINGS["jpeg_quality"]

    if "frames_per_request" in cleaned:
        try:
            cleaned["frames_per_request"] = max(1, min(30, int(float(cleaned["frames_per_request"]))))
        except (TypeError, ValueError):
            cleaned["frames_per_request"] = DEFAULT_SETTINGS["frames_per_request"]

    if "analysis_interval_sec" in cleaned:
        try:
            cleaned["analysis_interval_sec"] = max(0.5, float(cleaned["analysis_interval_sec"]))
        except (TypeError, ValueError):
            cleaned["analysis_interval_sec"] = DEFAULT_SETTINGS["analysis_interval_sec"]

    return cleaned


def load_settings() -> None:
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    with settings_lock:
        settings.update(coerce_settings(saved))


def save_settings() -> None:
    with settings_lock:
        data = dict(settings)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_settings_from_payload(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    cleaned = coerce_settings(data)
    with settings_lock:
        old_engine_path = settings["engine_path"]
        old_llm_inference_bin = settings["llm_inference_bin"]
        old_plugin_path = settings["plugin_path"]
        old_startup_test_prompt = settings["startup_test_prompt"]
        settings.update(cleaned)
        snapshot = dict(settings)
    save_settings()
    reload_model_flag = (
        bool(cleaned.get("engine_path") and cleaned["engine_path"] != old_engine_path)
        or bool(cleaned.get("llm_inference_bin") and cleaned["llm_inference_bin"] != old_llm_inference_bin)
        or bool(cleaned.get("plugin_path") and cleaned["plugin_path"] != old_plugin_path)
        or bool(cleaned.get("startup_test_prompt") and cleaned["startup_test_prompt"] != old_startup_test_prompt)
    )
    return snapshot, reload_model_flag


# ── WebSocket manager ─────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        with self._lock:
            if ws in self.active:
                self.active.remove(ws)

    async def broadcast(self, msg: dict[str, Any]) -> None:
        with self._lock:
            conns = list(self.active)
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def emit(self, msg: dict[str, Any]) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(msg), self._loop)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app_obj: FastAPI):
    manager._loop = asyncio.get_running_loop()
    load_settings()
    init_model_state(app_obj)
    start_model_loading(app_obj)
    yield
    stop_stream()


app = FastAPI(lifespan=lifespan)


# ── Model management ──────────────────────────────────────────────────────────

def init_model_state(app_obj: FastAPI) -> None:
    with settings_lock:
        engine_path = settings["engine_path"]
        llm_inference_bin = settings["llm_inference_bin"]
        plugin_path = settings["plugin_path"]
    with model_lock:
        app_obj.state.vlm_engine = None
        app_obj.state.model_status = "loading"
        app_obj.state.startup_test_status = "pending"
        app_obj.state.engine_path = engine_path
        app_obj.state.llm_inference_bin = llm_inference_bin
        app_obj.state.plugin_path = plugin_path
        app_obj.state.startup_test_result = None
        app_obj.state.last_error = None
        app_obj.state.model_generation = 0


def model_snapshot() -> dict[str, Any]:
    with settings_lock:
        configured_engine_path = settings["engine_path"]
        configured_llm_bin = settings["llm_inference_bin"]
        configured_plugin = settings["plugin_path"]
    with model_lock:
        return {
            "model_status": getattr(app.state, "model_status", "loading"),
            "startup_test_status": getattr(app.state, "startup_test_status", "pending"),
            "engine_path": getattr(app.state, "engine_path", configured_engine_path),
            "llm_inference_bin": getattr(app.state, "llm_inference_bin", configured_llm_bin),
            "plugin_path": getattr(app.state, "plugin_path", configured_plugin),
            "startup_test_result": getattr(app.state, "startup_test_result", None),
            "last_error": getattr(app.state, "last_error", None),
        }


def start_model_loading(app_obj: FastAPI) -> None:
    global model_loader_thread
    with settings_lock:
        engine_path = settings["engine_path"]
        startup_test_prompt = settings["startup_test_prompt"]
        llm_inference_bin = settings["llm_inference_bin"]
        plugin_path = settings["plugin_path"]

    with model_lock:
        generation = getattr(app_obj.state, "model_generation", 0) + 1
        app_obj.state.model_generation = generation
        app_obj.state.vlm_engine = None
        app_obj.state.model_status = "loading"
        app_obj.state.startup_test_status = "pending"
        app_obj.state.engine_path = engine_path
        app_obj.state.llm_inference_bin = llm_inference_bin
        app_obj.state.plugin_path = plugin_path
        app_obj.state.startup_test_result = None
        app_obj.state.last_error = None

    model_loader_thread = threading.Thread(
        target=model_loader_worker,
        args=(app_obj, generation, engine_path, startup_test_prompt, llm_inference_bin, plugin_path),
        daemon=True,
    )
    model_loader_thread.start()
    emit_status("TensorRT engine loading...", "model_loading")


def should_reload_model(reload_requested: bool) -> bool:
    if reload_requested:
        return True
    with model_lock:
        return getattr(app.state, "model_status", "loading") == "failed"


def reload_model(reason: str = "manual reload") -> dict[str, Any]:
    stop_detection()
    start_model_loading(app)
    return {"ok": True, "status": "model_reloading", "reason": reason, "app_status": current_status()}


def model_loader_worker(
    app_obj: FastAPI,
    generation: int,
    engine_path: str,
    startup_test_prompt: str,
    llm_inference_bin: str,
    plugin_path: str,
) -> None:
    try:
        engine = TensorRTQwenVL(
            engine_path,
            llm_inference_bin=llm_inference_bin,
            plugin_path=plugin_path,
        )
        test_frame = get_latest_frame_copy()
        if test_frame is None:
            test_frame = _create_test_image()

        with vlm_call_lock:
            raw_result = engine.infer([test_frame], startup_test_prompt)
        parsed = parse_engine_result(raw_result)
        if not isinstance(parsed, dict):
            raise ValueError("startup test did not return a dict")

        with model_lock:
            if generation != getattr(app_obj.state, "model_generation", None):
                return
            app_obj.state.vlm_engine = engine
            app_obj.state.model_status = "ready"
            app_obj.state.startup_test_status = "success"
            app_obj.state.startup_test_result = parsed
            app_obj.state.last_error = None
        emit_status("TensorRT engine ready", "model_ready")
    except Exception as exc:
        with model_lock:
            if generation != getattr(app_obj.state, "model_generation", None):
                return
            app_obj.state.vlm_engine = None
            app_obj.state.model_status = "failed"
            app_obj.state.startup_test_status = "failed"
            app_obj.state.startup_test_result = None
            app_obj.state.last_error = str(exc)
        emit_status(f"TensorRT engine failed: {exc}", "model_failed")


def _create_test_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (18, 24, 20)
    cv2.putText(image, "TensorRT Qwen2.5-VL startup test", (42, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 210, 110), 2, cv2.LINE_AA)
    return image


def get_ready_engine() -> TensorRTQwenVL:
    with model_lock:
        model_status = getattr(app.state, "model_status", "loading")
        engine = getattr(app.state, "vlm_engine", None)
    if model_status != "ready" or engine is None:
        raise RuntimeError(f"model is not ready: {model_status}")
    return engine


# ── Prompts & parsing ─────────────────────────────────────────────────────────

def action_detect_prompt(user_prompt: str, frame_count: int) -> str:
    return f"""다음 {frame_count}개의 이미지는 약 1초 동안 CCTV에서 순서대로 캡처한 프레임이다.
이 영상에서 아래 조건에 해당하는 행동을 하는 사람이 있는지 감지해라.

감지 조건:
"{user_prompt}"

반드시 아래 JSON 형식으로만 답해라.
감지되면:
{{
  "detected": true,
  "description": "감지된 행동과 상황을 구체적으로 설명",
  "confidence": 0.85
}}

감지되지 않으면:
{{
  "detected": false,
  "description": "",
  "confidence": 0.0
}}

JSON 외의 텍스트는 절대 출력하지 마라."""


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_engine_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return extract_json_object(result)
    raise TypeError(f"unsupported engine result type: {type(result).__name__}")


# ── State helpers ─────────────────────────────────────────────────────────────

def get_latest_frame_copy():
    with latest_frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()


def detection_snapshot() -> dict[str, Any]:
    with detection_lock:
        return {
            "active": bool(detection_state["active"]),
            "prompt": detection_state["prompt"],
            "running": bool(detection_state["running"]),
            "result": dict(detection_state["result"]),
        }


def current_status() -> dict[str, Any]:
    model = model_snapshot()
    with settings_lock:
        configured_rtsp_url = settings["rtsp_url"]
    with stream_lock:
        stream_running = bool(stream_state["running"])
        stream_error = stream_state["error"]
        rtsp_url = stream_state["rtsp_url"] or configured_rtsp_url
    detect = detection_snapshot()
    return {
        "model_status": model["model_status"],
        "startup_test_status": model["startup_test_status"],
        "engine_path": model["engine_path"],
        "llm_inference_bin": model["llm_inference_bin"],
        "plugin_path": model["plugin_path"],
        "stream_status": "running" if stream_running else "stopped",
        "detection": detect,
        "last_error": model["last_error"] or stream_error,
        "stream": {
            "running": stream_running,
            "rtsp_url": rtsp_url,
            "error": stream_error,
        },
    }


def emit_status(text: str, event_type: str = "status") -> None:
    manager.emit({"type": event_type, "text": text, "status": current_status()})


# ── Detection control ─────────────────────────────────────────────────────────

def start_detection(prompt: str) -> tuple[bool, str]:
    prompt = prompt.strip()
    if not prompt:
        return False, "prompt is empty"

    model = model_snapshot()
    if model["model_status"] != "ready":
        return False, f"model is not ready: {model['model_status']}"

    with stream_lock:
        if not stream_state["running"]:
            return False, "stream is not running"

    with detection_lock:
        detection_state["request_id"] += 1
        request_id = detection_state["request_id"]
        detection_state.update({
            "active": True,
            "prompt": prompt,
            "running": False,
            "result": {
                "detected": False,
                "description": "",
                "confidence": None,
                "time": None,
            },
        })

    threading.Thread(
        target=action_detect_worker,
        args=(prompt, request_id),
        daemon=True,
    ).start()
    emit_status("행동 감지 시작", "detection_start")
    return True, "detection started"


def stop_detection() -> None:
    with detection_lock:
        detection_state["request_id"] += 1
        detection_state["active"] = False
        detection_state["running"] = False
    emit_status("행동 감지 중지", "detection_stop")


# ── Detection worker ──────────────────────────────────────────────────────────

def action_detect_worker(prompt: str, request_id: int) -> None:
    while True:
        with detection_lock:
            if request_id != detection_state["request_id"] or not detection_state["active"]:
                return
            detection_state["running"] = True

        cycle_start = time.time()

        with settings_lock:
            frames_per_req = int(settings["frames_per_request"])
            interval_sec = float(settings["analysis_interval_sec"])

        with frame_buffer_lock:
            all_frames = list(frame_buffer)

        if len(all_frames) < 2:
            with detection_lock:
                if request_id == detection_state["request_id"]:
                    detection_state["running"] = False
            time.sleep(0.1)
            continue

        # Evenly subsample from the buffer
        n = min(frames_per_req, len(all_frames))
        if n >= 2:
            indices = [int(round(i * (len(all_frames) - 1) / (n - 1))) for i in range(n)]
        else:
            indices = [0]
        sampled_frames = [all_frames[i] for i in indices]

        try:
            engine = get_ready_engine()
            prompt_text = action_detect_prompt(prompt, len(sampled_frames))
            with vlm_call_lock:
                raw_result = engine.infer(sampled_frames, prompt_text)
            parsed = parse_engine_result(raw_result)

            detected = bool(parsed.get("detected", False))
            description = str(parsed.get("description") or "")
            confidence = None
            try:
                if parsed.get("confidence") is not None:
                    confidence = float(parsed["confidence"])
            except (TypeError, ValueError):
                pass

            with detection_lock:
                if request_id != detection_state["request_id"] or not detection_state["active"]:
                    return
                detection_state["running"] = False
                detection_state["result"] = {
                    "detected": detected,
                    "description": description,
                    "confidence": confidence,
                    "time": now_iso(),
                }

            if detected:
                emit_status(f"행동 감지됨: {description}", "action_detected")
            else:
                emit_status("감지 없음", "action_clear")

        except Exception as exc:
            with detection_lock:
                if request_id == detection_state["request_id"]:
                    detection_state["running"] = False
                    detection_state["result"] = {
                        "detected": False,
                        "description": f"오류: {exc}",
                        "confidence": None,
                        "time": now_iso(),
                    }
            emit_status(f"감지 오류: {exc}", "detection_error")
            time.sleep(0.5)
            continue

        elapsed = time.time() - cycle_start
        wait = interval_sec - elapsed
        if wait > 0:
            time.sleep(wait)


# ── Video overlay ─────────────────────────────────────────────────────────────

def draw_detection_overlay(frame_bgr):
    with detection_lock:
        active = bool(detection_state["active"])
        if not active:
            return frame_bgr
        result = dict(detection_state["result"])
        prompt = detection_state["prompt"]
        running = bool(detection_state["running"])

    h, w = frame_bgr.shape[:2]
    box_h = 96

    # Semi-transparent black box at the bottom
    roi = frame_bgr[h - box_h:h, 0:w]
    black = np.zeros_like(roi)
    cv2.addWeighted(black, 0.60, roi, 0.40, 0, roi)
    frame_bgr[h - box_h:h, 0:w] = roi

    detected = bool(result.get("detected", False))
    description = str(result.get("description") or "")
    confidence = result.get("confidence")
    y0 = h - box_h + 6

    if running:
        color = (255, 200, 0)
        status_text = "▷ 분석 중..."
    elif detected:
        color = (0, 255, 80)
        conf_str = f"  {int(confidence * 100)}%" if confidence is not None else ""
        status_text = f"● 행동 감지됨{conf_str}"
    else:
        color = (160, 160, 160)
        status_text = "○ 감지 없음"

    cv2.putText(frame_bgr, status_text, (14, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)

    prompt_display = prompt[:56] + "..." if len(prompt) > 56 else prompt
    cv2.putText(frame_bgr, f"조건: {prompt_display}", (14, y0 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 185, 185), 1, cv2.LINE_AA)

    if description and not running:
        desc_display = description[:82] + "..." if len(description) > 82 else description
        cv2.putText(frame_bgr, desc_display, (14, y0 + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 200), 1, cv2.LINE_AA)

    return frame_bgr


# ── Video pipeline ────────────────────────────────────────────────────────────

def publish_jpeg(frame_bgr) -> None:
    global latest_jpeg, latest_jpeg_seq
    with settings_lock:
        quality = int(settings["jpeg_quality"])
    display = draw_detection_overlay(frame_bgr.copy())
    ok, jpeg = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        with jpeg_condition:
            latest_jpeg = jpeg.tobytes()
            latest_jpeg_seq += 1
            jpeg_condition.notify_all()


def clear_latest_video() -> None:
    global latest_frame, latest_jpeg, latest_jpeg_seq
    with latest_frame_lock:
        latest_frame = None
    with jpeg_condition:
        latest_jpeg = None
        latest_jpeg_seq += 1
        jpeg_condition.notify_all()
    with frame_buffer_lock:
        frame_buffer.clear()


def open_capture(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    capture_props = {
        "CAP_PROP_BUFFERSIZE": 1,
        "CAP_PROP_OPEN_TIMEOUT_MSEC": 2000,
        "CAP_PROP_READ_TIMEOUT_MSEC": 2000,
    }
    for name, value in capture_props.items():
        prop = getattr(cv2, name, None)
        if prop is not None:
            cap.set(prop, value)
    return cap


def set_stream_error(generation: int, message: str | None) -> None:
    with stream_lock:
        if generation == stream_state["generation"]:
            stream_state["error"] = message


def rtsp_reader_worker(
    rtsp_url: str,
    stop_event: threading.Event,
    generation: int,
    frame_condition: threading.Condition,
    shared_frame: dict[str, Any],
) -> None:
    global latest_frame
    cap = open_capture(rtsp_url)
    reconnect_delay = 0.2

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            set_stream_error(generation, "frame read failed; reconnecting")
            cap.release()
            if stop_event.wait(reconnect_delay):
                break
            cap = open_capture(rtsp_url)
            continue

        with stream_lock:
            if generation != stream_state["generation"]:
                break

        set_stream_error(generation, None)

        with latest_frame_lock:
            latest_frame = frame

        with frame_condition:
            shared_frame["frame"] = frame
            shared_frame["seq"] += 1
            shared_frame["time"] = time.time()
            frame_condition.notify_all()

    cap.release()


def process_stream_frame(frame) -> None:
    with frame_buffer_lock:
        frame_buffer.append(frame.copy())
    publish_jpeg(frame)


def rtsp_capture_worker(rtsp_url: str, stop_event: threading.Event, generation: int) -> None:
    shared_frame: dict[str, Any] = {"frame": None, "seq": 0, "time": 0.0}
    frame_lock = threading.RLock()
    frame_condition = threading.Condition(frame_lock)
    reader_thread = threading.Thread(
        target=rtsp_reader_worker,
        args=(rtsp_url, stop_event, generation, frame_condition, shared_frame),
        daemon=True,
    )
    reader_thread.start()

    last_seq = 0
    last_publish_at = 0.0
    while not stop_event.is_set():
        with settings_lock:
            fps = max(1, int(settings["capture_fps"]))
        min_interval = 1.0 / fps

        with frame_condition:
            if shared_frame["seq"] == last_seq:
                frame_condition.wait(timeout=0.1)
            if shared_frame["frame"] is None or shared_frame["seq"] == last_seq:
                continue

            now = time.time()
            if now - last_publish_at < min_interval:
                last_seq = shared_frame["seq"]
                continue

            frame = shared_frame["frame"].copy()
            last_seq = shared_frame["seq"]

        process_stream_frame(frame)
        last_publish_at = time.time()

    stop_event.set()
    reader_thread.join(timeout=1.0)
    with frame_condition:
        frame_condition.notify_all()
    with stream_lock:
        if generation == stream_state["generation"]:
            stream_state["running"] = False
            stream_state["stop_event"] = None
    emit_status("● 스트림 중지됨", "stream_stopped")


def start_stream() -> str:
    global stream_thread
    with settings_lock:
        rtsp_url = settings["rtsp_url"]

    with stream_lock:
        if stream_state["running"] and stream_state["rtsp_url"] == rtsp_url:
            return "already_running"

        previous_stop = stream_state.get("stop_event")
        if previous_stop is not None:
            previous_stop.set()

        clear_latest_video()
        stream_state["generation"] += 1
        generation = stream_state["generation"]
        stop_event = threading.Event()
        stream_state.update({
            "running": True,
            "error": None,
            "rtsp_url": rtsp_url,
            "stop_event": stop_event,
        })
        stream_thread = threading.Thread(
            target=rtsp_capture_worker,
            args=(rtsp_url, stop_event, generation),
            daemon=True,
        )
        stream_thread.start()

    return "started"


def stop_stream() -> None:
    with stream_lock:
        stop_event = stream_state.get("stop_event")
        if stop_event is not None:
            stop_event.set()
        stream_state["running"] = False
        stream_state["stop_event"] = None
    clear_latest_video()
    stop_detection()


# ── MJPEG generator ───────────────────────────────────────────────────────────

def mjpeg_generator():
    last_seq = -1
    while True:
        with jpeg_condition:
            jpeg_condition.wait_for(
                lambda: latest_jpeg is not None and latest_jpeg_seq != last_seq,
                timeout=1.0,
            )
            data = latest_jpeg
            seq = latest_jpeg_seq
        if data is None or seq == last_seq:
            continue
        last_seq = seq
        header = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(data)}\r\n".encode("ascii")
            + b"Cache-Control: no-cache, no-store, must-revalidate\r\n\r\n"
        )
        yield header + data + b"\r\n"


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    with settings_lock:
        s = dict(settings)
    return templates.TemplateResponse(
        request, "index.html", {"settings": s, "status": current_status()}
    )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/frame")
def get_frame():
    with jpeg_lock:
        data = latest_jpeg
    if data is None:
        return Response(status_code=204)
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/status")
def api_status():
    return JSONResponse(current_status())


@app.post("/api/model/reload")
def api_model_reload():
    return JSONResponse(reload_model("manual REST reload"))


@app.post("/api/detection/start")
def api_detection_start(payload: DetectionStartRequest):
    ok, message = start_detection(payload.prompt)
    return JSONResponse({"ok": ok, "message": message, "status": current_status()})


@app.post("/api/detection/stop")
def api_detection_stop():
    stop_detection()
    return JSONResponse({"ok": True, "status": current_status()})


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_json({"type": "status", "text": "connected", "status": current_status()})
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action") or data.get("type")

            if action == "start_stream":
                _, reload_requested = update_settings_from_payload(data)
                if should_reload_model(reload_requested):
                    reload_model("stream start reload")
                result = start_stream()
                await ws.send_json({
                    "type": "stream_started",
                    "text": "● 스트리밍 중",
                    "result": result,
                    "status": current_status(),
                })

            elif action == "stop_stream":
                stop_stream()
                await ws.send_json({
                    "type": "stream_stopped",
                    "text": "● 중지됨",
                    "status": current_status(),
                })

            elif action == "start_detection":
                ok, message = start_detection(str(data.get("prompt") or ""))
                await ws.send_json({
                    "type": "detection_request",
                    "ok": ok,
                    "text": message,
                    "status": current_status(),
                })

            elif action == "stop_detection":
                stop_detection()
                await ws.send_json({
                    "type": "detection_stopped",
                    "text": "감지 중지됨",
                    "status": current_status(),
                })

            elif action == "reload_model":
                result = reload_model("manual WebSocket reload")
                await ws.send_json({
                    "type": "model_reloading",
                    "text": "TensorRT engine loading...",
                    "result": result,
                    "status": current_status(),
                })

            elif action == "save_settings":
                _, engine_changed = update_settings_from_payload(data)
                if should_reload_model(engine_changed):
                    reload_model("settings reload")
                with settings_lock:
                    saved_settings = dict(settings)
                await ws.send_json({
                    "type": "settings_saved",
                    "text": "설정 저장됨",
                    "settings": saved_settings,
                    "status": current_status(),
                })

            elif action == "get_status":
                await ws.send_json({
                    "type": "status",
                    "text": "status",
                    "status": current_status(),
                })

            else:
                await ws.send_json({
                    "type": "error",
                    "text": f"unknown action: {action}",
                    "status": current_status(),
                })

    except WebSocketDisconnect:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    print("서버 시작: http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
