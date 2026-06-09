"""
RTSP CCTV prompt target tracker.

- FastAPI + native WebSocket
- MJPEG video stream via /video_feed
- TensorRT Qwen2.5-VL wrapper for bbox lookup and verification
- OpenCV CSRT tracker for per-frame target tracking
"""

from __future__ import annotations

import asyncio
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
    "max_targets": 12,
    "verify_interval_sec": 3,
    "jpeg_quality": 80,
    "startup_test_prompt": "이미지에 사람이 있는지 확인하고 있으면 bbox를 JSON으로 반환해줘.",
    "min_confidence": 0.3,
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

tracking_lock = threading.RLock()
trackers: dict[int, Any] = {}
next_target_id = 1
tracking_state: dict[str, Any] = {
    "active": False,
    "status": "idle",
    "prompt": "",
    "bbox": None,
    "targets": [],
    "label": "",
    "confidence": None,
    "last_verify_result": {
        "found": None,
        "label": "",
        "confidence": None,
        "message": "not_verified",
        "time": None,
    },
    "last_verify_at": 0.0,
    "locating": False,
    "detecting": False,
    "verifying": False,
    "request_id": 0,
}

vlm_call_lock = threading.Lock()


class TrackStartRequest(BaseModel):
    prompt: str


def now_text() -> str:
    return datetime.now().strftime("%H:%M:%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def coerce_settings(data: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in SETTINGS_KEYS:
        if key in data:
            cleaned[key] = data[key]

    for key in (
        "rtsp_url",
        "engine_path",
        "llm_inference_bin",
        "plugin_path",
        "startup_test_prompt",
    ):
        if key in cleaned:
            cleaned[key] = str(cleaned[key]).strip()

    if "capture_fps" in cleaned:
        try:
            cleaned["capture_fps"] = max(1, min(120, int(float(cleaned["capture_fps"]))))
        except (TypeError, ValueError):
            cleaned["capture_fps"] = DEFAULT_SETTINGS["capture_fps"]

    if "max_targets" in cleaned:
        try:
            cleaned["max_targets"] = max(1, min(12, int(float(cleaned["max_targets"]))))
        except (TypeError, ValueError):
            cleaned["max_targets"] = DEFAULT_SETTINGS["max_targets"]

    if "verify_interval_sec" in cleaned:
        try:
            cleaned["verify_interval_sec"] = max(
                1.0, float(cleaned["verify_interval_sec"])
            )
        except (TypeError, ValueError):
            cleaned["verify_interval_sec"] = DEFAULT_SETTINGS["verify_interval_sec"]

    if "jpeg_quality" in cleaned:
        try:
            cleaned["jpeg_quality"] = max(
                10, min(100, int(float(cleaned["jpeg_quality"])))
            )
        except (TypeError, ValueError):
            cleaned["jpeg_quality"] = DEFAULT_SETTINGS["jpeg_quality"]

    if "min_confidence" in cleaned:
        try:
            cleaned["min_confidence"] = max(0.0, min(1.0, float(cleaned["min_confidence"])))
        except (TypeError, ValueError):
            cleaned["min_confidence"] = DEFAULT_SETTINGS["min_confidence"]

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
    reload_model = (
        bool(cleaned.get("engine_path") and cleaned["engine_path"] != old_engine_path)
        or bool(
            cleaned.get("llm_inference_bin")
            and cleaned["llm_inference_bin"] != old_llm_inference_bin
        )
        or bool(cleaned.get("plugin_path") and cleaned["plugin_path"] != old_plugin_path)
        or bool(
            cleaned.get("startup_test_prompt")
            and cleaned["startup_test_prompt"] != old_startup_test_prompt
        )
    )
    return snapshot, reload_model


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


def set_model_state(**updates: Any) -> None:
    with model_lock:
        for key, value in updates.items():
            setattr(app.state, key, value)


def model_snapshot() -> dict[str, Any]:
    with settings_lock:
        configured_engine_path = settings["engine_path"]
        configured_llm_inference_bin = settings["llm_inference_bin"]
        configured_plugin_path = settings["plugin_path"]
    with model_lock:
        return {
            "model_status": getattr(app.state, "model_status", "loading"),
            "startup_test_status": getattr(app.state, "startup_test_status", "pending"),
            "engine_path": getattr(app.state, "engine_path", configured_engine_path),
            "llm_inference_bin": getattr(
                app.state,
                "llm_inference_bin",
                configured_llm_inference_bin,
            ),
            "plugin_path": getattr(app.state, "plugin_path", configured_plugin_path),
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
        args=(
            app_obj,
            generation,
            engine_path,
            startup_test_prompt,
            llm_inference_bin,
            plugin_path,
        ),
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
    stop_tracking_state("idle", reason)
    start_model_loading(app)
    return {
        "ok": True,
        "status": "model_reloading",
        "reason": reason,
        "app_status": current_status(),
    }


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
            test_frame = create_startup_test_image()

        with vlm_call_lock:
            raw_result = engine.infer(test_frame, bbox_prompt(startup_test_prompt))
        parsed = parse_engine_result(raw_result)
        if "found" not in parsed:
            raise ValueError("startup test result does not contain found field")

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


def create_startup_test_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (18, 24, 20)
    cv2.putText(
        image,
        "TensorRT Qwen2.5-VL startup test",
        (42, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (120, 210, 110),
        2,
        cv2.LINE_AA,
    )
    return image


def get_ready_engine() -> TensorRTQwenVL:
    with model_lock:
        model_status = getattr(app.state, "model_status", "loading")
        engine = getattr(app.state, "vlm_engine", None)
    if model_status != "ready" or engine is None:
        raise RuntimeError(f"model is not ready: {model_status}")
    return engine


def bbox_prompt(user_prompt: str) -> str:
    return f"""이미지에서 다음 조건에 해당하는 대상 1개를 찾아라.

조건:
"{user_prompt}"

반드시 아래 JSON 형식으로만 답하라.
대상이 없으면 found=false로 답하라.

{{
  "found": true,
  "label": "target description",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0
}}

좌표는 원본 이미지 기준 pixel 좌표다.
설명 문장은 출력하지 마라."""


def multi_target_prompt(user_prompt: str, max_targets: int) -> str:
    return f"""현재 CCTV 이미지에서 다음 조건에 해당하는 모든 대상을 찾아라.

조건:
"{user_prompt}"

최대 {max_targets}개까지만 반환하라.
각 대상은 화면에 표시할 중심점 point와 tracker 초기화에 사용할 대략적인 bbox를 포함해야 한다.

반드시 아래 JSON 형식으로만 답하라.
대상이 없으면 found=false, objects=[]로 답하라.

{{
  "found": true,
  "objects": [
    {{
      "label": "target description",
      "point": [cx, cy],
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0
    }}
  ]
}}

좌표는 원본 이미지 기준 pixel 좌표다.
설명 문장은 출력하지 마라."""


def verify_prompt(current_prompt: str) -> str:
    return f"""현재 추적 중인 대상이 이미지 안에 아직 존재하는지 확인해라.
조건:
"{current_prompt}"

반드시 JSON으로만 답해라.

{{
  "found": true,
  "label": "target description",
  "confidence": 0.0
}}"""


def verify_bbox_prompt(current_prompt: str) -> str:
    return f"""현재 추적 중인 대상이 이미지 안에 아직 존재하는지 확인하고, 존재한다면 정확한 위치(bbox)도 함께 반환해라.

조건:
"{current_prompt}"

반드시 아래 JSON 형식으로만 답해라.
대상이 있으면:
{{
  "found": true,
  "label": "target description",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0
}}

대상이 없으면:
{{
  "found": false
}}

좌표는 원본 이미지 기준 pixel 좌표다. 설명 문장은 출력하지 마라."""


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


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def clamp_bbox_xyxy(raw_bbox: Any, width: int, height: int) -> list[int]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    x1 = int(round(max(0, min(width - 1, x1))))
    y1 = int(round(max(0, min(height - 1, y1))))
    x2 = int(round(max(0, min(width, x2))))
    y2 = int(round(max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox has no area")
    return [x1, y1, x2, y2]


def normalize_vlm_result(
    data: dict[str, Any],
    frame_shape: tuple[int, int, int],
    require_bbox: bool,
) -> dict[str, Any]:
    height, width = frame_shape[:2]
    found = coerce_bool(data.get("found"))
    result: dict[str, Any] = {
        "found": found,
        "label": str(data.get("label") or ""),
        "bbox": None,
        "confidence": None,
        "raw": data,
    }
    try:
        result["confidence"] = float(data.get("confidence"))
    except (TypeError, ValueError):
        result["confidence"] = None

    if found and (require_bbox or data.get("bbox") is not None):
        result["bbox"] = clamp_bbox_xyxy(data.get("bbox"), width, height)
    if found and require_bbox and result["bbox"] is None:
        raise ValueError("found=true but bbox is missing")
    return result


def infer_vlm_json(frame_bgr, prompt: str, require_bbox: bool) -> dict[str, Any]:
    engine = get_ready_engine()
    with vlm_call_lock:
        raw_result = engine.infer(frame_bgr, prompt)
    parsed = parse_engine_result(raw_result)
    return normalize_vlm_result(parsed, frame_bgr.shape, require_bbox=require_bbox)


def create_csrt_tracker():
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError(
        "OpenCV CSRT tracker is unavailable. Install opencv-contrib-python-headless."
    )


def xyxy_to_xywh(bbox: list[int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> list[int]:
    x, y, w, h = [float(v) for v in box]
    return clamp_bbox_xyxy([x, y, x + w, y + h], width, height)


def bbox_center(bbox: list[int]) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))]


def clamp_point(raw_point: Any, width: int, height: int, fallback_bbox: list[int]) -> list[int]:
    if isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
        try:
            x = int(round(max(0, min(width - 1, float(raw_point[0])))))
            y = int(round(max(0, min(height - 1, float(raw_point[1])))))
            return [x, y]
        except (TypeError, ValueError):
            pass
    return bbox_center(fallback_bbox)


def normalize_vlm_objects(
    data: dict[str, Any],
    frame_shape: tuple[int, int, int],
    max_targets: int,
) -> list[dict[str, Any]]:
    height, width = frame_shape[:2]
    raw_objects = (
        data.get("objects")
        or data.get("targets")
        or data.get("detections")
        or data.get("items")
    )
    if raw_objects is None:
        if coerce_bool(data.get("found")):
            raw_objects = [data]
        else:
            raw_objects = []
    if not isinstance(raw_objects, list):
        raw_objects = []

    objects: list[dict[str, Any]] = []
    for raw in raw_objects:
        if not isinstance(raw, dict):
            continue
        try:
            bbox = clamp_bbox_xyxy(raw.get("bbox"), width, height)
        except (TypeError, ValueError):
            continue
        point = clamp_point(raw.get("point"), width, height, bbox)
        confidence = None
        try:
            if raw.get("confidence") is not None:
                confidence = float(raw.get("confidence"))
        except (TypeError, ValueError):
            confidence = None
        objects.append(
            {
                "label": str(raw.get("label") or data.get("label") or "target"),
                "bbox": bbox,
                "point": point,
                "confidence": confidence,
            }
        )
        if len(objects) >= max_targets:
            break
    return objects


def infer_vlm_objects(frame_bgr, prompt: str, max_targets: int) -> list[dict[str, Any]]:
    engine = get_ready_engine()
    with vlm_call_lock:
        raw_result = engine.infer(frame_bgr, prompt)
    parsed = parse_engine_result(raw_result)
    return normalize_vlm_objects(parsed, frame_bgr.shape, max_targets=max_targets)


def get_latest_frame_copy():
    with latest_frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()


def tracking_snapshot() -> dict[str, Any]:
    with tracking_lock:
        targets = [dict(target) for target in tracking_state["targets"]]
        first_target = targets[0] if targets else None
        return {
            "active": bool(tracking_state["active"]),
            "status": tracking_state["status"],
            "prompt": tracking_state["prompt"],
            "bbox": list(first_target["bbox"]) if first_target else None,
            "point": list(first_target["point"]) if first_target else None,
            "targets": targets,
            "points": [list(target["point"]) for target in targets],
            "target_count": len(targets),
            "label": first_target["label"] if first_target else tracking_state["label"],
            "confidence": first_target["confidence"] if first_target else tracking_state["confidence"],
            "last_verify_result": dict(tracking_state["last_verify_result"]),
            "locating": bool(tracking_state["locating"]),
            "detecting": bool(tracking_state["detecting"]),
            "verifying": bool(tracking_state["verifying"]),
        }


def current_status() -> dict[str, Any]:
    model = model_snapshot()
    with settings_lock:
        configured_rtsp_url = settings["rtsp_url"]
    with stream_lock:
        stream_running = bool(stream_state["running"])
        stream_error = stream_state["error"]
        rtsp_url = stream_state["rtsp_url"] or configured_rtsp_url
    track = tracking_snapshot()
    return {
        "model_status": model["model_status"],
        "startup_test_status": model["startup_test_status"],
        "engine_path": model["engine_path"],
        "llm_inference_bin": model["llm_inference_bin"],
        "plugin_path": model["plugin_path"],
        "startup_test_result": model["startup_test_result"],
        "stream_status": "running" if stream_running else "stopped",
        "tracking_status": track["status"],
        "tracking_busy": track["locating"] or track["detecting"] or track["verifying"],
        "current_prompt": track["prompt"],
        "bbox": track["bbox"],
        "point": track["point"],
        "targets": track["targets"],
        "points": track["points"],
        "target_count": track["target_count"],
        "label": track["label"],
        "confidence": track["confidence"],
        "last_verify_result": track["last_verify_result"],
        "last_error": model["last_error"] or stream_error,
        "stream": {
            "running": stream_running,
            "rtsp_url": rtsp_url,
            "error": stream_error,
        },
        "tracking": {
            "active": track["active"],
            "state": track["status"],
            "locating": track["locating"],
            "detecting": track["detecting"],
            "verifying": track["verifying"],
        },
    }


def emit_status(text: str, event_type: str = "status") -> None:
    manager.emit({"type": event_type, "text": text, "status": current_status()})


def stop_tracking_state(state: str = "idle", message: str = "tracking stopped") -> None:
    global trackers
    with tracking_lock:
        tracking_state["request_id"] += 1
        trackers.clear()
        tracking_state.update(
            {
                "active": False,
                "status": state,
                "bbox": None,
                "targets": [],
                "label": "",
                "confidence": None,
                "locating": False,
                "detecting": False,
                "verifying": False,
                "last_verify_result": {
                    "found": None,
                    "label": "",
                    "confidence": None,
                    "message": message,
                    "time": now_iso(),
                },
            }
        )


def draw_tracking_points(frame_bgr):
    snapshot = tracking_snapshot()
    if snapshot["active"]:
        for target in snapshot["targets"]:
            x, y = target["point"]
            cv2.circle(frame_bgr, (x, y), 7, (0, 255, 80), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (x, y), 12, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(
                frame_bgr,
                str(target["id"]),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 80),
                2,
                cv2.LINE_AA,
            )
    return frame_bgr


def publish_jpeg(frame_bgr) -> None:
    global latest_jpeg, latest_jpeg_seq
    with settings_lock:
        quality = int(settings["jpeg_quality"])
    display = draw_tracking_points(frame_bgr.copy())
    ok, jpeg = cv2.imencode(
        ".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
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


def crop_bbox(frame_bgr, bbox: list[int]):
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = max(8, int((x2 - x1) * 0.2))
    pad_y = max(8, int((y2 - y1) * 0.2))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr.copy()
    return frame_bgr[y1:y2, x1:x2].copy()


def maybe_start_verification(frame_bgr) -> None:
    with settings_lock:
        interval = float(settings["verify_interval_sec"])
    now = time.time()

    with tracking_lock:
        if (
            not tracking_state["active"]
            or tracking_state["status"] != "tracking"
            or tracking_state["verifying"]
            or not tracking_state["targets"]
        ):
            return
        if now - float(tracking_state["last_verify_at"]) < interval:
            return
        tracking_state["verifying"] = True
        tracking_state["last_verify_at"] = now
        prompt = tracking_state["prompt"]
        request_id = tracking_state["request_id"]

    threading.Thread(
        target=verify_worker,
        args=(frame_bgr.copy(), prompt, request_id),
        daemon=True,
    ).start()
    emit_status("대상 검증 중...", "verify_start")


def verify_worker(frame_bgr, prompt: str, request_id: int) -> None:
    global trackers
    found = False
    result: dict[str, Any] | None = None
    message = ""
    new_tracker = None
    new_bbox: list[int] | None = None

    try:
        result = infer_vlm_json(frame_bgr, verify_bbox_prompt(prompt), require_bbox=False)
        found = bool(result["found"])
        message = "target_verified" if found else "target_not_found"
        if found and result.get("bbox"):
            try:
                h, w = frame_bgr.shape[:2]
                new_bbox = clamp_bbox_xyxy(result["bbox"], w, h)
                new_tracker = create_tracker_for_bbox(frame_bgr, new_bbox)
            except Exception:
                new_bbox = None
                new_tracker = None
    except Exception as exc:
        message = f"verify_error: {exc}"

    with tracking_lock:
        if request_id != tracking_state["request_id"]:
            return
        tracking_state["verifying"] = False
        tracking_state["last_verify_result"] = {
            "found": found,
            "label": result["label"] if result else "",
            "confidence": result["confidence"] if result else None,
            "message": message,
            "time": now_iso(),
        }
        if found:
            tracking_state["status"] = "tracking"
            if new_tracker is not None and new_bbox is not None and tracking_state["targets"]:
                target_id = int(tracking_state["targets"][0]["id"])
                trackers[target_id] = new_tracker
                tracking_state["targets"][0].update({
                    "bbox": new_bbox,
                    "point": bbox_center(new_bbox),
                    "source": "verify",
                    "last_seen": now_iso(),
                    "last_seen_ts": time.time(),
                    "vlm_miss_count": 0,
                })
                tracking_state["bbox"] = new_bbox
        else:
            trackers.clear()
            tracking_state["active"] = False
            tracking_state["status"] = "verify_failed"
            tracking_state["bbox"] = None
            tracking_state["targets"] = []

    if found:
        emit_status("검증 성공", "verify_result")
    else:
        emit_status("검증 실패 / 재탐색 필요", "verify_failed")


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


def new_target_record(
    target_id: int,
    detection: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    now = time.time()
    return {
        "id": target_id,
        "label": detection["label"],
        "bbox": list(detection["bbox"]),
        "point": list(detection["point"]),
        "confidence": detection["confidence"],
        "source": source,
        "last_seen": now_iso(),
        "last_seen_ts": now,
        "vlm_miss_count": 0,
    }


def create_tracker_for_bbox(frame, bbox: list[int]):
    tracker_obj = create_csrt_tracker()
    init_result = tracker_obj.init(frame, xyxy_to_xywh(bbox))
    if init_result is False:
        raise RuntimeError("tracker.init returned false")
    return tracker_obj


def center_distance(point_a: list[int], point_b: list[int]) -> float:
    return float(((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5)


def iou_xyxy(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def match_detection_to_target(
    detection: dict[str, Any],
    targets: list[dict[str, Any]],
    used_ids: set[int],
    frame_shape: tuple[int, int, int],
) -> int | None:
    height, width = frame_shape[:2]
    distance_limit = max(36.0, min(width, height) * 0.08)
    best_id = None
    best_score = -1.0
    for target in targets:
        target_id = int(target["id"])
        if target_id in used_ids:
            continue
        dist = center_distance(detection["point"], target["point"])
        overlap = iou_xyxy(detection["bbox"], target["bbox"])
        if dist > distance_limit and overlap < 0.1:
            continue
        score = overlap * 3.0 + max(0.0, 1.0 - (dist / distance_limit))
        if score > best_score:
            best_score = score
            best_id = target_id
    return best_id


def apply_vlm_detections(
    frame,
    detections: list[dict[str, Any]],
    request_id: int,
) -> None:
    global next_target_id
    VLM_MAX_MISS = 3

    with settings_lock:
        max_targets = int(settings["max_targets"])

    # Snapshot existing targets for matching (lock released during tracker init)
    with tracking_lock:
        if request_id != tracking_state["request_id"] or not tracking_state["active"]:
            return
        existing_snapshot = [dict(t) for t in tracking_state["targets"]]

    # Match detections to existing targets and build trackers (no lock held)
    pending: list[tuple[int | None, dict[str, Any], Any]] = []
    pre_used: set[int] = set()
    errors: list[str] = []

    for detection in detections[:max_targets]:
        matched_id = match_detection_to_target(detection, existing_snapshot, pre_used, frame.shape)
        if matched_id is not None:
            pre_used.add(matched_id)
        try:
            tracker_obj = create_tracker_for_bbox(frame, detection["bbox"])
        except Exception as exc:
            errors.append(str(exc))
            if matched_id is not None:
                pre_used.discard(matched_id)
            continue
        pending.append((matched_id, detection, tracker_obj))

    # Apply all changes in a single lock acquisition
    with tracking_lock:
        if request_id != tracking_state["request_id"] or not tracking_state["active"]:
            return

        working: list[dict[str, Any]] = [dict(t) for t in tracking_state["targets"]]
        confirmed_ids: set[int] = set()
        created = 0
        updated = 0

        for matched_id, detection, tracker_obj in pending:
            if matched_id is None:
                if len(working) >= max_targets:
                    continue
                target_id = next_target_id
                next_target_id += 1
                record = new_target_record(target_id, detection, "vlm")
                working.append(record)
                trackers[target_id] = tracker_obj
                confirmed_ids.add(target_id)
                created += 1
            else:
                for i, t in enumerate(working):
                    if int(t["id"]) == matched_id:
                        record = new_target_record(matched_id, detection, "vlm")
                        working[i] = record
                        trackers[matched_id] = tracker_obj
                        confirmed_ids.add(matched_id)
                        updated += 1
                        break

        # Prune targets not confirmed by this VLM scan
        pruned = 0
        survived: list[dict[str, Any]] = []
        for t in working:
            tid = int(t["id"])
            if tid in confirmed_ids:
                t["vlm_miss_count"] = 0
                survived.append(t)
            else:
                miss = t.get("vlm_miss_count", 0) + 1
                t["vlm_miss_count"] = miss
                if miss < VLM_MAX_MISS:
                    survived.append(t)
                else:
                    trackers.pop(tid, None)
                    pruned += 1

        if len(survived) > max_targets:
            for t in survived[max_targets:]:
                trackers.pop(int(t["id"]), None)
            survived = survived[:max_targets]

        tracking_state["targets"] = survived
        tracking_state["bbox"] = list(survived[0]["bbox"]) if survived else None
        tracking_state["label"] = survived[0]["label"] if survived else ""
        tracking_state["confidence"] = survived[0]["confidence"] if survived else None
        tracking_state["status"] = "tracking" if survived else "searching"
        tracking_state["locating"] = not bool(survived)

        msg = f"vlm_detected={len(detections)}, updated={updated}, created={created}"
        if pruned:
            msg += f", pruned={pruned}"
        if errors:
            msg += f", errors={len(errors)}"
        tracking_state["last_verify_result"] = {
            "found": bool(survived),
            "label": f"{len(survived)} target(s)",
            "confidence": None,
            "message": msg,
            "time": now_iso(),
        }


def process_stream_frame(frame) -> None:
    global trackers
    tracker_failed = False
    with tracking_lock:
        if tracking_state["active"] and trackers:
            next_targets: list[dict[str, Any]] = []
            failed_ids: list[int] = []
            targets_by_id = {int(target["id"]): target for target in tracking_state["targets"]}
            tracker_items = list(trackers.items())
            height, width = frame.shape[:2]
            for target_id, tracker_obj in tracker_items:
                target = targets_by_id.get(int(target_id))
                if target is None:
                    failed_ids.append(int(target_id))
                    continue
                updated = False
                box = None
                try:
                    updated, box = tracker_obj.update(frame)
                except Exception as exc:
                    tracking_state["last_verify_result"] = {
                        "found": False,
                        "label": "",
                        "confidence": None,
                        "message": f"tracker_error: {exc}",
                        "time": now_iso(),
                    }

                if updated and box is not None:
                    bbox = xywh_to_xyxy(box, width, height)
                    target.update(
                        {
                            "bbox": bbox,
                            "point": bbox_center(bbox),
                            "source": "tracker",
                            "last_seen": now_iso(),
                            "last_seen_ts": time.time(),
                        }
                    )
                    next_targets.append(target)
                else:
                    failed_ids.append(int(target_id))
                    tracker_failed = True

            for target_id in failed_ids:
                trackers.pop(target_id, None)

            tracking_state["targets"] = next_targets
            tracking_state["bbox"] = list(next_targets[0]["bbox"]) if next_targets else None
            tracking_state["label"] = next_targets[0]["label"] if next_targets else ""
            tracking_state["confidence"] = next_targets[0]["confidence"] if next_targets else None
            tracking_state["status"] = "tracking" if next_targets else "searching"
            tracking_state["locating"] = not bool(next_targets)
            if tracker_failed and not next_targets:
                tracking_state["last_verify_result"] = {
                    "found": False,
                    "label": "",
                    "confidence": None,
                    "message": "all trackers lost; continuing latest-frame search",
                    "time": now_iso(),
                }

    publish_jpeg(frame)
    maybe_start_verification(frame)


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
        stream_state.update(
            {
                "running": True,
                "error": None,
                "rtsp_url": rtsp_url,
                "stop_event": stop_event,
            }
        )
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
    stop_tracking_state("idle", "stream stopped")


def prepare_tracking_session(prompt: str) -> tuple[bool, str, int | None]:
    global trackers
    prompt = prompt.strip()
    if not prompt:
        return False, "prompt is empty", None

    model = model_snapshot()
    if model["model_status"] != "ready":
        return False, f"model is not ready: {model['model_status']}", None

    with stream_lock:
        if not stream_state["running"]:
            return False, "stream is not running", None

    with tracking_lock:
        tracking_state["request_id"] += 1
        request_id = tracking_state["request_id"]
        trackers.clear()
        tracking_state.update(
            {
                "active": True,
                "status": "searching",
                "prompt": prompt,
                "bbox": None,
                "targets": [],
                "label": "",
                "confidence": None,
                "locating": True,
                "detecting": False,
                "verifying": False,
                "last_verify_result": {
                    "found": None,
                    "label": "",
                    "confidence": None,
                    "message": "continuous_latest_frame_search",
                    "time": now_iso(),
                },
            }
        )
    return True, "continuous latest-frame search started", request_id


def start_tracking(prompt: str) -> tuple[bool, str]:
    ok, message, request_id = prepare_tracking_session(prompt)
    if not ok:
        return False, message
    threading.Thread(
        target=continuous_detect_worker,
        args=(prompt.strip(), request_id),
        daemon=True,
    ).start()
    emit_status(f"[{now_text()}] 최신 프레임 연속 탐색 시작", "tracking_start")
    return True, message


def continuous_detect_worker(prompt: str, request_id: int) -> None:
    while True:
        with tracking_lock:
            if request_id != tracking_state["request_id"] or not tracking_state["active"]:
                return
            tracking_state["detecting"] = True

        frame = get_latest_frame_copy()
        if frame is None:
            with tracking_lock:
                if request_id == tracking_state["request_id"]:
                    tracking_state["detecting"] = False
                    tracking_state["locating"] = True
            time.sleep(0.05)
            continue

        with settings_lock:
            max_targets = int(settings["max_targets"])
            min_conf = float(settings.get("min_confidence", 0.3))

        try:
            prompt_text = multi_target_prompt(prompt, max_targets)
            detections = infer_vlm_objects(frame, prompt_text, max_targets=max_targets)
            if min_conf > 0:
                detections = [
                    d for d in detections
                    if d["confidence"] is None or d["confidence"] >= min_conf
                ]
            apply_vlm_detections(frame, detections, request_id)
            with tracking_lock:
                if request_id == tracking_state["request_id"]:
                    tracking_state["detecting"] = False
            emit_status(
                f"최신 프레임 탐색: {len(detections)}개 후보",
                "tracking_update",
            )
        except Exception as exc:
            should_sleep = True
            with tracking_lock:
                if request_id != tracking_state["request_id"] or not tracking_state["active"]:
                    return
                tracking_state["detecting"] = False
                tracking_state["locating"] = not bool(tracking_state["targets"])
                tracking_state["status"] = "tracking" if tracking_state["targets"] else "searching"
                tracking_state["last_verify_result"] = {
                    "found": False,
                    "label": "",
                    "confidence": None,
                    "message": f"latest_frame_detect_error: {exc}",
                    "time": now_iso(),
                }
            emit_status(f"최신 프레임 탐색 오류: {exc}", "tracking_failed")
            if should_sleep:
                time.sleep(0.5)


def start_tracking_and_wait(prompt: str) -> dict[str, Any]:
    ok, message, request_id = prepare_tracking_session(prompt)
    if not ok:
        return {"ok": False, "status": "rejected", "message": message}
    threading.Thread(
        target=continuous_detect_worker,
        args=(prompt.strip(), request_id),
        daemon=True,
    ).start()
    return {
        "ok": True,
        "status": "continuous_search_started",
        "message": message,
        "request_id": request_id,
    }


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


@app.post("/api/track/start")
def api_track_start(payload: TrackStartRequest):
    return JSONResponse(start_tracking_and_wait(payload.prompt))


@app.post("/api/track/stop")
def api_track_stop():
    stop_tracking_state("idle", "tracking stopped by REST")
    return JSONResponse({"ok": True, "status": "tracking_stopped"})


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
                    reload_model_result = reload_model("stream start reload")
                else:
                    reload_model_result = None
                result = start_stream()
                await ws.send_json(
                    {
                        "type": "stream_started",
                        "text": "● 스트리밍 중",
                        "result": result,
                        "model_reload": reload_model_result,
                        "status": current_status(),
                    }
                )

            elif action == "stop_stream":
                stop_stream()
                await ws.send_json(
                    {"type": "stream_stopped", "text": "● 중지됨", "status": current_status()}
                )

            elif action == "start_tracking":
                ok, message = start_tracking(str(data.get("prompt") or ""))
                await ws.send_json(
                    {
                        "type": "tracking_request",
                        "ok": ok,
                        "text": message,
                        "status": current_status(),
                    }
                )

            elif action == "stop_tracking":
                stop_tracking_state("idle", "tracking stopped by user")
                await ws.send_json(
                    {
                        "type": "tracking_stopped",
                        "text": "추적 중지됨",
                        "status": current_status(),
                    }
                )

            elif action == "reload_model":
                result = reload_model("manual WebSocket reload")
                await ws.send_json(
                    {
                        "type": "model_reloading",
                        "text": "TensorRT engine loading...",
                        "result": result,
                        "status": current_status(),
                    }
                )

            elif action == "save_settings":
                _, engine_changed = update_settings_from_payload(data)
                if should_reload_model(engine_changed):
                    reload_model("settings reload")
                with settings_lock:
                    saved_settings = dict(settings)
                await ws.send_json(
                    {
                        "type": "settings_saved",
                        "text": "설정 저장됨",
                        "settings": saved_settings,
                        "status": current_status(),
                    }
                )

            elif action == "get_status":
                await ws.send_json(
                    {"type": "status", "text": "status", "status": current_status()}
                )

            else:
                await ws.send_json(
                    {
                        "type": "error",
                        "text": f"unknown action: {action}",
                        "status": current_status(),
                    }
                )

    except WebSocketDisconnect:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn

    print("서버 시작: http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
