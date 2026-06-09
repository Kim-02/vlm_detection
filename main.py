"""
RTSP CCTV prompt target tracker.

- FastAPI + native WebSocket
- MJPEG video stream via /video_feed
- Qwen2.5-VL bbox lookup in background threads
- OpenCV CSRT tracker for per-frame target tracking
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import cv2
import requests
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates


# FFMPEG low-latency options must be set before VideoCapture is created.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
)

templates = Jinja2Templates(directory="templates")

SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS: dict[str, Any] = {
    "model_path": "/media/ds/DATA/models/Qwen2.5-VL-3B",
    "vllm_url": "http://localhost:1111/v1",
    "api_key": "test-key",
    "rtsp_url": "rtsp://username:password@camera-ip:554/stream1",
    "capture_fps": 30,
    "verify_interval_sec": 3,
    "jpeg_quality": 80,
}
SETTINGS_KEYS = set(DEFAULT_SETTINGS)

settings_lock = threading.RLock()
settings: dict[str, Any] = dict(DEFAULT_SETTINGS)

latest_frame_lock = threading.RLock()
latest_frame = None

jpeg_lock = threading.RLock()
latest_jpeg: bytes | None = None

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
tracker = None
tracking_state: dict[str, Any] = {
    "active": False,
    "state": "idle",
    "prompt": "",
    "bbox": None,
    "label": "",
    "confidence": None,
    "last_verify_result": {"ok": None, "message": "not_verified", "time": None},
    "last_verify_at": 0.0,
    "locating": False,
    "verifying": False,
    "request_id": 0,
}

vlm_call_lock = threading.Lock()


def now_text() -> str:
    return datetime.now().strftime("%H:%M:%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def coerce_settings(data: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in SETTINGS_KEYS:
        if key in data:
            cleaned[key] = data[key]

    if "model_name" in data and "model_path" not in cleaned:
        cleaned["model_path"] = data["model_name"]

    if "rtsp_url" not in cleaned:
        old_keys = ("cam_ip", "cam_port", "cam_user", "cam_pw", "cam_path")
        if all(k in data for k in old_keys):
            user = str(data.get("cam_user") or "")
            password = str(data.get("cam_pw") or "")
            host = str(data.get("cam_ip") or "")
            port = str(data.get("cam_port") or "554")
            path = str(data.get("cam_path") or "")
            if path and not path.startswith("/"):
                path = "/" + path
            cred = f"{user}:{password}@" if user else ""
            cleaned["rtsp_url"] = f"rtsp://{cred}{host}:{port}{path}"

    for key in ("model_path", "vllm_url", "api_key", "rtsp_url"):
        if key in cleaned:
            cleaned[key] = str(cleaned[key]).strip()

    if "capture_fps" in cleaned:
        try:
            cleaned["capture_fps"] = max(1, min(60, int(float(cleaned["capture_fps"]))))
        except (TypeError, ValueError):
            cleaned["capture_fps"] = DEFAULT_SETTINGS["capture_fps"]

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


def update_settings_from_payload(data: dict[str, Any]) -> dict[str, Any]:
    cleaned = coerce_settings(data)
    with settings_lock:
        settings.update(cleaned)
        snapshot = dict(settings)
    save_settings()
    return snapshot


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
async def lifespan(_: FastAPI):
    manager._loop = asyncio.get_running_loop()
    load_settings()
    yield


app = FastAPI(lifespan=lifespan)


def chat_completions_url(vllm_url: str) -> str:
    base = vllm_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def encode_frame_base64(frame_bgr, quality: int = 90) -> str:
    ok, jpeg = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("failed to encode frame")
    return base64.b64encode(jpeg.tobytes()).decode("ascii")


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


def verify_prompt(user_prompt: str) -> str:
    return f"""이미지에는 현재 추적 중인 bbox 주변 crop이 들어 있다.
다음 조건에 해당하는 대상이 이미지 안에 아직 존재하는지 검증하라.

조건:
"{user_prompt}"

반드시 아래 JSON 형식으로만 답하라.
대상이 없거나 다른 대상이면 found=false로 답하라.

{{
  "found": true,
  "label": "target description",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0
}}

좌표는 입력 이미지 기준 pixel 좌표다.
설명 문장은 출력하지 마라."""


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


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def call_vlm_json(
    frame_bgr,
    prompt: str,
    require_bbox: bool = True,
) -> dict[str, Any]:
    with settings_lock:
        vllm_url = settings["vllm_url"]
        model_path = settings["model_path"]
        api_key = settings["api_key"]
        jpeg_quality = int(settings["jpeg_quality"])

    b64 = encode_frame_base64(frame_bgr, quality=jpeg_quality)
    payload = {
        "model": model_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(
        chat_completions_url(vllm_url),
        json=payload,
        headers=headers,
        timeout=(5, 60),
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    parsed = extract_json_object(str(content))
    return normalize_vlm_result(parsed, frame_bgr.shape, require_bbox=require_bbox)


def create_csrt_tracker():
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError(
        "OpenCV CSRT tracker is unavailable. Install opencv-contrib-python."
    )


def xyxy_to_xywh(bbox: list[int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> list[int]:
    x, y, w, h = [float(v) for v in box]
    return clamp_bbox_xyxy([x, y, x + w, y + h], width, height)


def get_latest_frame_copy():
    with latest_frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()


def tracking_snapshot() -> dict[str, Any]:
    with tracking_lock:
        return {
            "active": bool(tracking_state["active"]),
            "state": tracking_state["state"],
            "prompt": tracking_state["prompt"],
            "bbox": list(tracking_state["bbox"]) if tracking_state["bbox"] else None,
            "label": tracking_state["label"],
            "confidence": tracking_state["confidence"],
            "last_verify_result": dict(tracking_state["last_verify_result"]),
            "locating": bool(tracking_state["locating"]),
            "verifying": bool(tracking_state["verifying"]),
        }


def current_status() -> dict[str, Any]:
    with settings_lock:
        configured_rtsp_url = settings["rtsp_url"]
    with stream_lock:
        stream = {
            "running": bool(stream_state["running"]),
            "rtsp_url": stream_state["rtsp_url"] or configured_rtsp_url,
            "error": stream_state["error"],
        }
    track = tracking_snapshot()
    return {
        "stream": stream,
        "tracking": {
            "active": track["active"],
            "state": track["state"],
            "locating": track["locating"],
            "verifying": track["verifying"],
        },
        "prompt": track["prompt"],
        "bbox": track["bbox"],
        "label": track["label"],
        "confidence": track["confidence"],
        "last_verify_result": track["last_verify_result"],
    }


def emit_status(text: str, event_type: str = "status") -> None:
    manager.emit({"type": event_type, "text": text, "status": current_status()})


def stop_tracking_state(state: str = "idle", message: str = "tracking stopped") -> None:
    global tracker
    with tracking_lock:
        tracking_state["request_id"] += 1
        tracker = None
        tracking_state.update(
            {
                "active": False,
                "state": state,
                "bbox": None,
                "label": "",
                "confidence": None,
                "locating": False,
                "verifying": False,
                "last_verify_result": {
                    "ok": None,
                    "message": message,
                    "time": now_iso(),
                },
            }
        )


def draw_selected_bbox(frame_bgr):
    snapshot = tracking_snapshot()
    bbox = snapshot["bbox"]
    if snapshot["active"] and bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 80), 2)
    return frame_bgr


def publish_jpeg(frame_bgr) -> None:
    with settings_lock:
        quality = int(settings["jpeg_quality"])
    display = draw_selected_bbox(frame_bgr.copy())
    ok, jpeg = cv2.imencode(
        ".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if ok:
        with jpeg_lock:
            global latest_jpeg
            latest_jpeg = jpeg.tobytes()


def open_capture(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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


def maybe_start_verification(frame_bgr, bbox: list[int] | None) -> None:
    if not bbox:
        return
    with settings_lock:
        interval = float(settings["verify_interval_sec"])
    now = time.time()

    with tracking_lock:
        if (
            not tracking_state["active"]
            or tracking_state["state"] != "tracking"
            or tracking_state["verifying"]
        ):
            return
        if now - float(tracking_state["last_verify_at"]) < interval:
            return
        tracking_state["verifying"] = True
        tracking_state["last_verify_at"] = now
        prompt = tracking_state["prompt"]
        request_id = tracking_state["request_id"]

    crop = crop_bbox(frame_bgr, bbox)
    threading.Thread(
        target=verify_worker,
        args=(crop, prompt, request_id),
        daemon=True,
    ).start()
    emit_status("대상 검증 중...", "verify_start")


def verify_worker(frame_bgr, prompt: str, request_id: int) -> None:
    global tracker
    ok = False
    message = ""
    result: dict[str, Any] | None = None

    try:
        with vlm_call_lock:
            result = call_vlm_json(frame_bgr, verify_prompt(prompt), require_bbox=False)
        ok = bool(result["found"])
        message = "target_verified" if ok else "target_not_found"
    except Exception as exc:
        message = f"verify_error: {exc}"

    with tracking_lock:
        if request_id != tracking_state["request_id"]:
            return
        tracking_state["verifying"] = False
        tracking_state["last_verify_result"] = {
            "ok": ok,
            "message": message,
            "time": now_iso(),
            "result": result,
        }
        if ok:
            tracking_state["state"] = "tracking"
        else:
            tracker = None
            tracking_state["active"] = False
            tracking_state["state"] = "verify_failed"
            tracking_state["bbox"] = None

    if ok:
        emit_status("검증 성공", "verify_result")
    else:
        emit_status("검증 실패 / 재탐색 필요", "verify_failed")


def rtsp_capture_worker(rtsp_url: str, stop_event: threading.Event, generation: int) -> None:
    global latest_frame, tracker
    cap = open_capture(rtsp_url)
    reconnect_delay = 0.2

    while not stop_event.is_set():
        started = time.time()
        ret, frame = cap.read()
        if not ret:
            with stream_lock:
                if generation == stream_state["generation"]:
                    stream_state["error"] = "frame read failed; reconnecting"
            cap.release()
            if stop_event.wait(reconnect_delay):
                break
            cap = open_capture(rtsp_url)
            continue

        with stream_lock:
            if generation == stream_state["generation"]:
                stream_state["error"] = None

        with latest_frame_lock:
            latest_frame = frame.copy()

        current_bbox: list[int] | None = None
        tracker_failed = False
        with tracking_lock:
            if tracking_state["active"] and tracker is not None:
                try:
                    updated, box = tracker.update(frame)
                    if updated:
                        height, width = frame.shape[:2]
                        current_bbox = xywh_to_xyxy(box, width, height)
                except Exception as exc:
                    updated = False
                    tracking_state["last_verify_result"] = {
                        "ok": False,
                        "message": f"tracker_error: {exc}",
                        "time": now_iso(),
                    }

                if updated:
                    tracking_state["bbox"] = current_bbox
                    tracking_state["state"] = "tracking"
                else:
                    tracking_state["request_id"] += 1
                    tracker = None
                    tracking_state["active"] = False
                    tracking_state["state"] = "lost"
                    tracking_state["bbox"] = None
                    tracking_state["verifying"] = False
                    tracking_state["last_verify_result"] = {
                        "ok": False,
                        "message": "tracker_update_failed",
                        "time": now_iso(),
                    }
                    tracker_failed = True

        if current_bbox:
            maybe_start_verification(frame.copy(), current_bbox)
        elif tracker_failed:
            emit_status("트래커 실패 / 재탐색 필요", "tracking_failed")

        publish_jpeg(frame)

        with settings_lock:
            fps = max(1, int(settings["capture_fps"]))
        delay = max(0.0, (1.0 / fps) - (time.time() - started))
        if delay and stop_event.wait(delay):
            break

    cap.release()
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
    stop_tracking_state("idle", "stream stopped")


def start_tracking(prompt: str) -> tuple[bool, str]:
    global tracker
    prompt = prompt.strip()
    if not prompt:
        return False, "prompt is empty"

    with stream_lock:
        if not stream_state["running"]:
            return False, "stream is not running"

    frame = get_latest_frame_copy()
    if frame is None:
        return False, "latest frame is not ready"

    with tracking_lock:
        tracking_state["request_id"] += 1
        request_id = tracking_state["request_id"]
        tracker = None
        tracking_state.update(
            {
                "active": False,
                "state": "locating",
                "prompt": prompt,
                "bbox": None,
                "label": "",
                "confidence": None,
                "locating": True,
                "verifying": False,
                "last_verify_result": {
                    "ok": None,
                    "message": "locating_target",
                    "time": now_iso(),
                },
            }
        )

    threading.Thread(
        target=start_tracking_worker,
        args=(frame, prompt, request_id),
        daemon=True,
    ).start()
    return True, "locating target"


def start_tracking_worker(frame_bgr, prompt: str, request_id: int) -> None:
    global tracker
    emit_status(f"[{now_text()}] 대상 검색 중...", "tracking_start")
    try:
        with vlm_call_lock:
            result = call_vlm_json(frame_bgr, bbox_prompt(prompt), require_bbox=True)
    except Exception as exc:
        with tracking_lock:
            if request_id != tracking_state["request_id"]:
                return
            tracking_state.update(
                {
                    "active": False,
                    "state": "error",
                    "locating": False,
                    "bbox": None,
                    "last_verify_result": {
                        "ok": False,
                        "message": f"target_lookup_error: {exc}",
                        "time": now_iso(),
                    },
                }
            )
        emit_status(f"대상 검색 오류: {exc}", "tracking_failed")
        return

    if not result["found"]:
        with tracking_lock:
            if request_id != tracking_state["request_id"]:
                return
            tracking_state.update(
                {
                    "active": False,
                    "state": "not_found",
                    "locating": False,
                    "bbox": None,
                    "last_verify_result": {
                        "ok": False,
                        "message": "target_not_found",
                        "time": now_iso(),
                        "result": result,
                    },
                }
            )
        emit_status("대상을 찾지 못했습니다", "tracking_failed")
        return

    try:
        new_tracker = create_csrt_tracker()
        init_result = new_tracker.init(frame_bgr, xyxy_to_xywh(result["bbox"]))
        if init_result is False:
            raise RuntimeError("tracker.init returned false")
    except Exception as exc:
        with tracking_lock:
            if request_id != tracking_state["request_id"]:
                return
            tracking_state.update(
                {
                    "active": False,
                    "state": "error",
                    "locating": False,
                    "bbox": None,
                    "last_verify_result": {
                        "ok": False,
                        "message": f"tracker_init_error: {exc}",
                        "time": now_iso(),
                    },
                }
            )
        emit_status(f"트래커 시작 오류: {exc}", "tracking_failed")
        return

    with tracking_lock:
        if request_id != tracking_state["request_id"]:
            return
        tracker = new_tracker
        tracking_state.update(
            {
                "active": True,
                "state": "tracking",
                "bbox": result["bbox"],
                "label": result["label"],
                "confidence": result["confidence"],
                "locating": False,
                "verifying": False,
                "last_verify_at": time.time(),
                "last_verify_result": {
                    "ok": True,
                    "message": "target_found",
                    "time": now_iso(),
                    "result": result,
                },
            }
        )
    emit_status("추적 시작", "tracking_started")


def mjpeg_generator():
    while True:
        with jpeg_lock:
            data = latest_jpeg
        if data is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        time.sleep(1 / 30)


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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_json({"type": "status", "text": "connected", "status": current_status()})
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")

            if action == "start_stream":
                update_settings_from_payload(data)
                result = start_stream()
                await ws.send_json(
                    {
                        "type": "stream_started",
                        "text": "● 스트리밍 중",
                        "result": result,
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

            elif action == "save_settings":
                update_settings_from_payload(data)
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
