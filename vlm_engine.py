"""
TensorRT Qwen2.5-VL inference wrapper.

main.py intentionally depends only on:

    TensorRTQwenVL(engine_path).infer(image_bgr, prompt)

The TensorRT/Qwen2.5-VL preprocessing, CUDA buffer binding, tokenizer decode,
and JSON postprocessing details vary by engine build. Keep those details here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TensorRTQwenVL:
    def __init__(self, engine_path: str):
        self.engine_path = str(engine_path)
        self.engine_file: Path | None = None
        self.trt = None
        self.logger = None
        self.runtime = None
        self.engine = None
        self.load_engine()

    def load_engine(self) -> None:
        path = Path(self.engine_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"TensorRT engine path does not exist: {path}")

        self.engine_file = self._resolve_engine_file(path)
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT Python runtime is not installed. Install TensorRT for "
                "your Jetson/CUDA environment, then restart the server."
            ) from exc

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(self.engine_file, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.engine_file}")

    def infer(self, image_bgr, prompt: str) -> dict[str, Any]:
        """
        Run one Qwen2.5-VL inference.

        Args:
            image_bgr: OpenCV BGR frame.
            prompt: Fully wrapped prompt from main.py.

        Returns:
            A JSON-compatible dict. For bbox lookup:
            {
              "found": true,
              "label": "target",
              "bbox": [x1, y1, x2, y2],
              "confidence": 0.0
            }

            For verification, bbox may be omitted.
        """
        output_text = self._execute_qwen_vl(image_bgr, prompt)
        return self._parse_json(output_text)

    def _execute_qwen_vl(self, image_bgr, prompt: str) -> str:
        raise NotImplementedError(
            "TensorRT engine deserialization is wired, but this project's "
            "Qwen2.5-VL preprocessing, CUDA bindings, and tokenizer decode must "
            "be connected in vlm_engine.py for the specific engine build."
        )

    def _resolve_engine_file(self, path: Path) -> Path:
        if path.is_file():
            return path
        engine_files = sorted(path.glob("*.engine"))
        if not engine_files:
            raise FileNotFoundError(f"no .engine file found under: {path}")
        return engine_files[0]

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = str(text).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(cleaned[start : end + 1])
