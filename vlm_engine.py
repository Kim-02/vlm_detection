"""
TensorRT Qwen2.5-VL inference wrapper.

main.py intentionally depends only on:

    TensorRTQwenVL(engine_path).infer(image_bgr, prompt)

This wrapper uses the TensorRT-Edge-LLM runtime executable when available.
It does not require importing the low-level `tensorrt` Python module.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
from PIL import Image


class TensorRTQwenVL:
    def __init__(
        self,
        engine_path: str,
        llm_inference_bin: str | None = None,
        plugin_path: str | None = None,
    ):
        self.engine_path = str(engine_path)
        self.configured_llm_inference_bin = llm_inference_bin or ""
        self.configured_plugin_path = plugin_path or ""
        self.engine_dir: Path | None = None
        self.multimodal_engine_dir: Path | None = None
        self.llm_inference_bin: Path | None = None
        self.plugin_path: Path | None = None
        self.work_dir = Path(os.environ.get("EDGELLM_RUNTIME_DIR", "~/edgellm_work/runtime")).expanduser()
        self.system_prompt = os.environ.get(
            "EDGELLM_SYSTEM_PROMPT",
            "You are a vision-language assistant. Return only valid JSON.",
        )
        self.load_engine()

    def load_engine(self) -> None:
        root = Path(self.engine_path).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"TensorRT engine path does not exist: {root}")

        self.engine_dir, self.multimodal_engine_dir = self._resolve_engine_dirs(root)
        self.llm_inference_bin = self._resolve_llm_inference_bin()
        self.plugin_path = self._resolve_plugin_path()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def infer(self, image_bgr, prompt: str) -> dict[str, Any]:
        output_text = self._execute_qwen_vl(image_bgr, prompt)
        return self._parse_json(output_text)

    def _execute_qwen_vl(self, image_bgr, prompt: str) -> str:
        if self.engine_dir is None or self.multimodal_engine_dir is None:
            raise RuntimeError("TensorRT engine dirs are not initialized")
        if self.llm_inference_bin is None:
            raise RuntimeError("llm_inference binary is not initialized")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)

        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_path = tmpdir_path / "frame.jpg"
            input_path = tmpdir_path / "input.json"
            output_path = tmpdir_path / "output.json"

            image.save(image_path, quality=95)
            input_path.write_text(
                json.dumps(
                    self._build_input_payload(str(image_path), prompt),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            if self.plugin_path is not None:
                env["EDGELLM_PLUGIN_PATH"] = str(self.plugin_path)

            cmd = [
                str(self.llm_inference_bin),
                "--engineDir",
                str(self.engine_dir),
                "--multimodalEngineDir",
                str(self.multimodal_engine_dir),
                "--inputFile",
                str(input_path),
                "--outputFile",
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "TensorRT-Edge-LLM inference failed\n"
                    f"returncode={result.returncode}\n"
                    f"cmd={' '.join(cmd)}\n"
                    f"stdout=\n{result.stdout}\n"
                    f"stderr=\n{result.stderr}"
                )

            if not output_path.exists():
                raise RuntimeError(
                    "TensorRT-Edge-LLM did not create output file\n"
                    f"cmd={' '.join(cmd)}\n"
                    f"stdout=\n{result.stdout}\n"
                    f"stderr=\n{result.stderr}"
                )

            raw = output_path.read_text(encoding="utf-8")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return self._clean_output(raw)
        return self._clean_output(self._extract_text(parsed))

    def _build_input_payload(self, image_path: str, prompt: str) -> dict[str, Any]:
        return {
            "batch_size": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_generate_length": 256,
            "requests": [
                {
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ]
                }
            ],
        }

    def _resolve_engine_dirs(self, root: Path) -> tuple[Path, Path]:
        env_engine_dir = self._usable_env_path("EDGELLM_ENGINE_DIR")
        env_visual_dir = self._usable_env_path("EDGELLM_MULTIMODAL_ENGINE_DIR")
        if env_engine_dir and env_visual_dir:
            return self._require_dir(env_engine_dir, "EDGELLM_ENGINE_DIR"), self._require_dir(
                env_visual_dir,
                "EDGELLM_MULTIMODAL_ENGINE_DIR",
            )

        root = root.resolve()
        name = root.name
        candidates = [
            (
                root,
                root / "visual",
            ),
            (
                root,
                root / "visual_engine",
            ),
            (
                root,
                root / "visual_engines",
            ),
            (
                root / "llm",
                root / "visual",
            ),
            (
                root / "engine",
                root / "visual_engine",
            ),
            (
                root,
                root.parent.parent / "visual_engines" / name if root.parent.name == "engines" else root.parent / "visual_engines" / name,
            ),
            (
                root,
                root.parent.parent / "visual_engine" / name if root.parent.name == "engines" else root.parent / "visual_engine" / name,
            ),
            (
                root,
                root.parent.parent / "multimodal_engines" / name if root.parent.name == "engines" else root.parent / "multimodal_engines" / name,
            ),
        ]

        for engine_dir, visual_dir in candidates:
            if engine_dir.exists() and visual_dir.exists():
                return engine_dir, visual_dir

        raise FileNotFoundError(
            "TensorRT engine directories were not found.\n"
            f"configured engine_path={root}\n"
            "Expected one of:\n"
            f"- engine_dir={root}, multimodal_engine_dir={root}/visual\n"
            f"- engine_dir={root}, multimodal_engine_dir={root}/visual_engine\n"
            f"- engine_dir={root}, multimodal_engine_dir={root}/visual_engines\n"
            f"- engine_dir={root}, multimodal_engine_dir={root.parent.parent / 'visual_engines' / name if root.parent.name == 'engines' else root.parent / 'visual_engines' / name}\n"
            "Or set EDGELLM_ENGINE_DIR and EDGELLM_MULTIMODAL_ENGINE_DIR."
        )

    def _resolve_llm_inference_bin(self) -> Path:
        configured_path = self._usable_path_value(self.configured_llm_inference_bin)
        if configured_path:
            return self._require_file(configured_path, "llm_inference_bin")

        env_path = self._usable_env_path("EDGELLM_LLM_INFERENCE_BIN")
        if env_path:
            return self._require_file(env_path, "EDGELLM_LLM_INFERENCE_BIN")

        path_hit = shutil.which("llm_inference")
        if path_hit:
            return Path(path_hit)

        candidates = [
            "/home/ds/edge_llm/TensorRT-Edge-LLM/build/examples/llm/llm_inference",
            "~/TensorRT-Edge-LLM/build/examples/llm/llm_inference",
            "~/TensorRT-Edge-LLM/cpp/build/examples/llm/llm_inference",
            "~/TensorRT-Edge-LLM/build/bin/llm_inference",
            "~/TensorRT-Edge-LLM/build/llm_inference",
        ]
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.exists():
                return path

        raise FileNotFoundError(
            "llm_inference binary was not found. Build TensorRT-Edge-LLM or set "
            "EDGELLM_LLM_INFERENCE_BIN to the binary path."
        )

    def _resolve_plugin_path(self) -> Path | None:
        configured_path = self._usable_path_value(self.configured_plugin_path)
        if configured_path:
            return self._require_file(configured_path, "plugin_path")

        env_path = self._usable_env_path("EDGELLM_PLUGIN_PATH")
        if env_path:
            return self._require_file(env_path, "EDGELLM_PLUGIN_PATH")

        candidates = [
            "/home/ds/edge_llm/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so",
            "/home/ds/edge_llm/TensorRT-Edge-LLM/build/lib/libNvInfer_edgellm_plugin.so",
            "~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so",
            "~/TensorRT-Edge-LLM/build/lib/libNvInfer_edgellm_plugin.so",
            "~/TensorRT-Edge-LLM/cpp/build/libNvInfer_edgellm_plugin.so",
            "~/TensorRT-Edge-LLM/cpp/build/lib/libNvInfer_edgellm_plugin.so",
        ]
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.exists():
                return path
        return None

    def _usable_env_path(self, name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return self._usable_path_value(value)

    def _usable_path_value(self, value: str) -> Path | None:
        if not value or self._looks_like_placeholder(value):
            return None
        return Path(value).expanduser()

    def _looks_like_placeholder(self, value: str) -> bool:
        normalized = value.strip().lower()
        return (
            normalized.startswith("/path/to")
            or normalized.startswith("path/to")
            or normalized.startswith("/actual/path")
            or normalized.startswith("actual/path")
            or "실제/경로" in value
        )

    def _require_dir(self, path: Path, label: str) -> Path:
        if not path.is_dir():
            raise FileNotFoundError(f"{label} is not a directory: {path}")
        return path

    def _require_file(self, path: Path, label: str) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
        return path

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, dict):
            for key in ("text", "content", "output_text", "generated_text", "response", "message"):
                if key in data:
                    text = self._extract_text(data[key])
                    if text:
                        return text
            for value in data.values():
                text = self._extract_text(value)
                if text:
                    return text
        if isinstance(data, list):
            return "\n".join(filter(None, (self._extract_text(item) for item in data))).strip()
        return ""

    def _clean_output(self, text: str) -> str:
        return str(text).replace("Assistant:", "").replace("User:", "").strip()

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
