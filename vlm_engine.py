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

    def infer(self, images_bgr, prompt: str) -> dict[str, Any]:
        if isinstance(images_bgr, list):
            frames = images_bgr
        else:
            frames = [images_bgr]
        output_text = self._execute_qwen_vl(frames, prompt)
        return self._parse_json(output_text)

    def _execute_qwen_vl(self, images_bgr: list, prompt: str) -> str:
        if self.engine_dir is None or self.multimodal_engine_dir is None:
            raise RuntimeError("TensorRT engine dirs are not initialized")
        if self.llm_inference_bin is None:
            raise RuntimeError("llm_inference binary is not initialized")

        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.json"
            output_path = tmpdir_path / "output.json"

            image_paths: list[str] = []
            for idx, image_bgr in enumerate(images_bgr):
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
                image_path = tmpdir_path / f"frame_{idx:03d}.jpg"
                image.save(image_path, quality=90)
                image_paths.append(str(image_path))

            input_path.write_text(
                json.dumps(
                    self._build_input_payload(image_paths, prompt),
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
            output_text = self._clean_output(raw)
        else:
            answer_json = self._extract_answer_json(parsed)
            if answer_json is not None:
                output_text = json.dumps(answer_json, ensure_ascii=False)
            else:
                output_text = self._clean_output(self._extract_text(parsed))

        if not output_text:
            raise RuntimeError(
                "TensorRT-Edge-LLM returned no generated text\n"
                f"cmd={' '.join(cmd)}\n"
                f"raw_output=\n{self._clip(raw)}\n"
                f"stdout=\n{self._clip(result.stdout)}\n"
                f"stderr=\n{self._clip(result.stderr)}"
            )
        return output_text

    def _build_input_payload(self, image_paths: list[str], prompt: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": p} for p in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        return {
            "batch_size": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_generate_length": 512,
            "requests": [
                {
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content},
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
        for candidate in self._iter_generated_text_candidates(data):
            text = self._clean_output(candidate)
            if self._is_generated_text_candidate(text):
                return text
        return ""

    def _extract_answer_json(self, data: Any) -> dict[str, Any] | None:
        if isinstance(data, dict):
            if "found" in data or "objects" in data:
                return data
            for key, value in data.items():
                if key in {"input", "request", "requests", "prompt"}:
                    continue
                found = self._extract_answer_json(value)
                if found is not None:
                    return found
        if isinstance(data, list):
            for item in data:
                found = self._extract_answer_json(item)
                if found is not None:
                    return found
        return None

    def _iter_generated_text_candidates(
        self,
        data: Any,
        response_context: bool = False,
    ):
        if isinstance(data, str):
            if response_context or self._looks_like_json_text(data):
                yield data
            return

        if isinstance(data, list):
            for item in data:
                yield from self._iter_generated_text_candidates(item, response_context)
            return

        if not isinstance(data, dict):
            return

        role = str(data.get("role") or "").strip().lower()
        content_type = str(data.get("type") or "").strip().lower()
        if role in {"system", "user"}:
            return
        if content_type in {"image", "text"} and not response_context:
            return

        if role in {"assistant", "model"}:
            response_context = True

        response_keys = (
            "output_text",
            "generated_text",
            "generated",
            "response",
            "responses",
            "answer",
            "answers",
            "completion",
            "completions",
            "prediction",
            "predictions",
            "result",
            "results",
            "output",
            "outputs",
            "choices",
            "message",
            "messages",
            "content",
            "text",
        )
        for key in response_keys:
            if key in data:
                yield from self._iter_generated_text_candidates(data[key], True)

        metadata_keys = {
            "batch_size",
            "engine_dir",
            "engineDir",
            "file",
            "filename",
            "image",
            "image_path",
            "imagePath",
            "input",
            "input_file",
            "inputFile",
            "input_path",
            "inputPath",
            "multimodalEngineDir",
            "path",
            "prompt",
            "request",
            "requests",
            "role",
            "system_prompt",
            "type",
        }
        for key, value in data.items():
            if key in response_keys or key in metadata_keys:
                continue
            yield from self._iter_generated_text_candidates(value, response_context)

    def _looks_like_json_text(self, text: str) -> bool:
        cleaned = str(text).strip()
        return (
            cleaned.startswith("{")
            or cleaned.startswith("```")
            or ('"found"' in cleaned and "{" in cleaned and "}" in cleaned)
        )

    def _is_generated_text_candidate(self, text: str) -> bool:
        cleaned = str(text).strip()
        if not cleaned:
            return False
        if self._looks_like_runtime_path(cleaned):
            return False
        return True

    def _looks_like_runtime_path(self, text: str) -> bool:
        cleaned = str(text).strip()
        if "\n" in cleaned:
            return False
        suffixes = (
            ".json",
            ".jpg",
            ".jpeg",
            ".png",
            ".engine",
            ".safetensors",
        )
        if cleaned.startswith(("/", "~/")) and (
            "/" in cleaned or cleaned.endswith(suffixes)
        ):
            return True
        return cleaned.endswith(("/input.json", "/output.json", "/frame.jpg"))

    def _clean_output(self, text: str) -> str:
        return str(text).replace("Assistant:", "").replace("User:", "").strip()

    def _clip(self, value: Any, limit: int = 1600) -> str:
        text = "" if value is None else str(value)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}... <truncated {len(text) - limit} chars>"

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = str(text).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        if not cleaned:
            raise ValueError("Model output was empty; expected bbox JSON")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(
                    f"Model output was not valid JSON: {self._clip(cleaned)}"
                ) from exc
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as inner_exc:
                raise ValueError(
                    f"Model output contained malformed JSON: {self._clip(cleaned)}"
                ) from inner_exc
