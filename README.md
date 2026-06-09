# TensorRT Qwen2.5-VL RTSP Tracker

FastAPI + RTSP MJPEG 화면 + TensorRT Qwen2.5-VL bbox 추론 + OpenCV CSRT tracker 기반 대상 추적 서버입니다.

## 실행 전

1. `settings.json`에서 `engine_path`를 실제 TensorRT engine 경로로 수정합니다.
2. `settings.json`에서 `rtsp_url`을 CCTV RTSP 주소로 수정합니다.
3. Jetson/TensorRT 환경에 맞게 TensorRT-Edge-LLM 런타임 바이너리와 plugin을 준비합니다.

`vlm_engine.py`는 기본적으로 `~/TensorRT-Edge-LLM/build/examples/llm/llm_inference`를 찾습니다.
다른 위치에 있으면 환경변수를 지정합니다.

```bash
export EDGELLM_LLM_INFERENCE_BIN=/path/to/llm_inference
export EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so
```

`engine_path`가 LLM engine 디렉터리라면 visual engine은 다음 위치를 자동으로 찾습니다.

```text
<engine_path>/visual
<engine_path>/visual_engine
<engine_path>/visual_engines
<engine_path의 상위>/visual_engines/<engine_path 이름>
```

자동 탐색과 다른 구조라면 아래 환경변수로 직접 지정합니다.

```bash
export EDGELLM_ENGINE_DIR=/path/to/llm_engine
export EDGELLM_MULTIMODAL_ENGINE_DIR=/path/to/visual_engine
```

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
python3 main.py
```

브라우저에서 `http://127.0.0.1:5000`으로 접속합니다.

## 동작 흐름

1. 서버 시작
2. TensorRT engine 로딩
3. startup test 수행
4. 성공 시 웹 화면 표시
5. 프롬프트 입력
6. TensorRT Qwen2.5-VL이 대상 bbox 추론
7. CSRT tracker가 bbox 유지
8. 주기적으로 TensorRT Qwen2.5-VL이 대상 존재 여부 검증

## TensorRT wrapper

`main.py`는 TensorRT 세부 구현에 의존하지 않고 아래 호출만 사용합니다.

```python
app.state.vlm_engine.infer(image_bgr, prompt)
```

실제 Qwen2.5-VL 전처리, TensorRT binding, tokenizer decode, JSON 후처리는 `vlm_engine.py` 안에서 engine build에 맞게 연결하면 됩니다.
