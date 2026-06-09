# TensorRT Qwen2.5-VL RTSP Tracker

FastAPI + RTSP MJPEG 화면 + TensorRT Qwen2.5-VL bbox 추론 + OpenCV CSRT tracker 기반 대상 추적 서버입니다.

## 실행 전

1. `settings.json`에서 `engine_path`를 실제 TensorRT engine 경로로 수정합니다.
2. `settings.json`에서 `rtsp_url`을 CCTV RTSP 주소로 수정합니다.
3. Jetson/TensorRT 환경에 맞게 TensorRT Python runtime, CUDA Python 또는 pycuda 등 engine 실행에 필요한 시스템 패키지를 설치합니다.

TensorRT, pycuda, cuda-python은 Jetson/CUDA/TensorRT 버전에 따라 설치 방식이 달라서 `requirements.txt`에 고정하지 않았습니다.

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
