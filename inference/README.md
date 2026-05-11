# KV-AI — 키비주얼 생성 AI 시스템

SDXL + ControlNet + IP-Adapter + FastAPI 기반의 키비주얼 자동 생성 서버입니다.

## 프로젝트 구조

```
kv-ai/
 ├── app/
 │    ├── __init__.py
 │    ├── main.py          # FastAPI 서버
 │    ├── pipeline.py      # SDXL Base + Refiner 생성 파이프라인
 │    ├── controlnet.py    # ControlNet (Canny) 모듈
 │    ├── ipadapter.py     # IP-Adapter (placeholder)
 │    └── utils.py         # 전처리 유틸 (Canny 변환 등)
 ├── models/               # 로컬 모델 캐시
 ├── requirements.txt
 └── README.md
```

## 실행 방법

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API 사용

`POST /generate`

- `prompt`: (form) 생성 프롬프트
- `image`: (file) 입력 이미지

→ `output.png` 저장 + JSON 응답

## 다음 단계

- [ ] ControlNet 실제 연결
- [ ] IP-Adapter 적용 (제품/얼굴 유지)
- [ ] LoRA (브랜드 스타일)
- [ ] GPU 최적화 (속도 2~3배 개선)
- [ ] 키비주얼 전용 프롬프트 엔진
