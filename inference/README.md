# KV-AI — 키비주얼 생성 시스템

## 현재 구조

SDXL + ControlNet + IP-Adapter + LoRA + FastAPI 기반의 키비주얼 자동 생성 서버.

```
inference/
 ├── app/
 │    ├── __init__.py
 │    ├── main.py                # FastAPI 서버 (POST /generate)
 │    ├── pipeline.py            # SDXL Base + Refiner + ControlNet + IP-Adapter + LoRA
 │    └── element_extractor.py   # SAM 기반 요소 추출 + 패턴 생성
 ├── models/
 │    ├── sdxl/sd_xl_base/       # SDXL Base (8.4GB)
 │    ├── sdxl/sd_xl_refiner/    # SDXL Refiner (5.9GB)
 │    ├── controlnet/canny/      # ControlNet Canny (2.7GB)
 │    ├── controlnet/depth/      # ControlNet Depth (2.7GB)
 │    ├── ip_adapter/            # IP-Adapter (4.6GB)
 │    ├── sam/                   # SAM ViT-H (2.4GB)
 │    └── lora/                  # LoRA 가중치 저장 위치
 ├── download_models.py          # 모델 일괄 다운로드
 └── requirements.txt
```

### 현재 API (v1)

`POST /generate`

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| prompt | form | 필수 | 생성 프롬프트 |
| product_image | file | 필수 | IP-Adapter 입력 (제품 정체성) |
| composition_image | file | product_image 대체 | ControlNet 입력 (구도 참고) |
| negative_prompt | form | "low quality, blurry, distorted" |
| controlnet_scale | form | 0.5 |
| ip_adapter_scale | form | 0.6 |
| num_inference_steps | form | 30 |
| use_pattern | form | false | SAM 요소 추출 → 타일 패턴 → ControlNet |
| pattern_threshold_ratio | form | 0.15 | SAM 요소 면적 기준 |
| pattern_spacing | form | 50 | 타일 간 간격(px) |

### 현재 실행 상태

- [x] SDXL Base + Refiner 로컬 다운로드 완료
- [x] ControlNet (Canny, Depth) 로컬 다운로드 완료
- [x] IP-Adapter 로컬 다운로드 완료
- [x] SAM ViT-H 로컬 다운로드 완료
- [x] txt2img + ControlNet + IP-Adapter + LoRA 파이프라인 개발 완료
- [x] SAM 요소 추출 → 타일 패턴 → ControlNet 연결 완료
- [ ] LoRA 학습 데이터 준비 중
- [ ] 첫 LoRA 학습 (kohya_ss)

---

## 개선 방향 (논의 중)

### 문제 인식

- ControlNet + IP-Adapter + LoRA 3개 동시 사용 시 scale 튜닝이 어렵고 조합이 불안정함
- ControlNet이 edge를 강제해서 프롬프트 의도와 충돌 가능
- 각 모듈(ControlNet, IP-Adapter, LoRA)이 생성 결과에 어떤 영향을 주는지 예측이 어려움

### 제안 아키텍처 (v2)

```
브랜드 로고 → SAM(브랜드별 fine-tuned) → 분해된 요소들 → SDXL img2img + LoRA → 출력
```

- **ControlNet/IP-Adapter 제거** → 파이프라인 단순화
- **txt2img → img2img 전환** → 분해된 요소를 초기 이미지(latent)로 주입
- **SAM 브랜드별 fine-tuning** → 일관된 요소 분해
- **LoRA** → 순수 브랜드 스타일만 학습 (원본 KV 이미지로만 학습, 분해 요소는 포함하지 않음)

| 구성 요소 | 역할 |
|---|---|
| SAM (브랜드별 fine-tuned) | 브랜드 요소를 일관되게 분해 |
| 분해된 요소 → img2img 초기 이미지 | SDXL에 브랜드 요소 직접 주입 |
| LoRA (KV 이미지로만 학습) | 브랜드 고유 스타일 입히기 |
| denoising_strength | 원본 요소 보존 vs 생성 자유도 조절 (유일한 컨트롤) |

### 장점

- 단순함: 로딩 시간↓, VRAM↓, 튜닝할 파라미터↓
- 예측 가능: 초기 이미지가 있으므로 결과 방향 제어 용이
- 브랜드 정체성 유지: SAM이 요소를 직접 분해해서 주입하므로 요소 소실 위험 없음

### 리스크 및 검증 필요 사항

1. **브랜드별 SAM fine-tuning 데이터 부족**: 브랜드당 충분한 segmentation mask 데이터 확보 필요
2. **분해 요소 합성**: 여러 요소를 하나의 img2img 초기 이미지로 합성하는 방식 정의 필요 (배치, 크기, 배경)
3. **denoising_strength 단일 컨트롤**: 유일한 조절 변수, 각 브랜드/요소별 최적값 탐색 필요
4. **LoRA 학습 데이터 분리**: LoRA는 완성형 KV 이미지만 학습, 분해 요소는 학습 데이터에서 제외

### 검증 순서 (제안)

1. txt2img → img2img로 파이프라인만 전환 (SAM 없이, 제품 이미지를 초기 입력으로)
2. LoRA 학습 없이 denoising_strength만 튜닝해서 효과 확인
3. SAM 분해 및 주입 로직 추가
4. 브랜드별 SAM fine-tuning
