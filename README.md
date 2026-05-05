본 프로젝트는 **전이 학습(Transfer Learning)**을 활용하여 7,000장의 포켓몬 이미지를 분류하는 딥러닝 모델을 구축하고, 모델 간의 성능 차이를 비교 분석한 과제입니다.

## 1. 실험 환경 (Environment)
- **Framework**: PyTorch
- **Device**: NVIDIA CUDA (GPU 사용) / CPU
- **Dataset**: Kaggle Pokemon Dataset (약 7,000개 이미지)
- **Data Split**: Train 80%, Validation 20%
- **Input Size**: 128x128 (Resize)

## 2. 실험 결과 비교 (Experiment Results)

교수님께서 제시하신 가이드라인에 따라 4가지 다른 설정으로 실험을 진행하였습니다.

| 실험 번호 | 설정 (Setting) | Val Acc (최종) | 분석 및 통찰 |
| :--- | :--- | :--- | :--- |
| **Exp 1** | **ResNet18 + Pretrained (Base)** | **92.06%** | 가장 높은 정확도. 기존 학습된 가중치가 포켓몬 분류에도 효과적임을 증명. |
| **Exp 2** | **ResNet18 (No Pretrained)** | **69.84%** | 가중치 없이 처음부터 학습하여 성능이 약 20% 하락함. 데이터셋 대비 학습량이 부족함. |
| **Exp 3** | **MobileNet V2 (Model Change)** | **89.42%** | 경량화 모델로 학습 속도는 빠르나, ResNet18 대비 특징 추출 성능이 낮음. |
| **Exp 4** | **ResNet18 + lr=0.0001 (LR Down)** | **88.42%** | 학습률을 낮추어 더 세밀하게 학습했으나, 10 Epoch 내에서는 Base 모델보다 수렴이 느림. |

## 3. 학습 곡선 (Learning Curves)
- **Exp 1 (Best Model)**: 10 Epoch 내에 빠르게 90% 이상의 정확도에 도달함.
- **Exp 2**: Loss가 매우 천천히 감소하며, 전이 학습의 중요성을 확인하는 대조군 역할을 함.

## 4. 데모 실행 화면 (Streamlit Application)
Streamlit을 활용하여 실제 포켓몬 이미지를 업로드하고 예측 결과를 확인했습니다.

<img width="849" height="878" alt="스크린샷 2026-05-05 175319" src="https://github.com/user-attachments/assets/03753bde-e6ac-47ac-8d20-9642cae4ea8a" />

<img width="799" height="823" alt="스크린샷 2026-05-05 175330" src="https://github.com/user-attachments/assets/dc562b1f-64b6-4ed3-b477-b3b84d2bfe47" />

### ✅ 주요 기능
- 이미지 업로드 시 실시간 분류 (Top-5 결과 출력)
