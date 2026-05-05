# 🐉 Pokemon Classifier (Transfer Learning)

## 1. 실험 환경 및 설정
- **Model**: ResNet18 (Pretrained)
- **Optimizer**: Adam (lr=0.001)
- **Dataset**: Kaggle Pokemon Dataset (8:2 Split)

## 2. 실험 결과 (Top-1 Accuracy)
| Experiment | Setting | Val Acc |
| :--- | :--- | :--- |
| **Exp 1** | **ResNet18 + Pretrained (Base)** | **92.06%** |
| Exp 2 | ResNet18 (No Pretrained) | 00.00% |
| Exp 3 | MobileNet V2 | 00.00% |
| Exp 4 | ResNet18 (lr=0.0001) | 00.00% |

## 3. 학습 곡선 (Learning Curve)
- (여기에 터미널 숫자 캡처본이나 그래프를 넣으세요)

## 4. 데모 결과
<img width="1920" height="1032" alt="image" src="https://github.com/user-attachments/assets/4a5ca3cf-c6cd-4460-ba33-43225f71496d" />
<img width="1920" height="1032" alt="image" src="https://github.com/user-attachments/assets/c8d462f6-ee3f-4105-98fb-d58974ac467a" />
