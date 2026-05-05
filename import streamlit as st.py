import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. 클래스 이름 불러오기
try:
    with open("classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except:
    class_names = ["학습을 먼저 진행해주세요"]

# 2. 모델 로드 설정
device = torch.device("cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('pokemon_model.pth', map_location=device))
model.eval()

# 3. 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Streamlit UI
st.title("Homework #6: Pokemon Classification")
uploaded_file = st.file_uploader("포켓몬 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_container_width=True)
    
    # 예측
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_idx = torch.topk(probs, 5)

    st.subheader("Top-5 Predictions:")
    for i in range(5):
        st.write(f"{i+1}. {class_names[top5_idx[i]]} ({top5_prob[i]*100:.2f}%)")