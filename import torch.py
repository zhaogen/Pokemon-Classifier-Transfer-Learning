import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

# 1. 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 폴더 경로를 본인의 환경에 맞게 확인하세요
data_dir = r"C:/Users/rlawh/OneDrive/바탕 화면/포케몬/pokemon_dataset/all_data" 

# 2. 전처리 (속도를 위해 128x128로 조절)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 데이터 로드 및 8:2 분할
full_dataset = datasets.ImageFolder(data_dir, transform=transform, allow_empty=True)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 클래스 이름 저장 (나중에 app.py에서 쓰기 위함)
class_names = full_dataset.classes
with open("classes.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# 4. 모델 설정 (ResNet18 Transfer Learning)
model = models.mobilenet_v2(weights='DEFAULT') # 미리 학습된 가중치 사용
# MobileNet V2는 'classifier'라는 이름의 마지막 층을 사용합니다.
num_ftrs = model.classifier[1].in_features 
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}\n' + '-'*10)
    for phase in ['train', 'val']:
        if phase == 'train': model.train()
        else: model.eval()

        running_loss, running_corrects = 0.0, 0
        loader = train_loader if phase == 'train' else val_loader

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / (train_size if phase == 'train' else val_size)
        print(f'{phase} Acc: {epoch_acc:.4f}')

# 6. 저장
torch.save(model.state_dict(), 'pokemon_model.pth')
print("모델 저장 완료: pokemon_model.pth")