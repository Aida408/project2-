import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ===============================
# 1️⃣ Определяем архитектуру модели (должна совпадать с обученной)
# ===============================
class BetterNet(nn.Module):
    def __init__(self):
        super(BetterNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ===============================
# 2️⃣ Загрузка модели
# ===============================
@st.cache_resource
def load_model():
    model = BetterNet()
    model.load_state_dict(torch.load("improved_cifar10.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ===============================
# 3️⃣ Настройка интерфейса
# ===============================
st.title("🧠 Классификация изображений CIFAR-10")
st.write("Загрузите изображение (32x32 или больше), и модель определит, к какому классу оно относится.")

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

uploaded_file = st.file_uploader("📁 Загрузите изображение...", type=["jpg", "jpeg", "png"])

# ===============================
# 4️⃣ Обработка и предсказание
# ===============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_t = transform(image)
    img_t = img_t.unsqueeze(0)  # добавляем batch dimension

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]

    st.markdown(f"### 🎯 Предсказанный класс: **{class_name.upper()}**")

