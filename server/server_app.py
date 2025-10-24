
import os
# FastAPI основной объект сервера / File, UploadFile — для обработки файлов, пришедших через POST
from fastapi import FastAPI, File, UploadFile
# JSONResponse — способ возвращать JSON
from fastapi.responses import JSONResponse
# uvicorn — сервер для запуска FastAPI
import uvicorn
#torch, nn — PyTorch для модели
import torch
import torch.nn as nn
# transforms — предобработка изображений
from torchvision import transforms
# PIL.Image — загрузка и конвертация изображений в RGB
from PIL import Image


# Хз списал с гпт попозже разберусь
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out


# Инициализация FastAPI
app = FastAPI()

# MODEL_PATH — путь к обученной модели
MODEL_PATH = 'server/model/model.pth'
# THRESHOLD_PATH — путь к файлу порога
THRESHOLD_PATH = 'server/model/threshold.txt'
# IMG_SIZE — размер, к которому ресайзятся изображения (должен совпадать с тренировкой)
IMG_SIZE = 128
# DEVICE — выбираем GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model.eval() — переводим модель в режим инференса
model = ConvAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Короче момент какой, сейчас распишу поподробнее, наша модель учится на фото из train, и сравнивает их с фото из val
# Для чего это надо, у нас и там, и там, фотографии с качественным продуктом, не дефектным, но фото то отличаются
# И модель сверяя их определяет параметр threshold - параметр разброса, больше которого модель будет считать что на фото анамалия (тоесть дефект)
# Этот параметр она сохраняет в файл, и мы его берем
with open(THRESHOLD_PATH, 'r') as f:
    threshold = float(f.read())

'''
transforms.Compose([...]) — создаёт цепочку преобразований, которые будут применяться последовательно к изображению.
transforms.Resize((IMG_SIZE, IMG_SIZE))
    Меняет размер изображения на 128x128 пикселей (значение IMG_SIZE = 128, как у нас).
    Модель была обучена на картинках именно этого размера.
    Если не сделать ресайз, PyTorch выдаст ошибку о несоответствии размера входа.

transforms.ToTensor()
    Конвертирует PIL Image или NumPy массив в PyTorch тензор.
    Результат — тензор с форматом [C, H, W]:
    C = каналы (3 для RGB)
    H = высота
    W = ширина

Автоматически нормализует пиксели к диапазону [0.0, 1.0] (раньше пиксели были 0–255).
'''
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

'''
img = Image.open(file).convert('RGB')
    file — это объект UploadFile из FastAPI. Его можно передать PIL напрямую.
    Image.open(file) — открываем файл как изображение.
    .convert('RGB') — превращает изображение в RGB (3 канала).
    PNG может быть с альфа-каналом (4 канала)
    JPEG может быть grayscale (1 канал)
    Модель ожидает 3 канала, поэтому конвертируем.

img = transform(img)
    Применяем цепочку трансформаций, которую описали выше:
    Resize → 128x128
    ToTensor → тензор [3,128,128] с диапазоном [0,1] - > 128 это пока затычка, потом возьму данные своей камеры

img = img.unsqueeze(0).to(DEVICE)
    unsqueeze(0) — добавляем batch dimension.
    PyTorch модели всегда ожидают тензор с батчем [batch_size, channels, H, W].
    Мы анализируем одно изображение → делаем [1, 3, 128, 128].
    .to(DEVICE) — переносим тензор на CPU или GPU, в зависимости от доступности CUDA.

return img
    Возвращаем тензор, готовый для подачи в модель на inference.
    Формат: [1,3,128,128], тип torch.float32, значения пикселей в [0,1].




'''
def preprocess_image(file) -> torch.Tensor:
    img = Image.open(file).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)  # batch dimension
    return img

# Честно хз че это, сложно
def compute_score(img_tensor):
    with torch.no_grad():
        recon = model(img_tensor)
        mse = nn.functional.mse_loss(recon, img_tensor, reduction='mean').item()
    return mse


# Эндпоинт /analyze
@app.post('/analyze')

# async def analyze_image(image: UploadFile = File(...)):
# async позволяет обрабатывать несколько запросов одновременно (асинхронно).
# image: UploadFile = File(...) — FastAPI автоматически парсит файл из запроса multipart/form-data.
async def analyze_image(image: UploadFile = File(...)):
    try:
        img_tensor = preprocess_image(image.file) # Вызываем функцию, о которой говорили ранее
        score = compute_score(img_tensor) # Хз
        result = 'Good' if score <= threshold else 'Bad'
        return JSONResponse(content={'result': result, 'score': score})
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

# Запуск сервера
# ------------------------------
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

    
