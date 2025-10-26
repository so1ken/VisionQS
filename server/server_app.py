import os
import time
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ------------------------------
# Conv Autoencoder (должен совпадать с train.py)
# ------------------------------
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

# ------------------------------
# App init and config
# ------------------------------
app = FastAPI(title="VisionQC Server")

MODEL_PATH = 'server/model/model.pth'
THRESHOLD_PATH = 'server/model/threshold.txt'
IMG_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ensure folders exist
os.makedirs('server/model', exist_ok=True)
os.makedirs('server/logs', exist_ok=True)

# demo / real mode selection
demo_mode = True
model = None
threshold = None

if os.path.exists(MODEL_PATH) and os.path.exists(THRESHOLD_PATH):
    try:
        model = ConvAutoencoder().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        with open(THRESHOLD_PATH, 'r') as f:
            threshold = float(f.read().strip())
        demo_mode = False
        print(f"[server] Loaded model from {MODEL_PATH}, threshold={threshold:.6g}. Device={DEVICE}")
    except Exception as e:
        demo_mode = True
        print(f"[server] Failed to load model/threshold (falling back to demo mode): {e}")
else:
    print("[server] model.pth or threshold.txt not found -> DEMO MODE. Incoming images will be saved to server/logs/")

# ------------------------------
# Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def preprocess_image(file_like) -> torch.Tensor:
    """file_like: file object or BytesIO"""
    img = Image.open(file_like).convert('RGB')
    img = transform(img)  # [C,H,W], float32, 0..1
    img = img.unsqueeze(0).to(DEVICE)  # [1,C,H,W]
    return img

def compute_score(img_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        recon = model(img_tensor)
        mse = nn.functional.mse_loss(recon, img_tensor, reduction='mean').item()
    return mse

# ------------------------------
# Endpoint
# ------------------------------
@app.post('/analyze')
async def analyze_image(image: UploadFile = File(...)):
    """
    Accepts multipart/form-data with field 'image'.
    In demo mode: saves the incoming file to server/logs/ and returns acknowledgment.
    In real mode: runs preprocess -> model -> compute_score and returns {'result','score'}.
    """
    try:
        # read all bytes
        raw = await image.read()
        # save incoming file for debugging
        ts = int(time.time())
        safe_name = f"recv_{ts}.jpg"
        saved_path = os.path.join('server', 'logs', safe_name)
        with open(saved_path, 'wb') as wf:
            wf.write(raw)

        if demo_mode:
            return JSONResponse(content={'result': 'Received', 'note': 'demo-mode', 'saved': saved_path})

        # real inference
        buf = BytesIO(raw)
        img_tensor = preprocess_image(buf)
        score = compute_score(img_tensor)
        result = 'Good' if score <= threshold else 'Bad'
        return JSONResponse(content={'result': result, 'score': score, 'saved': saved_path})
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

# ------------------------------
# Health/check endpoint (optional)
# ------------------------------
@app.get('/health')
def health():
    return {'status': 'ok', 'mode': 'demo' if demo_mode else 'real', 'device': str(DEVICE)}

# ------------------------------
# Run server
# ------------------------------
if __name__ == '__main__':
    # uvicorn will print logs and serve OpenAPI at /docs
    print("[server] Starting uvicorn on 0.0.0.0:8000 ...")
    uvicorn.run(app, host='0.0.0.0', port=8000)
