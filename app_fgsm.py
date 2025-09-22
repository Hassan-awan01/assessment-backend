from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PIL import Image
import torch
from torchvision import transforms
from mangum import Mangum
from model import load_model
from fgsm import Attack

app = FastAPI(title="MNIST FGSM Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

handler = Mangum(app)
MODEL_PATH = "./mnist-classifier.pt"
DEVICE = "cpu"

model = load_model(MODEL_PATH, device=DEVICE)
attack = Attack(model=model, device=DEVICE)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class AttackResponse(BaseModel):
    clean_prediction: int
    adversarial_prediction: int
    adversarial_image_b64: str
    attack_success: bool
    epsilon: float

def pil_to_input_tensor(pil_image: Image.Image):
    return transform(pil_image).unsqueeze(0)

@app.post("/attack", response_model=AttackResponse)
async def attack_endpoint(file: UploadFile = File(...), epsilon: float = Form(0.1)):
    if epsilon < 0 or epsilon > 1.0:
        raise HTTPException(status_code=400, detail="epsilon must be between 0 and 1.0")

    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    input_tensor = pil_to_input_tensor(pil).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, clean_pred = torch.max(outputs.data, 1)
        clean_pred_item = int(clean_pred.item())

    true_label = torch.tensor([clean_pred_item], dtype=torch.long)
    adv_tensor, _ = attack.fgsm(input_tensor.clone(), true_label, epsilon=epsilon)

    with torch.no_grad():
        outputs_adv = model(adv_tensor)
        _, adv_pred = torch.max(outputs_adv.data, 1)
        adv_pred_item = int(adv_pred.item())

    adv_b64 = attack.tensor_to_base64_image(
        adv_tensor, unnormalize_mean=0.5, unnormalize_std=0.5
    )

    success = adv_pred_item != clean_pred_item

    return AttackResponse(
        clean_prediction=clean_pred_item,
        adversarial_prediction=adv_pred_item,
        adversarial_image_b64=adv_b64,
        attack_success=success,
        epsilon=epsilon
    )

@app.get("/ping")
async def ping():
    return {"status": "ok"}
