from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn
import io
import os

loaded_weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT
loaded_model = torchvision.models.efficientnet_b0(weights = None)

loaded_model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280,6)
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "waste_classifier_model.pth")
loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
loaded_model.eval()

to_transform = loaded_weight.transforms()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image_data = Image.open(io.BytesIO(image_data)).convert("RGB")
    with torch.inference_mode():
        transformed_image = to_transform(image_data).unsqueeze(dim=0)
        traget_image_pred = loaded_model(transformed_image)

    traget_image_pred_prob = torch.softmax(traget_image_pred,dim = 1)
    traget_image_pred_label = torch.argmax(traget_image_pred_prob, dim=1)

    return JSONResponse({
        "filename" : file.filename,
        "predicted_class" : labels[traget_image_pred_label],
        "confidence" : round(traget_image_pred_prob[0][traget_image_pred_label].item(),3)
    })