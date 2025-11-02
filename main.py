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

# ---- Load Model ----
loaded_weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT
loaded_model = torchvision.models.efficientnet_b0(weights=None)
loaded_model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 6)
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "waste_classifier_model.pth")
loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
loaded_model.eval()

to_transform = loaded_weight.transforms()
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ---- App Setup ----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for testing/deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Routes ----
@app.get("/")
def home():
    return {"message": "✅ Waste Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image_data = Image.open(io.BytesIO(image_data)).convert("RGB")
    with torch.inference_mode():
        transformed_image = to_transform(image_data).unsqueeze(dim=0)
        target_image_pred = loaded_model(transformed_image)
    target_image_pred_prob = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_prob, dim=1)
    return JSONResponse({
        "filename": file.filename,
        "predicted_class": labels[target_image_pred_label],
        "confidence": round(target_image_pred_prob[0][target_image_pred_label].item(), 3)
    })

# ---- Run Server ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # <-- automatically picks Render’s PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
