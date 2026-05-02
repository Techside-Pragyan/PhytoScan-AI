import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import random

app = FastAPI(title="PhytoScan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load disease info
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
disease_info_path = os.path.join(BASE_DIR, "disease_info.json")
with open(disease_info_path, "r") as f:
    disease_info = json.load(f)

CLASSES = list(disease_info.keys())

# Define image transforms matching the training process
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "plant_disease_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

try:
    if os.path.exists(MODEL_PATH):
        # Load MobileNetV2 architecture
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Using mock predictions for UI testing.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if model is not None:
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            predicted_class = CLASSES[predicted_idx.item()]
            confidence_score = confidence.item() * 100
        else:
            # Mock prediction if model is not trained yet
            predicted_class = random.choice(CLASSES)
            confidence_score = random.uniform(75.0, 99.9)

        info = disease_info[predicted_class]
        
        return {
            "success": True,
            "disease_name": info["name"],
            "confidence": f"{confidence_score:.2f}%",
            "description": info["description"],
            "remedy": info["remedy"]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Welcome to PhytoScan AI API"}
