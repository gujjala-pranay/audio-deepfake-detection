from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import librosa
import numpy as np
import os
import shutil
from src.model import DeepfakeDetectorCNN
import torch.nn.functional as F

app = FastAPI(title="Audio Deepfake Detection API")

# Load Model
model = DeepfakeDetectorCNN()
model_path = 'model/deepfake_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def preprocess_audio(audio_path, max_len=64000):
    audio, sr = librosa.load(audio_path, sr=16000)
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)), 'constant')
    else:
        audio = audio[:max_len]
    
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    # Add batch and channel dims: (1, 1, 128, T)
    spectrogram_db = np.expand_dims(spectrogram_db, axis=(0, 1)) 
    return torch.FloatTensor(spectrogram_db)

@app.get("/")
async def root():
    return {"message": "Audio Deepfake Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .wav and .mp3 are supported.")
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        input_tensor = preprocess_audio(temp_path)
        # Ensure input tensor matches model expected size (1, 1, 128, 126)
        if input_tensor.shape[3] != 126:
            input_tensor = F.interpolate(input_tensor, size=(128, 126), mode='bilinear', align_corners=False)
            
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        result = {
            "filename": file.filename,
            "prediction": "REAL" if prediction.item() == 1 else "FAKE",
            "confidence": float(confidence.item()),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
