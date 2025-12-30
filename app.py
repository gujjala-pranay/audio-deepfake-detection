import streamlit as st
import torch
import numpy as np
import librosa
import os
from src.model import DeepfakeDetectorCNN
from utils import plot_spectrogram, plot_waveform
import torch.nn.functional as F

# Page Configuration
st.set_page_config(page_title="Audio Deepfake Detector", page_icon="🎙️", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    model = DeepfakeDetectorCNN()
    model_path = 'model/deepfake_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        # Create a dummy model for demonstration if weights don't exist
        st.warning("Model weights not found. Using an uninitialized model for demonstration.")
    model.eval()
    return model

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

# UI Header
st.title("🎙️ Audio Deepfake Detection System")
st.markdown("""
This application uses a **2D Convolutional Neural Network (CNN)** to distinguish between **Real (Bona fide)** and **Fake (Deepfake)** audio.
Upload an audio file to see the analysis.
""")

# Sidebar
st.sidebar.header("About the Project")
st.sidebar.info("""
- **Model**: 2D CNN
- **Features**: Mel-Spectrograms
- **Dataset**: ASVspoof 2019
- **Goal**: Detect synthetic speech and voice conversion.
""")

# File Upload
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio Visualization")
        st.audio(uploaded_file, format='audio/wav')
        
        # Waveform
        fig_wave = plot_waveform(temp_path)
        st.pyplot(fig_wave)
        
        # Spectrogram
        fig_spec = plot_spectrogram(temp_path)
        st.pyplot(fig_spec)
        
    with col2:
        st.subheader("Prediction Result")
        model = load_model()
        
        with st.spinner('Analyzing audio...'):
            input_tensor = preprocess_audio(temp_path)
            # Ensure input tensor matches model expected size (1, 1, 128, 126)
            # If T is different, we might need to interpolate or pad
            if input_tensor.shape[3] != 126:
                input_tensor = F.interpolate(input_tensor, size=(128, 126), mode='bilinear', align_corners=False)
                
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                
            # Display Result
            if prediction.item() == 1:
                st.success("✅ RESULT: REAL (Bona fide)")
            else:
                st.error("❌ RESULT: FAKE (Deepfake)")
                
            st.metric("Confidence Score", f"{confidence.item()*100:.2f}%")
            
            # Progress bar for confidence
            st.progress(confidence.item())
            
            st.markdown("---")
            st.write("**Technical Explanation:**")
            st.write("""
            The model analyzes the **Mel-Spectrogram** of the audio. Synthetic audio often leaves subtle artifacts in the frequency domain that are invisible to the human ear but detectable by CNNs. 
            The 2D CNN treats the spectrogram as an image, identifying patterns characteristic of TTS or VC algorithms.
            """)

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
else:
    st.info("Please upload an audio file to begin analysis.")
