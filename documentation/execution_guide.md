# Exact Implementation & Execution Steps

This guide outlines the steps to run the Audio Deepfake Detection project in a clean environment, as demonstrated in the Manus sandbox.

## 1. Environment Setup

### 1.1 Prerequisites
- Python 3.11+
- `pip` (Python package installer)
- `virtualenv` (recommended)

### 1.2 Installation Steps
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/gujjala-pranay/audio-deepfake-detection.git
    cd audio-deepfake-detection
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 2. Running the Services

### 2.1 Launch the FastAPI Backend
The backend handles the core logic and model inference.
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000`.*

### 2.2 Launch the Streamlit UI
The UI provides a user-friendly interface for uploading and analyzing audio.
```bash
streamlit run app.py --server.port 8501
```
*The UI will be available at `http://localhost:8501`.*

## 3. Using the Application

1.  **Open the UI**: Navigate to `http://localhost:8501` in your web browser.
2.  **Upload Audio**: Drag and drop or browse for a `.wav` or `.mp3` file.
3.  **View Analysis**:
    - The **Waveform** and **Mel-Spectrogram** will be generated automatically.
    - The **Prediction Result** (REAL or FAKE) will be displayed along with a **Confidence Score**.
4.  **Technical Insight**: Read the explanation provided in the UI to understand how the CNN identifies synthetic artifacts.

## 4. Troubleshooting
- **Permission Denied**: Ensure you are using a virtual environment or have the necessary permissions to install packages.
- **Port Already in Use**: If port 8000 or 8501 is busy, you can specify different ports using the `--port` flag for both `uvicorn` and `streamlit`.
- **Model Weights Missing**: If `model/deepfake_model.pth` is not found, the app will run with an uninitialized model for demonstration purposes. Ensure you have trained the model or provided the weights.
