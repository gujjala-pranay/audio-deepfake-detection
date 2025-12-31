# API and Docker Setup Guide

This guide provides instructions for setting up the Audio Deepfake Detection system as a robust, containerized application using **FastAPI** for the API and **Docker** for environment consistency.

## 1. FastAPI Backend Implementation

The model is served via a RESTful API using FastAPI, a modern, fast web framework for Python. This decouples the model from the frontend, allowing for scalability and easy integration with other services.

### 1.1 API Endpoints

| Endpoint | Method | Description | Request Body | Response Body |
| :--- | :--- | :--- | :--- | :--- |
| `/` | `GET` | Health check and root message. | None | `{"message": "..."}` |
| `/predict` | `POST` | Uploads an audio file and returns the deepfake prediction. | `file`: Audio file (`.wav`, `.mp3`) | `{"filename": "...", "prediction": "REAL/FAKE", "confidence": 0.99, "status": "success"}` |

### 1.2 Example API Usage (using `curl`)

Once the Docker container is running on `localhost:8000`, you can test the prediction endpoint:

```bash
# Replace /path/to/your/audio.wav with a real audio file path
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/audio.wav;type=audio/wav"
```

## 2. Docker and Containerization

Docker ensures that the application runs identically regardless of the host environment, solving the "it works on my machine" problem, especially critical for deep learning projects with complex dependencies like `librosa` and `torch`.

### 2.1 Prerequisites

- **Docker** and **Docker Compose** installed on your system.

### 2.2 Dockerfile Overview

The `Dockerfile` is based on a slim Python image and includes necessary system libraries (`libsndfile1`, `ffmpeg`) required by `librosa` for audio processing.

### 2.3 Docker Compose Setup

The `docker-compose.yml` file defines two services:

1.  **`api`**: Runs the FastAPI application using `uvicorn` on port `8000`.
2.  **`ui`**: Runs the Streamlit application (`app.py`) on port `8501`.

This setup allows both the API and the UI to run simultaneously and independently within their own containers, sharing the same consistent environment.

### 2.4 Running the Application with Docker Compose

1.  **Build and Run Containers**: Execute the following command in the project root directory:

    ```bash
    docker-compose up --build
    ```

    This command builds the Docker image (if not already built) and starts both the `api` and `ui` services.

2.  **Access Services**:
    - **Deepfake API**: Accessible at `http://localhost:8000`
    - **Streamlit UI**: Accessible at `http://localhost:8501`

3.  **Stop Containers**: To stop and remove the containers:

    ```bash
    docker-compose down
    ```

## 3. Benefits of Containerization

| Feature | Benefit |
| :--- | :--- |
| **Reproducibility** | Guarantees the exact same environment (OS, Python, libraries) for every deployment. |
| **Isolation** | Prevents conflicts between project dependencies and the host system's libraries. |
| **Scalability** | The API service can be easily scaled up by increasing the number of replicas in the Docker Compose file or using a container orchestrator like Kubernetes. |
| **Simplified Deployment** | Deployment to cloud platforms (AWS, Azure, GCP) becomes a single, standardized process. |
