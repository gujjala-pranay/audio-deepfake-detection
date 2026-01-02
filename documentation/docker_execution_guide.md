# Docker Execution Guide: Audio Deepfake Detection

This guide provides the exact, step-by-step process for running the entire Audio Deepfake Detection application (FastAPI backend and Streamlit UI) using Docker and `docker-compose`. This method is highly recommended to eliminate dependency and environment compatibility issues.

## Prerequisites

You must have **Docker** and **Docker Compose** installed on your system.

## Step 1: Clone the Repository

First, clone the project repository from GitHub and navigate into the project directory.

```bash
# 1. Clone the repository
git clone https://github.com/gujjala-pranay/audio-deepfake-detection.git

# 2. Navigate to the project directory
cd audio-deepfake-detection
```

## Step 2: Build and Run the Containers

The `docker-compose.yml` file is configured to build two services: `api` (FastAPI) and `ui` (Streamlit). The `--build` flag ensures that the images are built from the latest source code.

```bash
# Build the images and start the containers in detached mode (-d)
docker-compose up --build -d
```

### Expected Output:
You will see output indicating that the images are being built and the services are starting:
```
[+] Running 2/2
 ⠿ Network audio-deepfake-detection_default  Created
 ⠿ Container audio-deepfake-detection-api-1  Started
 ⠿ Container audio-deepfake-detection-ui-1   Started
```

## Step 3: Access the Application

Once the containers are running, you can access the application components via your web browser:

| Component | Access URL | Description |
| :--- | :--- | :--- |
| **Streamlit UI** | `http://localhost:8501` | The user interface for uploading audio and viewing predictions. |
| **FastAPI API** | `http://localhost:8000` | The backend service. You can view the interactive documentation (Swagger UI) at `http://localhost:8000/docs`. |

## Step 4: Verify Container Status (Optional)

You can check the status of the running containers to ensure they are healthy:

```bash
docker-compose ps
```

### Expected Output:
```
      Name                     Command               State           Ports
----------------------------------------------------------------------------------
...-api-1             uvicorn main:app --host ...   Up      0.0.0.0:8000->8000/tcp
...-ui-1              streamlit run app.py ...      Up      0.0.0.0:8501->8501/tcp
```

## Step 5: Stop and Clean Up

When you are finished using the application, you can stop and remove the containers, networks, and volumes created by `docker-compose`.

```bash
# Stop and remove containers, networks, and volumes
docker-compose down
```

This command will gracefully shut down the services and clean up your system, leaving only the downloaded source code.
