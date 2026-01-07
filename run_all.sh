#!/bin/bash

# Function to handle cleanup on exit
cleanup() {
    echo "Stopping services..."
    kill $API_PID
    kill $UI_PID
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
API_PID=$!

echo "Starting Streamlit UI on port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > ui.log 2>&1 &
UI_PID=$!

echo "--------------------------------------------------"
echo "Both services are now running!"
echo "- API: http://localhost:8000"
echo "- UI:  http://localhost:8501"
echo "--------------------------------------------------"
echo "Logs are being written to api.log and ui.log"
echo "Press Ctrl+C to stop both services."

# Wait for background processes
wait
