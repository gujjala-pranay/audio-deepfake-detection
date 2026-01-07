@echo off
echo Starting FastAPI backend and Streamlit UI...

:: Activate virtual environment if it exists
if exist venv\Scripts\activate (
    call venv\Scripts\activate
)

:: Start FastAPI in a new minimized window
start /min "FastAPI Backend" uvicorn main:app --host 0.0.0.0 --port 8000

:: Start Streamlit in the current window
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo Both services have been started.
pause
