
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import requests
import time
import threading

# Load .env file  
env_path = Path(".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="Cloud Coding Agent API")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# Start server in thread
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Wait for server to start
time.sleep(2)

# Test the /health endpoint
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"Backend /health endpoint response: {response.status_code} - {response.json()}")
except Exception as e:
    print(f"Error: {e}")
