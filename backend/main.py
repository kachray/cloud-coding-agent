"""Minimal FastAPI application for Cloud Coding Agent."""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from github.router import router as github_router  # noqa: E402

app = FastAPI(title="Cloud Coding Agent API")
app.include_router(github_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)