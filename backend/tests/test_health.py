"""Functional tests for the backend API."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Test that /health endpoint returns status ok."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_response_is_json(self, client):
        """Test that /health endpoint returns valid JSON."""
        response = client.get("/health")

        assert response.headers["content-type"] == "application/json"

    def test_health_has_correct_structure(self, client):
        """Test that /health response has the expected structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "ok"