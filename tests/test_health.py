from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /api/health returns 200 OK and expected payload."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "vaaniflow-backend",
    }


def test_root_endpoint():
    """Verify GET / returns 200 OK and expected service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["service"] == "vaaniflow-backend"
    assert data["version"] == "0.1.0"


def test_cors_preflight_allowed_origin():
    """Verify CORS preflight returns correct access-control headers for allowed local origins."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_settings_port_and_cors_configuration():
    """Verify Settings correctly resolves PORT alias and configurable origins."""
    from app.core.config import Settings

    s = Settings(
        PORT=10000,
        ALLOWED_ORIGINS="https://vaaniflow.vercel.app, https://demo.onrender.com/",
        FRONTEND_URL="https://frontend.example.com",
    )
    assert s.app_port == 10000
    origins = s.cors_origins
    assert "http://localhost:5173" in origins
    assert "http://127.0.0.1:5173" in origins
    assert "http://localhost:3000" in origins
    assert "https://vaaniflow.vercel.app" in origins
    assert "https://demo.onrender.com" in origins
    assert "https://frontend.example.com" in origins
