from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.routes import agents
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="VaaniFlow AI Backend",
    version="0.1.0",
    description="Backend API for VaaniFlow AI voice workflows",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(agents.router, prefix="/api/agents", tags=["agents"])


class HealthResponse(BaseModel):
    status: str
    service: str


class RootResponse(BaseModel):
    service: str
    version: str
    status: str


@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    """Root endpoint for service identity and health discovery."""
    return RootResponse(
        service="vaaniflow-backend",
        version="0.1.0",
        status="running",
    )


@app.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Health check endpoint returning service status."""
    return HealthResponse(
        status="ok",
        service="vaaniflow-backend",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
