import os
import tempfile
import pytest
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.repositories.agent_repository import (
    AgentRepository,
    get_agent_repository,
)
from app.services.elevenlabs_agent_service import (
    ElevenLabsAgentService,
    ElevenLabsProviderError,
    InvalidProviderResponseError,
    MissingElevenLabsApiKeyError,
    get_elevenlabs_service,
)

client = TestClient(app)


def test_session_success_for_existing_agent():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_session.db")
        temp_repo = AgentRepository(db_path=db_path)

        # Seed local agent
        temp_repo.save_agent(
            agent_id="agent_known_123",
            name="Aanya",
            business_type="Dental Clinic",
            voice_id="voice_known_abc",
            configuration_json="{}",
            created_at="2026-10-25T00:00:00Z",
        )

        class MockElevenLabsService:
            def get_signed_url(self, agent_id: str) -> str:
                assert agent_id == "agent_known_123"
                return "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=agent_known_123&conversation_signature=mock_token_sig_456"

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        app.dependency_overrides[get_elevenlabs_service] = lambda: MockElevenLabsService()

        try:
            response = client.post("/api/agents/agent_known_123/session")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["agent_id"] == "agent_known_123"
            assert data["signed_url"].startswith("wss://api.elevenlabs.io/")
            assert data["expires_in_seconds"] == 900
        finally:
            app.dependency_overrides.clear()


def test_session_verifies_official_endpoint_and_auth_header():
    captured_requests = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={"signed_url": "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=agent_live_001&conversation_signature=sig_xyz"},
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    svc = ElevenLabsAgentService(
        api_key="mock_secret_key_non_leak",
        http_client=mock_client,
    )

    signed_url = svc.get_signed_url("agent_live_001")
    assert signed_url.startswith("wss://")
    assert len(captured_requests) == 1

    req = captured_requests[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/convai/conversation/get-signed-url"
    assert req.url.params["agent_id"] == "agent_live_001"
    assert req.headers["xi-api-key"] == "mock_secret_key_non_leak"


def test_session_unknown_agent_returns_404_without_calling_provider():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_session_empty.db")
        temp_repo = AgentRepository(db_path=db_path)

        provider_called = {"called": False}

        class MockElevenLabsService:
            def get_signed_url(self, agent_id: str) -> str:
                provider_called["called"] = True
                return "wss://fake"

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        app.dependency_overrides[get_elevenlabs_service] = lambda: MockElevenLabsService()

        try:
            response = client.post("/api/agents/unknown_external_agent_id/session")
            assert response.status_code == 404
            assert response.json()["detail"] == "Agent not found in local registry."
            # Confirm zero calls to ElevenLabs provider
            assert provider_called["called"] is False
        finally:
            app.dependency_overrides.clear()


def test_session_missing_elevenlabs_api_key():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_session_nokey.db")
        temp_repo = AgentRepository(db_path=db_path)
        temp_repo.save_agent(
            agent_id="agent_seed_1",
            name="Test",
            business_type="Retail",
            voice_id="v1",
            configuration_json="{}",
            created_at="2026-10-25T00:00:00Z",
        )

        class MockMissingKeyService:
            def get_signed_url(self, agent_id: str) -> str:
                raise MissingElevenLabsApiKeyError("ElevenLabs API key is not configured on the server.")

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        app.dependency_overrides[get_elevenlabs_service] = lambda: MockMissingKeyService()

        try:
            response = client.post("/api/agents/agent_seed_1/session")
            assert response.status_code == 500
            assert response.json()["detail"] == "ElevenLabs API key is not configured on the server."
        finally:
            app.dependency_overrides.clear()


def test_session_provider_http_failure_returns_502():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_session_err.db")
        temp_repo = AgentRepository(db_path=db_path)
        temp_repo.save_agent(
            agent_id="agent_seed_2",
            name="Test",
            business_type="Retail",
            voice_id="v1",
            configuration_json="{}",
            created_at="2026-10-25T00:00:00Z",
        )

        class MockProviderErrorService:
            def get_signed_url(self, agent_id: str) -> str:
                raise ElevenLabsProviderError("Failed to communicate with ElevenLabs voice provider.")

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        app.dependency_overrides[get_elevenlabs_service] = lambda: MockProviderErrorService()

        try:
            response = client.post("/api/agents/agent_seed_2/session")
            assert response.status_code == 502
            assert response.json()["detail"] == "Failed to communicate with ElevenLabs voice provider."
        finally:
            app.dependency_overrides.clear()


def test_session_network_timeout_returns_502():
    def timeout_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out", request=request)

    mock_client = httpx.Client(transport=httpx.MockTransport(timeout_transport))
    svc = ElevenLabsAgentService(api_key="test_key", http_client=mock_client)

    with pytest.raises(ElevenLabsProviderError):
        svc.get_signed_url("agent_timeout_test")


def test_session_malformed_json_and_missing_signed_url():
    # 1. Non-JSON response
    def non_json_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>502 Bad Gateway from Cloudflare</html>")

    mock_client = httpx.Client(transport=httpx.MockTransport(non_json_transport))
    svc = ElevenLabsAgentService(api_key="test_key", http_client=mock_client)
    with pytest.raises(InvalidProviderResponseError):
        svc.get_signed_url("agent_test")

    # 2. JSON without signed_url
    def missing_url_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": "ok"})

    mock_client2 = httpx.Client(transport=httpx.MockTransport(missing_url_transport))
    svc2 = ElevenLabsAgentService(api_key="test_key", http_client=mock_client2)
    with pytest.raises(InvalidProviderResponseError):
        svc2.get_signed_url("agent_test")

    # 3. Empty string signed_url
    def empty_url_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"signed_url": "   "})

    mock_client3 = httpx.Client(transport=httpx.MockTransport(empty_url_transport))
    svc3 = ElevenLabsAgentService(api_key="test_key", http_client=mock_client3)
    with pytest.raises(InvalidProviderResponseError):
        svc3.get_signed_url("agent_test")


def test_session_whitespace_agent_id_returns_404():
    response = client.post("/api/agents/%20%20%20/session")
    assert response.status_code == 404
    assert response.json()["detail"] == "Agent not found in local registry."


def test_session_signed_url_never_leaked_in_server_logs(caplog):
    import logging
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_session_leak.db")
        temp_repo = AgentRepository(db_path=db_path)
        temp_repo.save_agent(
            agent_id="agent_leak_check_1",
            name="Leak Check",
            business_type="Clinic",
            voice_id="v_leak",
            configuration_json="{}",
            created_at="2026-10-25T00:00:00Z",
        )

        sensitive_token = "conversation_signature=SECRET_LEAK_TOKEN_9999"
        class MockSafeService:
            def get_signed_url(self, agent_id):
                return f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}&{sensitive_token}"

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        app.dependency_overrides[get_elevenlabs_service] = lambda: MockSafeService()

        with caplog.at_level(logging.DEBUG):
            response = client.post("/api/agents/agent_leak_check_1/session")
            assert response.status_code == 200
            assert "SECRET_LEAK_TOKEN_9999" in response.json()["signed_url"]

        # Ensure the secret token or full URL does NOT appear in captured log records
        for record in caplog.records:
            assert "SECRET_LEAK_TOKEN_9999" not in record.getMessage()
