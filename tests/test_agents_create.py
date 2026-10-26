import os
import tempfile
import pytest
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.repositories.agent_repository import (
    AgentRepository,
    DatabasePersistenceError,
    get_agent_repository,
)
from app.schemas.agent_config import (
    AppointmentHandling,
    GeneratedAgentConfig,
    VoicePreference,
)
from app.services.elevenlabs_agent_service import (
    ElevenLabsAgentService,
    ElevenLabsProviderError,
    MissingElevenLabsApiKeyError,
    MissingVoiceIdError,
    get_elevenlabs_service,
)

client = TestClient(app)


def sample_config(voice_id: str = None) -> GeneratedAgentConfig:
    return GeneratedAgentConfig(
        agent_name="Pooja",
        business_type="Wellness Clinic",
        primary_language="hi",
        additional_languages=["en"],
        first_message="Namaste! Pooja speaking from Wellness Clinic.",
        system_prompt="You are Pooja, a wellness clinic receptionist. Be courteous and helpful.",
        voice_preference=VoicePreference(
            gender="female",
            accent="Indian English",
            description="Calm tone",
            voice_id=voice_id,
        ),
        intended_capabilities=["answer_faqs", "appointment_requests"],
        faqs=[],
        appointment_handling=AppointmentHandling(
            enabled=True,
            instructions="Record appointment requests; clinic confirms later.",
        ),
    )


def test_create_agent_success():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_agents.db")
        temp_repo = AgentRepository(db_path=db_path)

        class MockElevenLabsService:
            def create_agent(self, config, explicit_voice_id=None):
                return {
                    "agent_id": "agent_test_verified_123",
                    "voice_id": "21m00Tcm4TlvDq8ikWAM",
                }

        app.dependency_overrides[get_elevenlabs_service] = lambda: MockElevenLabsService()
        app.dependency_overrides[get_agent_repository] = lambda: temp_repo

        try:
            req_payload = {
                "configuration": sample_config().model_dump(),
                "voice_id": "21m00Tcm4TlvDq8ikWAM",
            }
            response = client.post("/api/agents", json=req_payload)
            assert response.status_code == 201
            data = response.json()
            assert data["status"] == "created"
            assert data["agent_id"] == "agent_test_verified_123"
            assert data["name"] == "Pooja"
            assert data["business_type"] == "Wellness Clinic"
            assert data["voice_id"] == "21m00Tcm4TlvDq8ikWAM"
            assert "created_at" in data

            # Verify SQLite persistence directly
            stored = temp_repo.get_agent_by_id("agent_test_verified_123")
            assert stored is not None
            assert stored["agent_id"] == "agent_test_verified_123"
            assert stored["name"] == "Pooja"
            assert stored["voice_id"] == "21m00Tcm4TlvDq8ikWAM"
        finally:
            app.dependency_overrides.clear()


def test_create_agent_payload_and_auth_headers_verification():
    captured_requests = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json={"agent_id": "agent_live_mock_777"})

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    svc = ElevenLabsAgentService(
        api_key="mock_test_key_non_secret",
        http_client=mock_client,
    )

    cfg = sample_config()
    res = svc.create_agent(cfg, explicit_voice_id="voice_custom_001")

    assert res["agent_id"] == "agent_live_mock_777"
    assert res["voice_id"] == "voice_custom_001"
    assert len(captured_requests) == 1

    req = captured_requests[0]
    assert req.url == "https://api.elevenlabs.io/v1/convai/agents/create"
    assert req.headers["xi-api-key"] == "mock_test_key_non_secret"
    assert req.headers["content-type"] == "application/json"

    import json
    body = json.loads(req.content.decode("utf-8"))
    assert body["name"] == "Pooja"
    conv_cfg = body["conversation_config"]
    assert conv_cfg["agent"]["language"] == "hi"
    assert conv_cfg["agent"]["hinglish_mode"] is True
    assert conv_cfg["tts"]["voice_id"] == "voice_custom_001"
    assert conv_cfg["tts"]["model_id"] == "eleven_turbo_v2_5"
    assert body.get("platform_settings") == {
        "auth": {
            "enable_auth": True,
        }
    }


def test_build_create_payload_includes_auth_settings():
    svc = ElevenLabsAgentService(api_key="mock_test_key_non_secret")
    cfg = sample_config()
    payload = svc.build_create_payload(cfg, voice_id="voice_custom_001")

    assert "platform_settings" in payload
    assert payload["platform_settings"] == {
        "auth": {
            "enable_auth": True,
        }
    }
    assert payload["conversation_config"]["tts"]["voice_id"] == "voice_custom_001"


def test_create_agent_missing_api_key():
    class MockMissingKeyService:
        def create_agent(self, config, explicit_voice_id=None):
            raise MissingElevenLabsApiKeyError("ElevenLabs API key is not configured on the server.")

    app.dependency_overrides[get_elevenlabs_service] = lambda: MockMissingKeyService()

    try:
        response = client.post(
            "/api/agents",
            json={"configuration": sample_config().model_dump(), "voice_id": "v1"},
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "ElevenLabs API key is not configured on the server."
    finally:
        app.dependency_overrides.clear()


def test_create_agent_missing_voice_id():
    # If no voice_id is passed in request, configuration, or environment
    svc = ElevenLabsAgentService(api_key="dummy_key", default_voice_id=None)
    app.dependency_overrides[get_elevenlabs_service] = lambda: svc

    try:
        response = client.post(
            "/api/agents",
            json={"configuration": sample_config(voice_id=None).model_dump(), "voice_id": None},
        )
        assert response.status_code == 422
        assert "A voice ID must be provided" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_create_agent_provider_error():
    class MockProviderErrorService:
        def create_agent(self, config, explicit_voice_id=None):
            raise ElevenLabsProviderError("Failed to communicate with ElevenLabs voice provider.")

    app.dependency_overrides[get_elevenlabs_service] = lambda: MockProviderErrorService()

    try:
        response = client.post(
            "/api/agents",
            json={"configuration": sample_config().model_dump(), "voice_id": "v1"},
        )
        assert response.status_code == 502
        assert response.json()["detail"] == "Failed to communicate with ElevenLabs voice provider."
    finally:
        app.dependency_overrides.clear()


def test_create_agent_persistence_failure():
    class MockFailingRepo:
        def save_agent(self, *args, **kwargs):
            raise DatabasePersistenceError("Disk error during save")

    class MockElevenLabsService:
        def create_agent(self, config, explicit_voice_id=None):
            return {"agent_id": "agent_saved_remotely_999", "voice_id": "voice_1"}

    app.dependency_overrides[get_elevenlabs_service] = lambda: MockElevenLabsService()
    app.dependency_overrides[get_agent_repository] = lambda: MockFailingRepo()

    try:
        response = client.post(
            "/api/agents",
            json={"configuration": sample_config().model_dump(), "voice_id": "voice_1"},
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "Agent was provisioned with provider but failed to persist locally."
    finally:
        app.dependency_overrides.clear()


def test_voice_id_resolution_order():
    svc = ElevenLabsAgentService(default_voice_id="default_env_voice")

    # 1. Explicit request voice overrides everything
    cfg_with_voice = sample_config(voice_id="config_voice")
    assert svc.resolve_voice_id(cfg_with_voice, explicit_voice_id="explicit_req_voice") == "explicit_req_voice"

    # 2. Config voice overrides environment default
    assert svc.resolve_voice_id(cfg_with_voice, explicit_voice_id=None) == "config_voice"

    # 3. Environment default used when request and config are None
    cfg_no_voice = sample_config(voice_id=None)
    assert svc.resolve_voice_id(cfg_no_voice, explicit_voice_id=None) == "default_env_voice"

    # 4. Raises MissingVoiceIdError when none configured
    svc_empty = ElevenLabsAgentService(default_voice_id=None)
    with pytest.raises(MissingVoiceIdError):
        svc_empty.resolve_voice_id(cfg_no_voice, explicit_voice_id=None)


def test_preview_voice_id_is_none_when_unresolved():
    cfg_no_voice = sample_config(voice_id=None)
    preview = cfg_no_voice.to_elevenlabs_conversation_config_preview()
    # Must be None, never a placeholder string
    assert preview["tts"]["voice_id"] is None


def test_extract_error_summary_handles_validation_list():
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "conversation_config", "agent", "hinglish_mode"],
                    "msg": "hinglish_mode cannot be enabled for language en",
                    "type": "value_error"
                }
            ]
        }
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert summary == "conversation_config.agent.hinglish_mode: hinglish_mode cannot be enabled for language en"


def test_extract_error_summary_handles_allowlisted_error_type():
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "param"],
                    "msg": "",
                    "type": "missing"
                }
            ]
        }
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert summary == "param: missing"


def test_extract_error_summary_unknown_error_types_fall_back_to_validation_error():
    for bad_type in ["secret99", "password123", "custom_eval_code"]:
        resp = httpx.Response(
            400,
            json={
                "detail": [
                    {
                        "loc": ["body", "param"],
                        "msg": "",
                        "type": bad_type
                    }
                ]
            }
        )
        summary = ElevenLabsAgentService._extract_error_summary(resp)
        assert bad_type not in summary
        assert summary == "param: validation_error"


def test_match_allowlisted_category_exact_canonical_matching():
    # Exactly matching entries match after conservative normalization
    assert ElevenLabsAgentService._match_allowlisted_category("voice_not_found") == "voice_not_found"
    assert ElevenLabsAgentService._match_allowlisted_category("Voice Not Found") == "voice_not_found"
    assert ElevenLabsAgentService._match_allowlisted_category("voice-not-found") == "voice_not_found"
    assert ElevenLabsAgentService._match_allowlisted_category("Voice ID is not valid") == "invalid_voice_id"
    assert ElevenLabsAgentService._match_allowlisted_category("rate-limit-exceeded") == "rate_limit_exceeded"


def test_match_allowlisted_category_does_not_substring_match_longer_unrelated_text():
    longer_msgs = [
        "Internal DB error: voice not found for customer 123",
        "voice not found in storage cluster",
        "Error occurred while searching: voice not found",
        "voice not found for id=abc12345",
    ]
    for msg in longer_msgs:
        # Category matching strictly requires canonical exact match
        assert ElevenLabsAgentService._match_allowlisted_category(msg) == ""
        # Summary extraction sanitizes tokens or falls back safely without leaking
        resp = httpx.Response(400, json={"detail": msg})
        summary = ElevenLabsAgentService._extract_error_summary(resp)
        assert "abc12345" not in summary
        assert summary in [msg, "provider_validation_error", "voice not found for id=[REDACTED]"]


def test_extract_error_summary_omits_unstructured_arbitrary_text():
    resp = httpx.Response(400, json={"message": "Internal unexpected DB state user=admin"})
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    # Unsafe characters (like '=') cause sanitization to fall back safely to provider_validation_error
    assert "user=admin" not in summary
    assert summary == "provider_validation_error"


def test_extract_error_summary_handles_non_json_safely():
    resp = httpx.Response(502, text="<html>502 Bad Gateway</html>")
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert summary == ""


def test_extract_error_summary_masks_short_secret_in_quotes():
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "tts", "voice_id"],
                    "msg": "Voice 'abc12345' not found",
                    "type": "value_error"
                }
            ]
        }
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    # The 8-character secret 'abc12345' must be masked to '[VALUE]'
    assert "abc12345" not in summary
    assert summary == "tts.voice_id: Voice '[VALUE]' not found"


def test_extract_error_summary_suppresses_message_with_short_secret_keyword():
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "auth", "api_key"],
                    "msg": "Invalid key: pass123",
                    "type": "secret99"
                }
            ]
        }
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    # pass123 (7 chars), 'key', and unallowlisted type 'secret99' are completely suppressed
    assert "pass123" not in summary
    assert "secret99" not in summary
    assert summary == "auth.api_key: validation_error"


def test_extract_error_summary_redacts_urls():
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "webhook"],
                    "msg": "Failed to connect to wss://api.elevenlabs.io/v1/convai?sig=123",
                    "type": "value_error"
                }
            ]
        }
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert "wss://" not in summary
    assert summary == "webhook: value_error"


def test_extract_error_summary_discards_unsafe_loc():
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "<script>bad</script>", "field"],
                    "msg": "invalid value",
                    "type": "value_error"
                }
            ]
        }
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert "<script>" not in summary
    assert summary == "invalid value"


def test_create_agent_endpoint_omits_short_secrets_from_logs(caplog):
    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "detail": [
                    {
                        "loc": ["body", "auth", "token"],
                        "msg": "Unauthorized token: secret99",
                        "type": "password123"
                    }
                ]
            }
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    svc = ElevenLabsAgentService(
        api_key="mock_test_key_non_secret",
        http_client=mock_client,
    )
    app.dependency_overrides[get_elevenlabs_service] = lambda: svc

    try:
        with caplog.at_level("ERROR"):
            response = client.post(
                "/api/agents",
                json={"configuration": sample_config().model_dump(), "voice_id": "v1"},
            )
            # 1. Client-facing response remains strictly generic 502
            assert response.status_code == 502
            assert response.json()["detail"] == "ElevenLabs agent creation failed with status 400."
            assert "secret99" not in response.text
            assert "password123" not in response.text

            # 2. Server log has sanitized provider summary with safe fallback 'validation_error'
            assert "ElevenLabs returned HTTP error status 400: auth.token: validation_error" in caplog.text
            assert "secret99" not in caplog.text
            assert "password123" not in caplog.text
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Fix B: short unquoted opaque-token masking
# ---------------------------------------------------------------------------

def test_extract_error_summary_masks_short_opaque_unquoted_token():
    """A short mixed letter+digit token without quotes must be masked as [REDACTED]."""
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "tts", "voice_id"],
                    "msg": "Voice id123abc8 not found",
                    "type": "value_error",
                }
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert "id123abc8" not in summary, f"Short opaque token leaked: {summary!r}"
    assert "[REDACTED]" in summary
    assert "tts.voice_id" in summary


def test_sanitize_validation_message_pure_alpha_words_are_not_masked():
    """Ordinary English words (no digits) must not be masked by the short-token pattern."""
    msg = "voice identifier not found"
    result = ElevenLabsAgentService._sanitize_validation_message(msg, "value_error")
    # No [REDACTED] should appear; the message should remain human-readable.
    assert "[REDACTED]" not in result, f"Ordinary words were wrongly masked: {result!r}"
    assert result == "voice identifier not found"


def test_sanitize_validation_message_short_mixed_token_is_masked():
    """A short (8-15 char) value mixing letters and digits is masked as [REDACTED]."""
    msg = "Invalid value abc12345 supplied"
    result = ElevenLabsAgentService._sanitize_validation_message(msg, "value_error")
    assert "abc12345" not in result, f"Short mixed token leaked: {result!r}"
    assert "[REDACTED]" in result


def test_sanitize_validation_message_existing_behaviors_intact():
    """Confirm quoted secrets, URLs, sensitive keywords, long tokens, and fallback still work."""
    svc = ElevenLabsAgentService

    # Quoted secret -> '[VALUE]'
    r1 = svc._sanitize_validation_message("Voice 'abc12345' not found", "value_error")
    assert "abc12345" not in r1
    assert "[VALUE]" in r1

    # URL -> suppressed to fallback
    r2 = svc._sanitize_validation_message("See https://evil.com/steal?x=1 for details", "value_error")
    assert "evil.com" not in r2
    assert r2 == "value_error"

    # Sensitive keyword -> suppressed
    r3 = svc._sanitize_validation_message("Invalid api_key supplied", "value_error")
    assert "api_key" not in r3
    assert r3 == "value_error"

    # Long pure-alpha token prefixed by 'Token' (a sensitive keyword): the whole message
    # is suppressed by SENSITIVE_KEYWORD_PATTERN before TOKEN_PATTERN fires.
    r4 = svc._sanitize_validation_message("Token abcdefghijklmnopqr is expired", "value_error")
    assert "abcdefghijklmnopqr" not in r4
    assert r4 == "value_error"  # suppressed by sensitive keyword, not TOKEN_PATTERN

    # Unknown error type falls back to validation_error
    r5 = svc._sanitize_validation_message("", "secret99")
    assert r5 == "validation_error"


# ---------------------------------------------------------------------------
# Fix A: field-path length limit
# ---------------------------------------------------------------------------

def test_extract_error_summary_accepts_valid_length_path():
    """A valid path of acceptable length must be accepted."""
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "conversation_config", "agent", "hinglish_mode"],
                    "msg": "value required",
                    "type": "missing",
                }
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert "conversation_config" in summary
    # msg="value required" sanitizes successfully; path and message appear, not error type.
    assert "value required" in summary


def test_extract_error_summary_rejects_path_over_200_chars():
    """A field path longer than 200 characters must be omitted; the safe message is retained."""
    long_segment = "a" * 201          # 201-char single segment => path len > 200
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", long_segment],
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    # Path must not appear; sanitized message alone survives
    assert long_segment not in summary
    assert summary == "invalid value"


def test_extract_error_summary_rejects_path_with_script_tag():
    """Existing XSS behaviour: unsafe loc path discarded, message retained."""
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "<script>bad</script>", "field"],
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert "<script>" not in summary
    assert summary == "invalid value"


def test_extract_error_summary_handles_integer_loc_elements():
    """Integer loc elements (FastAPI list indices) are stringified safely."""
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", "items", 0, "voice_id"],
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert "items.0.voice_id" in summary
    assert "invalid value" in summary


def test_extract_error_summary_handles_empty_loc():
    """An empty loc list produces no field path; only the sanitized message is used."""
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": [],
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert summary == "invalid value"


def test_extract_error_summary_large_validation_list_is_bounded():
    """A large number of validation errors must not produce excessive log output (>250 chars)."""
    items = [
        {"loc": ["body", f"field{i}"], "msg": "invalid", "type": "value_error"}
        for i in range(50)
    ]
    resp = httpx.Response(400, json={"detail": items})
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert len(summary) <= 250, f"Summary exceeded 250 chars: {len(summary)}"

# ---------------------------------------------------------------------------
# Defensive handling of malformed loc values & payloads
# ---------------------------------------------------------------------------

def test_extract_error_summary_handles_null_loc():
    """A null loc in validation error item must not raise an exception and preserves msg."""
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": None,
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert summary == "invalid value"


def test_extract_error_summary_handles_missing_loc():
    """A validation item without loc key must not raise an exception and preserves msg."""
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert summary == "invalid value"


def test_extract_error_summary_handles_string_loc():
    """String loc values: valid dotted paths are recognized, unsafe strings are rejected safely."""
    # Valid dotted path as string
    resp_valid = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": "conversation_config.agent",
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary_valid = ElevenLabsAgentService._extract_error_summary(resp_valid)
    assert summary_valid == "conversation_config.agent: invalid value"

    # Malformed / unsafe string with spaces
    resp_invalid = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": "not a valid field path",
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary_invalid = ElevenLabsAgentService._extract_error_summary(resp_invalid)
    assert summary_invalid == "invalid value"


def test_extract_error_summary_handles_integer_and_dict_loc():
    """Non-iterable loc types (integer, dict) do not raise and preserve the sanitized msg."""
    resp_int = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": 12345,
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary_int = ElevenLabsAgentService._extract_error_summary(resp_int)
    assert summary_int == "invalid value"

    resp_dict = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": {"unexpected": "object"},
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary_dict = ElevenLabsAgentService._extract_error_summary(resp_dict)
    assert summary_dict == "invalid value"


def test_extract_error_summary_handles_list_with_unexpected_element_types():
    """A loc list containing None, nested dicts, or other unexpected element types does not raise."""
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", None, {"bad": "type"}, 123],
                    "msg": "invalid value",
                    "type": "value_error",
                }
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    # The nested dict makes SAFE_FIELD_PATH_PATTERN reject the candidate; sanitized message alone is kept
    assert summary == "invalid value"


def test_extract_error_summary_exact_200_and_201_path_boundaries():
    """Confirms exact boundary behavior at 200 chars (accepted) vs 201 chars (rejected)."""
    # Exactly 200 chars: accepted
    path_200 = "a" * 200
    resp_200 = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", path_200],
                    "msg": "error_detail",
                    "type": "value_error",
                }
            ]
        },
    )
    summary_200 = ElevenLabsAgentService._extract_error_summary(resp_200)
    assert path_200 in summary_200
    assert summary_200 == f"{path_200}: error_detail"

    # Exactly 201 chars: rejected
    path_201 = "a" * 201
    resp_201 = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": ["body", path_201],
                    "msg": "error_detail",
                    "type": "value_error",
                }
            ]
        },
    )
    summary_201 = ElevenLabsAgentService._extract_error_summary(resp_201)
    assert path_201 not in summary_201
    assert summary_201 == "error_detail"


def test_extract_error_summary_malformed_item_alongside_valid_item():
    """A malformed item (null loc) does not crash processing of adjacent valid items."""
    resp = httpx.Response(
        400,
        json={
            "detail": [
                {
                    "loc": None,
                    "msg": "malformed loc error",
                    "type": "value_error",
                },
                {
                    "loc": ["body", "tts", "voice_id"],
                    "msg": "valid voice error",
                    "type": "value_error",
                },
            ]
        },
    )
    summary = ElevenLabsAgentService._extract_error_summary(resp)
    assert "malformed loc error" in summary
    assert "tts.voice_id: valid voice error" in summary
    assert summary == "malformed loc error; tts.voice_id: valid voice error"

# ---------------------------------------------------------------------------
# Language & Hinglish-mode payload construction tests
# ---------------------------------------------------------------------------

def test_payload_english_primary_with_hindi_additional_disables_hinglish_mode():
    """English primary with Hindi additional language must NOT enable hinglish_mode (causes 400)."""
    svc = ElevenLabsAgentService(api_key="mock_key")
    cfg = sample_config()
    cfg.primary_language = "en"
    cfg.additional_languages = ["hi"]

    payload = svc.build_create_payload(cfg, voice_id="test_voice_1")
    agent_cfg = payload["conversation_config"]["agent"]
    assert agent_cfg["language"] == "en"
    assert "hinglish_mode" not in agent_cfg, "hinglish_mode must be omitted when language is 'en'"

    # Also verify preview matches
    preview = cfg.to_elevenlabs_conversation_config_preview()
    assert preview["agent"]["language"] == "en"
    assert "hinglish_mode" not in preview["agent"]


def test_payload_hindi_primary_enables_hinglish_mode():
    """Hindi primary language must enable hinglish_mode for code-switching support."""
    svc = ElevenLabsAgentService(api_key="mock_key")

    # Hindi primary with English additional
    cfg_bilingual = sample_config()
    cfg_bilingual.primary_language = "hi"
    cfg_bilingual.additional_languages = ["en"]
    payload_bilingual = svc.build_create_payload(cfg_bilingual, voice_id="test_voice_1")
    agent_bilingual = payload_bilingual["conversation_config"]["agent"]
    assert agent_bilingual["language"] == "hi"
    assert agent_bilingual["hinglish_mode"] is True
    assert cfg_bilingual.to_elevenlabs_conversation_config_preview()["agent"]["hinglish_mode"] is True

    # Hindi primary only (no additional languages)
    cfg_hindi_only = sample_config()
    cfg_hindi_only.primary_language = "hi"
    cfg_hindi_only.additional_languages = []
    payload_hindi_only = svc.build_create_payload(cfg_hindi_only, voice_id="test_voice_1")
    agent_hindi_only = payload_hindi_only["conversation_config"]["agent"]
    assert agent_hindi_only["language"] == "hi"
    assert agent_hindi_only["hinglish_mode"] is True


def test_payload_english_only_disables_hinglish_mode():
    """English-only configuration must have language 'en' and hinglish_mode False."""
    svc = ElevenLabsAgentService(api_key="mock_key")
    cfg = sample_config()
    cfg.primary_language = "en"
    cfg.additional_languages = []

    payload = svc.build_create_payload(cfg, voice_id="test_voice_1")
    agent_cfg = payload["conversation_config"]["agent"]
    assert agent_cfg["language"] == "en"
    assert "hinglish_mode" not in agent_cfg
    assert "hinglish_mode" not in cfg.to_elevenlabs_conversation_config_preview()["agent"]


def test_create_agent_sends_valid_language_payload_for_en_hi_bilingual():
    """End-to-end service test verifying outgoing payload for English-primary Hindi-bilingual config."""
    captured_requests = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json={"agent_id": "agent_valid_lang_123"})

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    svc = ElevenLabsAgentService(api_key="mock_key", http_client=mock_client)

    cfg = sample_config()
    cfg.primary_language = "en"
    cfg.additional_languages = ["hi"]

    res = svc.create_agent(cfg, explicit_voice_id="SZfY4K69FwXus87eayHK")
    assert res["agent_id"] == "agent_valid_lang_123"
    assert len(captured_requests) == 1

    import json
    req_body = json.loads(captured_requests[0].content.decode("utf-8"))
    agent_conf = req_body["conversation_config"]["agent"]
    assert agent_conf["language"] == "en"
    assert "hinglish_mode" not in agent_conf

def test_payload_normalizes_whitespace_and_uppercase_primary_language():
    """Whitespace and mixed-case primary language inputs serialize consistently as lowercase ISO codes."""
    svc = ElevenLabsAgentService(api_key="mock_key")

    # Uppercase and padded Hindi: "  HI  "
    cfg_hi = sample_config()
    cfg_hi.primary_language = "  HI  "
    cfg_hi.additional_languages = ["en"]
    payload_hi = svc.build_create_payload(cfg_hi, voice_id="test_voice_1")
    agent_hi = payload_hi["conversation_config"]["agent"]
    assert agent_hi["language"] == "hi", f"Expected normalized 'hi', got {agent_hi['language']!r}"
    assert agent_hi["hinglish_mode"] is True
    preview_hi = cfg_hi.to_elevenlabs_conversation_config_preview()
    assert preview_hi["agent"]["language"] == "hi"
    assert preview_hi["agent"]["hinglish_mode"] is True

    # Mixed-case and padded English: "  En  "
    cfg_en = sample_config()
    cfg_en.primary_language = "  En  "
    cfg_en.additional_languages = ["hi"]
    payload_en = svc.build_create_payload(cfg_en, voice_id="test_voice_1")
    agent_en = payload_en["conversation_config"]["agent"]
    assert agent_en["language"] == "en", f"Expected normalized 'en', got {agent_en['language']!r}"
    assert "hinglish_mode" not in agent_en
    preview_en = cfg_en.to_elevenlabs_conversation_config_preview()
    assert preview_en["agent"]["language"] == "en"
    assert "hinglish_mode" not in preview_en["agent"]

# ---------------------------------------------------------------------------
# Enhanced provider error diagnostics & mock payload tests
# ---------------------------------------------------------------------------

def test_extract_error_summary_nested_error_object_and_string_variations():
    """Extracts safe diagnostics from nested error objects and string errors."""
    svc = ElevenLabsAgentService

    # 1. error as string allowlisted
    r1 = httpx.Response(400, json={"error": "rate_limit_exceeded"})
    assert svc._extract_error_summary(r1) == "rate_limit_exceeded"

    # 2. error as nested object with status/code
    r2 = httpx.Response(400, json={"error": {"status": "voice_not_found"}})
    assert svc._extract_error_summary(r2) == "voice_not_found"

    r3 = httpx.Response(400, json={"error": {"code": "invalid_voice_id", "message": "Ignored"}})
    assert svc._extract_error_summary(r3) == "invalid_voice_id"

    # 3. error as nested object with sanitized message containing short opaque token
    r4 = httpx.Response(400, json={"error": {"message": "Voice id123abc8 not found"}})
    summary_r4 = svc._extract_error_summary(r4)
    assert "id123abc8" not in summary_r4
    assert summary_r4 == "Voice [REDACTED] not found"

    # 4. error as nested object with details list
    r5 = httpx.Response(400, json={"error": {"details": [{"loc": ["body", "param"], "msg": "value required", "type": "missing"}]}})
    assert svc._extract_error_summary(r5) == "param: value required"

    # 5. top-level errors list
    r6 = httpx.Response(400, json={"errors": [{"loc": ["body", "voice_id"], "msg": "invalid value", "type": "value_error"}]})
    assert svc._extract_error_summary(r6) == "voice_id: invalid value"

    # 6. top-level message string allowlisted
    r7 = httpx.Response(400, json={"message": "quota_exceeded"})
    assert svc._extract_error_summary(r7) == "quota_exceeded"

    # 7. top-level message string sanitized
    r8 = httpx.Response(400, json={"message": "Invalid parameter name"})
    assert svc._extract_error_summary(r8) == "Invalid parameter name"


def test_extract_error_summary_malicious_and_sensitive_payloads_safely_suppressed():
    """Malicious scripts, URLs, sensitive keywords, and secret tokens never appear in extracted summary."""
    svc = ElevenLabsAgentService

    # Malicious script tag in message -> falls back to provider_validation_error
    r_script = httpx.Response(400, json={"message": "<script>alert('pwned')</script>"})
    s_script = svc._extract_error_summary(r_script)
    assert "<script>" not in s_script
    assert s_script == "provider_validation_error"

    # Malicious URL in detail -> suppressed to provider_validation_error
    r_url = httpx.Response(400, json={"detail": "Failed connecting to https://attacker.com/leak?k=123"})
    s_url = svc._extract_error_summary(r_url)
    assert "https://" not in s_url
    assert "attacker.com" not in s_url
    assert s_url == "provider_validation_error"

    # Sensitive keyword (api_key) in error object -> suppressed to provider_validation_error
    r_key = httpx.Response(400, json={"error": {"message": "Invalid api_key supplied: super_secret_123"}})
    s_key = svc._extract_error_summary(r_key)
    assert "super_secret_123" not in s_key
    assert "api_key" not in s_key
    assert s_key == "provider_validation_error"

    # Sensitive keyword (password) in detail string -> suppressed
    r_pass = httpx.Response(400, json={"detail": "Incorrect password provided"})
    s_pass = svc._extract_error_summary(r_pass)
    assert "password" not in s_pass
    assert s_pass == "provider_validation_error"

    # Long token masked in error message
    r_tok = httpx.Response(400, json={"error": "Invalid voice 21m00Tcm4TlvDq8ikWAM specified"})
    s_tok = svc._extract_error_summary(r_tok)
    assert "21m00Tcm4TlvDq8ikWAM" not in s_tok
    assert "[REDACTED_TOKEN]" in s_tok


def test_extract_error_summary_fallback_for_empty_missing_and_null_payloads():
    """Empty dictionaries, null values, and unrecognized keys return provider_validation_error without crashing."""
    svc = ElevenLabsAgentService

    # Empty dict
    assert svc._extract_error_summary(httpx.Response(400, json={})) == "provider_validation_error"

    # Null values
    assert svc._extract_error_summary(httpx.Response(400, json={"detail": None, "error": None, "message": None})) == "provider_validation_error"

    # Empty list
    assert svc._extract_error_summary(httpx.Response(400, json={"detail": []})) == "provider_validation_error"

    # Unrecognized keys
    assert svc._extract_error_summary(httpx.Response(400, json={"unrecognized_field": 12345})) == "provider_validation_error"

    # Malformed non-JSON remains empty string
    assert svc._extract_error_summary(httpx.Response(502, text="<html>502 Bad Gateway</html>")) == ""


def test_tts_model_id_selection_by_primary_language():
    """English agents must use eleven_turbo_v2, Hindi agents use eleven_turbo_v2_5."""
    svc = ElevenLabsAgentService(api_key="mock_key")

    cfg_en = sample_config()
    cfg_en.primary_language = "en"
    p_en = svc.build_create_payload(cfg_en, voice_id="voice_1")
    assert p_en["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2"
    assert cfg_en.to_elevenlabs_conversation_config_preview()["tts"]["model_id"] == "eleven_turbo_v2"

    cfg_hi = sample_config()
    cfg_hi.primary_language = "hi"
    p_hi = svc.build_create_payload(cfg_hi, voice_id="voice_1")
    assert p_hi["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2_5"
    assert cfg_hi.to_elevenlabs_conversation_config_preview()["tts"]["model_id"] == "eleven_turbo_v2_5"


def test_list_agents_empty_returns_200_and_empty_list():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_empty_list.db")
        temp_repo = AgentRepository(db_path=db_path)
        app.dependency_overrides[get_agent_repository] = lambda: temp_repo

        try:
            response = client.get("/api/agents")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["agents"] == []
            assert data["total"] == 0
        finally:
            app.dependency_overrides.clear()


def test_list_agents_returns_stored_agents_descending():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_list_populated.db")
        temp_repo = AgentRepository(db_path=db_path)
        temp_repo.save_agent(
            agent_id="agent_first_001",
            name="First Receptionist",
            business_type="Dental Clinic",
            voice_id="voice_001",
            configuration_json="{}",
            created_at="2026-10-25T10:00:00Z",
        )
        temp_repo.save_agent(
            agent_id="agent_second_002",
            name="Second Receptionist",
            business_type="Salon",
            voice_id="voice_002",
            configuration_json="{}",
            created_at="2026-10-25T11:00:00Z",
        )

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        try:
            response = client.get("/api/agents")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["total"] == 2
            assert len(data["agents"]) == 2
            # Descending order: second created should be first
            assert data["agents"][0]["agent_id"] == "agent_second_002"
            assert data["agents"][0]["name"] == "Second Receptionist"
            assert data["agents"][1]["agent_id"] == "agent_first_001"
            assert data["agents"][1]["name"] == "First Receptionist"
        finally:
            app.dependency_overrides.clear()


def test_list_agents_database_error_returns_500():
    class BrokenRepository:
        def list_agents(self):
            raise DatabasePersistenceError("Disk I/O error")

    app.dependency_overrides[get_agent_repository] = lambda: BrokenRepository()
    try:
        response = client.get("/api/agents")
        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to retrieve agents from database."
    finally:
        app.dependency_overrides.clear()


def test_e2e_configure_create_list_flow():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_e2e_flow.db")
        temp_repo = AgentRepository(db_path=db_path)

        class MockElevenLabsService:
            def create_agent(self, config, explicit_voice_id=None):
                return {
                    "agent_id": "agent_e2e_live_456",
                    "voice_id": explicit_voice_id or "voice_mock_default",
                }

            def get_signed_url(self, agent_id):
                return f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}&conversation_signature=sig_abc"

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        app.dependency_overrides[get_elevenlabs_service] = lambda: MockElevenLabsService()

        try:
            # 1. Create agent
            cfg = sample_config(voice_id="voice_custom_e2e")
            create_res = client.post(
                "/api/agents",
                json={
                    "configuration": cfg.model_dump(),
                    "voice_id": "voice_custom_e2e",
                },
            )
            assert create_res.status_code == 201
            created_data = create_res.json()
            assert created_data["agent_id"] == "agent_e2e_live_456"

            # 2. List agents
            list_res = client.get("/api/agents")
            assert list_res.status_code == 200
            list_data = list_res.json()
            assert list_data["total"] == 1
            assert list_data["agents"][0]["agent_id"] == "agent_e2e_live_456"

            # 3. Session for created agent
            sess_res = client.post("/api/agents/agent_e2e_live_456/session")
            assert sess_res.status_code == 200
            sess_data = sess_res.json()
            assert sess_data["status"] == "ok"
            assert sess_data["agent_id"] == "agent_e2e_live_456"
            assert sess_data["signed_url"].startswith("wss://")
        finally:
            app.dependency_overrides.clear()


def test_payload_bilingual_alias_matrix():
    svc = ElevenLabsAgentService(api_key="mock_key")

    # English alias
    cfg_en = sample_config()
    cfg_en.primary_language = "English"
    p_en = svc.build_create_payload(cfg_en, voice_id="v1")
    agent_en = p_en["conversation_config"]["agent"]
    assert agent_en["language"] == "en"
    assert "hinglish_mode" not in agent_en
    assert p_en["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2"

    # Hindi alias
    cfg_hi = sample_config()
    cfg_hi.primary_language = "Hindi"
    p_hi = svc.build_create_payload(cfg_hi, voice_id="v1")
    agent_hi = p_hi["conversation_config"]["agent"]
    assert agent_hi["language"] == "hi"
    assert agent_hi["hinglish_mode"] is True
    assert p_hi["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2_5"

    # Hinglish alias
    cfg_hing = sample_config()
    cfg_hing.primary_language = "Hinglish"
    p_hing = svc.build_create_payload(cfg_hing, voice_id="v1")
    agent_hing = p_hing["conversation_config"]["agent"]
    assert agent_hing["language"] == "hi"
    assert agent_hing["hinglish_mode"] is True
    assert p_hing["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2_5"

    # Regional code hi-IN
    cfg_hi_in = sample_config()
    cfg_hi_in.primary_language = "hi-IN"
    p_hi_in = svc.build_create_payload(cfg_hi_in, voice_id="v1")
    agent_hi_in = p_hi_in["conversation_config"]["agent"]
    assert agent_hi_in["language"] == "hi"
    assert agent_hi_in["hinglish_mode"] is True

    # Regional code en-US
    cfg_en_us = sample_config()
    cfg_en_us.primary_language = "en-US"
    p_en_us = svc.build_create_payload(cfg_en_us, voice_id="v1")
    agent_en_us = p_en_us["conversation_config"]["agent"]
    assert agent_en_us["language"] == "en"
    assert "hinglish_mode" not in agent_en_us
    assert p_en_us["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2"


def test_create_agent_persists_normalized_language_in_db():
    import json
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_norm_db.db")
        temp_repo = AgentRepository(db_path=db_path)

        class MockElevenLabsService:
            def create_agent(self, config, explicit_voice_id=None):
                return {
                    "agent_id": "agent_norm_test_789",
                    "voice_id": "v_norm",
                }

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        app.dependency_overrides[get_elevenlabs_service] = lambda: MockElevenLabsService()

        try:
            cfg = sample_config(voice_id="v_norm")
            cfg.primary_language = "Hindi"
            cfg.additional_languages = ["English"]

            response = client.post(
                "/api/agents",
                json={
                    "configuration": cfg.model_dump(),
                    "voice_id": "v_norm",
                },
            )
            assert response.status_code == 201

            stored = temp_repo.get_agent_by_id("agent_norm_test_789")
            assert stored is not None
            persisted_cfg = json.loads(stored["configuration_json"])
            assert persisted_cfg["primary_language"] == "hi"
            assert persisted_cfg["additional_languages"] == ["en"]
        finally:
            app.dependency_overrides.clear()


def test_create_agent_unsupported_primary_language_returns_422():
    raw_payload = sample_config().model_dump()
    raw_payload["primary_language"] = "french"
    response = client.post(
        "/api/agents",
        json={"configuration": raw_payload, "voice_id": "v1"},
    )
    assert response.status_code == 422
    assert "Unsupported language 'french'" in response.text


def test_create_agent_unsupported_additional_language_returns_422():
    raw_payload = sample_config().model_dump()
    raw_payload["additional_languages"] = ["klingon"]
    response = client.post(
        "/api/agents",
        json={"configuration": raw_payload, "voice_id": "v1"},
    )
    assert response.status_code == 422
    assert "Unsupported language 'klingon'" in response.text


def test_create_agent_provider_error_leaves_zero_local_records():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_clean_failure.db")
        temp_repo = AgentRepository(db_path=db_path)

        class MockProviderErrorService:
            def create_agent(self, config, explicit_voice_id=None):
                raise ElevenLabsProviderError("Failed to communicate with ElevenLabs voice provider.")

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        app.dependency_overrides[get_elevenlabs_service] = lambda: MockProviderErrorService()

        try:
            response = client.post(
                "/api/agents",
                json={"configuration": sample_config().model_dump(), "voice_id": "v1"},
            )
            assert response.status_code == 502
            # Explicitly verify local database has 0 records
            assert len(temp_repo.list_agents()) == 0
        finally:
            app.dependency_overrides.clear()
