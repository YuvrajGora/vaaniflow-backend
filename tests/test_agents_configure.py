import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.agent_config import (
    AppointmentHandling,
    GeneratedAgentConfig,
    VoicePreference,
)
from app.services.agent_config_generator import (
    GeminiAgentConfigGenerator,
    InvalidModelOutputError,
    MissingApiKeyError,
    ProviderError,
    get_config_generator,
)

client = TestClient(app)


def sample_valid_config() -> GeneratedAgentConfig:
    return GeneratedAgentConfig(
        agent_name="Aanya",
        business_type="Dental Clinic",
        primary_language="hi",
        additional_languages=["en"],
        first_message="Namaste! Sharma Dental Clinic mein aapka swagat hai.",
        system_prompt="You are Aanya, a polite receptionist at Sharma Dental Clinic...",
        voice_preference=VoicePreference(
            gender="female",
            accent="Indian English / Hindi",
            description="Warm and professional tone",
        ),
        intended_capabilities=["answer_faqs", "appointment_requests"],
        faqs=[],
        appointment_handling=AppointmentHandling(
            enabled=True,
            instructions="Collect requested date, time, and service. Inform caller staff will confirm via call/SMS.",
        ),
    )


def test_configure_agent_success():
    class MockSuccessGenerator:
        def generate_config(self, request):
            return sample_valid_config()

    app.dependency_overrides[get_config_generator] = lambda: MockSuccessGenerator()

    try:
        response = client.post(
            "/api/agents/configure",
            json={
                "prompt": "Create a friendly bilingual receptionist for a dental clinic who answers common questions and helps patients request appointments.",
                "preferences": {
                    "language": "hi",
                    "additional_languages": ["en"],
                    "business_type": "Dental Clinic",
                    "voice_gender": "female",
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "configuration" in data
        assert data["configuration"]["agent_name"] == "Aanya"
        assert data["configuration"]["primary_language"] == "hi"
        assert data["configuration"]["appointment_handling"]["enabled"] is True

        # Check ElevenLabs preview mapping
        preview = data["elevenlabs_config_preview"]
        assert preview["agent"]["first_message"] == "Namaste! Sharma Dental Clinic mein aapka swagat hai."
        assert preview["agent"]["hinglish_mode"] is True
        assert "prompt" in preview["agent"]["prompt"]
    finally:
        app.dependency_overrides.clear()


def test_configure_agent_missing_api_key():
    class MockMissingKeyGenerator:
        def generate_config(self, request):
            raise MissingApiKeyError("Gemini API key is not configured on the server.")

    app.dependency_overrides[get_config_generator] = lambda: MockMissingKeyGenerator()

    try:
        response = client.post(
            "/api/agents/configure",
            json={"prompt": "Create a clinic assistant."},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Gemini API key is not configured on the server."
        # Ensure no credential fragments are leaked
        assert "AIza" not in response.text
    finally:
        app.dependency_overrides.clear()


def test_configure_agent_provider_failure():
    class MockProviderErrorGenerator:
        def generate_config(self, request):
            raise ProviderError("Upstream AI provider error occurred while generating agent configuration.")

    app.dependency_overrides[get_config_generator] = lambda: MockProviderErrorGenerator()

    try:
        response = client.post(
            "/api/agents/configure",
            json={"prompt": "Create a salon booking assistant."},
        )
        assert response.status_code == 502
        data = response.json()
        assert data["detail"] == "Upstream AI provider error occurred while generating agent configuration."
        # Ensure no stack trace in response body
        assert "Traceback" not in response.text
    finally:
        app.dependency_overrides.clear()


def test_configure_agent_malformed_output():
    class MockMalformedOutputGenerator:
        def generate_config(self, request):
            raise InvalidModelOutputError("Upstream AI provider returned an invalid configuration payload.")

    app.dependency_overrides[get_config_generator] = lambda: MockMalformedOutputGenerator()

    try:
        response = client.post(
            "/api/agents/configure",
            json={"prompt": "Create a repair shop helper."},
        )
        assert response.status_code == 502
        data = response.json()
        assert data["detail"] == "Upstream AI provider returned an invalid configuration payload."
    finally:
        app.dependency_overrides.clear()


def test_configure_agent_empty_prompt_validation():
    # FastAPI/Pydantic validation should reject prompts with min_length < 3 with HTTP 422
    response = client.post(
        "/api/agents/configure",
        json={"prompt": ""},
    )
    assert response.status_code == 422

    response_missing = client.post(
        "/api/agents/configure",
        json={},
    )
    assert response_missing.status_code == 422


def test_service_unit_missing_key_detection():
    # Direct unit test on service class verifying it aborts safely without network calls
    gen = GeminiAgentConfigGenerator(api_key=None)
    with pytest.raises(MissingApiKeyError):
        gen.generate_config(request=type("Req", (), {"prompt": "test", "preferences": None})())

    gen_empty = GeminiAgentConfigGenerator(api_key="   ")
    with pytest.raises(MissingApiKeyError):
        gen_empty.generate_config(request=type("Req", (), {"prompt": "test", "preferences": None})())


def test_existing_endpoints_preserved():
    # Verify health and root endpoints still pass
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok", "service": "vaaniflow-backend"}

    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "running"


def test_configure_agent_english_primary_success():
    def english_config():
        return GeneratedAgentConfig(
            agent_name="Sarah",
            business_type="Dental Clinic",
            primary_language="en",
            additional_languages=["hi"],
            first_message="Hello! Welcome to Dr. Smile Dental Clinic. How can I help you?",
            system_prompt="You are Sarah, a friendly receptionist for Dr. Smile Dental Clinic.",
            voice_preference=VoicePreference(
                gender="female",
                accent="Indian English",
                description="Professional receptionist",
            ),
            intended_capabilities=["answer_faqs", "appointment_requests"],
            faqs=[],
            appointment_handling=AppointmentHandling(enabled=True),
        )

    class MockEnglishGenerator:
        def generate_config(self, request):
            return english_config()

    app.dependency_overrides[get_config_generator] = lambda: MockEnglishGenerator()
    try:
        response = client.post(
            "/api/agents/configure",
            json={
                "prompt": "Create an English front-desk assistant for our clinic.",
                "preferences": {
                    "language": "en",
                    "additional_languages": ["hi"],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        cfg = data["configuration"]
        assert cfg["primary_language"] == "en"
        assert cfg["additional_languages"] == ["hi"]

        # Preview verification
        preview = data["elevenlabs_config_preview"]
        assert preview["agent"]["language"] == "en"
        assert "hinglish_mode" not in preview["agent"]
        assert preview["tts"]["model_id"] == "eleven_turbo_v2"
    finally:
        app.dependency_overrides.clear()


def test_configure_agent_hinglish_preference_normalization():
    def hinglish_config():
        return GeneratedAgentConfig(
            agent_name="Pooja",
            business_type="Wellness Clinic",
            primary_language="Hinglish",
            additional_languages=["English"],
            first_message="Namaste! Kaise madad kar sakti hoon?",
            system_prompt="You are Pooja, bilingual receptionist.",
        )

    class MockHinglishGenerator:
        def generate_config(self, request):
            assert request.preferences.language == "hi"
            return hinglish_config()

    app.dependency_overrides[get_config_generator] = lambda: MockHinglishGenerator()
    try:
        response = client.post(
            "/api/agents/configure",
            json={
                "prompt": "Need a Hinglish speaking assistant for wellness clinic.",
                "preferences": {
                    "language": "Hinglish",
                    "additional_languages": ["English"],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        cfg = data["configuration"]
        assert cfg["primary_language"] == "hi"
        assert cfg["additional_languages"] == ["en"]

        preview = data["elevenlabs_config_preview"]
        assert preview["agent"]["language"] == "hi"
        assert preview["agent"]["hinglish_mode"] is True
        assert preview["tts"]["model_id"] == "eleven_turbo_v2_5"
    finally:
        app.dependency_overrides.clear()


def test_configure_agent_whitespace_and_mixed_case_languages():
    def dirty_lang_config():
        return GeneratedAgentConfig(
            agent_name="Aman",
            business_type="Auto Repair",
            primary_language="  HI  ",
            additional_languages=["  EN  ", "HINDI"],
            first_message="Hello, welcome!",
            system_prompt="You are Aman, auto assistant.",
        )

    class MockDirtyGenerator:
        def generate_config(self, request):
            return dirty_lang_config()

    app.dependency_overrides[get_config_generator] = lambda: MockDirtyGenerator()
    try:
        response = client.post(
            "/api/agents/configure",
            json={
                "prompt": "Create an assistant for auto repair shop.",
                "preferences": {
                    "language": "  hi-IN  ",
                    "additional_languages": ["  en-US  "],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        cfg = data["configuration"]
        assert cfg["primary_language"] == "hi"
        assert cfg["additional_languages"] == ["en"]
    finally:
        app.dependency_overrides.clear()


def test_configure_agent_whitespace_only_prompt_rejected():
    response = client.post(
        "/api/agents/configure",
        json={"prompt": "     "},
    )
    assert response.status_code == 422


def test_generator_parses_markdown_code_fences_safely(monkeypatch):
    from google import genai
    from app.schemas.agent_config import AgentConfigureRequest

    raw_json_str = '''```json
{
  "agent_name": "Dr. Smile Assistant",
  "business_type": "Dental Clinic",
  "primary_language": "hi",
  "additional_languages": ["en"],
  "first_message": "Namaste! Dr. Smile Clinic mein swagat hai.",
  "system_prompt": "You are a courteous receptionist.",
  "voice_preference": {
    "gender": "female",
    "accent": "Indian",
    "description": "Warm tone",
    "voice_id": null
  },
  "intended_capabilities": ["answer_faqs"],
  "faqs": [],
  "appointment_handling": {
    "enabled": true,
    "instructions": "Record appointment request."
  }
}
```'''

    class MockGenaiResponse:
        text = raw_json_str

    class MockModels:
        def generate_content(self, *args, **kwargs):
            return MockGenaiResponse()

    class MockGenaiClient:
        def __init__(self, *args, **kwargs):
            self.models = MockModels()

    monkeypatch.setattr(genai, "Client", MockGenaiClient)

    generator = GeminiAgentConfigGenerator(api_key="mock_key")
    req = AgentConfigureRequest(prompt="Create a dental assistant.")
    result = generator.generate_config(req)
    assert result.agent_name == "Dr. Smile Assistant"
    assert result.primary_language == "hi"
    assert result.additional_languages == ["en"]


def test_generator_timeout_raises_provider_error(monkeypatch):
    import httpx
    from google import genai
    from app.schemas.agent_config import AgentConfigureRequest

    class MockModels:
        def generate_content(self, *args, **kwargs):
            raise httpx.ReadTimeout("Gemini connection timed out")

    class MockGenaiClient:
        def __init__(self, *args, **kwargs):
            self.models = MockModels()

    monkeypatch.setattr(genai, "Client", MockGenaiClient)

    generator = GeminiAgentConfigGenerator(api_key="mock_key")
    req = AgentConfigureRequest(prompt="Create a salon assistant.")
    with pytest.raises(ProviderError) as exc_info:
        generator.generate_config(req)
    assert "Upstream AI provider error" in str(exc_info.value)


def test_configure_agent_unsupported_language_returns_422():
    response = client.post(
        "/api/agents/configure",
        json={
            "prompt": "Create an assistant.",
            "preferences": {
                "language": "french",
            },
        },
    )
    assert response.status_code == 422
    assert "Unsupported language 'french'" in response.text


def test_configure_agent_unsupported_additional_language_returns_422():
    response = client.post(
        "/api/agents/configure",
        json={
            "prompt": "Create an assistant.",
            "preferences": {
                "language": "en",
                "additional_languages": ["spanish"],
            },
        },
    )
    assert response.status_code == 422
    assert "Unsupported language 'spanish'" in response.text
