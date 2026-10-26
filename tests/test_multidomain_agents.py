import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.repositories.agent_repository import AgentRepository, get_agent_repository
from app.schemas.agent_config import (
    AppointmentHandling,
    GeneratedAgentConfig,
    VoicePreference,
)
from app.services.agent_config_generator import (
    GeminiAgentConfigGenerator,
    get_config_generator,
)
from app.services.elevenlabs_agent_service import (
    ElevenLabsAgentService,
    get_elevenlabs_service,
)

client = TestClient(app)


def build_ecommerce_config(lang: str = "en") -> GeneratedAgentConfig:
    if lang == "hi":
        return GeneratedAgentConfig(
            agent_name="TrackFast Sahayak",
            business_type="E-Commerce Logistics",
            primary_language="hi",
            additional_languages=["en"],
            first_message="Namaste! TrackFast Delivery mein aapka swagat hai. Main aapke parcel ya order ki kya sahayata kar sakta hoon?",
            system_prompt=(
                "You are TrackFast Sahayak, a helpful customer service assistant for TrackFast Delivery. "
                "Help callers check parcel statuses, understand delivery timelines, and process return requests. "
                "Ground all answers in provided tracking details. Never fabricate delivery status without an order ID. "
                "Speak naturally in Hindi and Hinglish."
            ),
            voice_preference=VoicePreference(
                gender="male",
                accent="Indian Hindi",
                description="Helpful and clear customer support voice",
            ),
            intended_capabilities=["track_order", "shipping_inquiry", "return_policy_explanation"],
            faqs=[],
            appointment_handling=AppointmentHandling(enabled=False),
        )
    return GeneratedAgentConfig(
        agent_name="TrackFast Support",
        business_type="E-Commerce Logistics",
        primary_language="en",
        additional_languages=["hi"],
        first_message="Hello! Thank you for contacting TrackFast Delivery. How can I assist you with your shipment today?",
        system_prompt=(
            "You are TrackFast Support, an efficient logistics customer service assistant. "
            "Help callers track packages, explain delivery schedules, and initiate return inquiries. "
            "Ask for the tracking number or order ID when missing. Never invent delivery status or promise unrealistic delivery times."
        ),
        voice_preference=VoicePreference(
            gender="female",
            accent="Indian English",
            description="Polite and concise logistics assistant",
        ),
        intended_capabilities=["track_order", "shipping_inquiry", "return_policy_explanation"],
        faqs=[],
        appointment_handling=AppointmentHandling(enabled=False),
    )


def build_edtech_config(lang: str = "en") -> GeneratedAgentConfig:
    if lang == "hi":
        return GeneratedAgentConfig(
            agent_name="CodeMitra Python Guru",
            business_type="EdTech / Programming Education",
            primary_language="hi",
            additional_languages=["en"],
            first_message="Namaste! Main hoon CodeMitra, aapka Python coding guide. Aaj hum kaunsa programming concept seekhenge?",
            system_prompt=(
                "You are CodeMitra, a patient and encouraging Python programming tutor for beginners. "
                "Explain concepts such as loops, functions, and lists using simple real-world analogies. "
                "Never give complete answers immediately; provide scaffolding and debugging hints. "
                "Speak in conversational Hindi and Hinglish."
            ),
            voice_preference=VoicePreference(
                gender="male",
                accent="Indian Hindi",
                description="Friendly mentor tone",
            ),
            intended_capabilities=["explain_coding_concepts", "debugging_hints", "practice_exercises"],
            faqs=[],
            appointment_handling=AppointmentHandling(enabled=False),
        )
    return GeneratedAgentConfig(
        agent_name="CodeMitra Python Tutor",
        business_type="EdTech / Programming Education",
        primary_language="en",
        additional_languages=["hi"],
        first_message="Hi there! I'm CodeMitra, your friendly Python learning assistant. What programming concept or coding problem are we tackling today?",
        system_prompt=(
            "You are CodeMitra, a patient programming mentor for beginners. "
            "Guide students through Python concepts, syntax errors, and algorithmic thinking. "
            "Encourage students by asking guiding questions rather than writing full solutions for them. "
            "Maintain an encouraging, curious, and pedagogical tone."
        ),
        voice_preference=VoicePreference(
            gender="female",
            accent="Indian English",
            description="Warm, patient teacher voice",
        ),
        intended_capabilities=["explain_coding_concepts", "debugging_hints", "practice_exercises"],
        faqs=[],
        appointment_handling=AppointmentHandling(enabled=False),
    )


def test_ecommerce_support_agent_english_and_hindi_configurations():
    # 1. English E-Commerce Agent
    cfg_en = build_ecommerce_config(lang="en")
    preview_en = cfg_en.to_elevenlabs_conversation_config_preview()
    assert cfg_en.business_type == "E-Commerce Logistics"
    assert cfg_en.appointment_handling.enabled is False
    assert preview_en["agent"]["language"] == "en"
    assert "hinglish_mode" not in preview_en["agent"]
    assert preview_en["tts"]["model_id"] == "eleven_turbo_v2"

    svc = ElevenLabsAgentService(api_key="mock_key")
    payload_en = svc.build_create_payload(cfg_en, voice_id="v_logistics_en")
    assert payload_en["conversation_config"]["agent"]["language"] == "en"
    assert "hinglish_mode" not in payload_en["conversation_config"]["agent"]
    assert payload_en["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2"

    # 2. Hindi E-Commerce Agent
    cfg_hi = build_ecommerce_config(lang="hi")
    preview_hi = cfg_hi.to_elevenlabs_conversation_config_preview()
    assert cfg_hi.business_type == "E-Commerce Logistics"
    assert cfg_hi.appointment_handling.enabled is False
    assert preview_hi["agent"]["language"] == "hi"
    assert preview_hi["agent"]["hinglish_mode"] is True
    assert preview_hi["tts"]["model_id"] == "eleven_turbo_v2_5"

    payload_hi = svc.build_create_payload(cfg_hi, voice_id="v_logistics_hi")
    assert payload_hi["conversation_config"]["agent"]["language"] == "hi"
    assert payload_hi["conversation_config"]["agent"]["hinglish_mode"] is True
    assert payload_hi["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2_5"


def test_edtech_coding_tutor_agent_english_and_hindi_configurations():
    # 1. English EdTech Agent
    cfg_en = build_edtech_config(lang="en")
    preview_en = cfg_en.to_elevenlabs_conversation_config_preview()
    assert cfg_en.business_type == "EdTech / Programming Education"
    assert cfg_en.appointment_handling.enabled is False
    assert preview_en["agent"]["language"] == "en"
    assert "hinglish_mode" not in preview_en["agent"]
    assert preview_en["tts"]["model_id"] == "eleven_turbo_v2"

    svc = ElevenLabsAgentService(api_key="mock_key")
    payload_en = svc.build_create_payload(cfg_en, voice_id="v_edtech_en")
    assert payload_en["conversation_config"]["agent"]["language"] == "en"
    assert payload_en["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2"

    # 2. Hindi EdTech Agent
    cfg_hi = build_edtech_config(lang="hi")
    preview_hi = cfg_hi.to_elevenlabs_conversation_config_preview()
    assert cfg_hi.business_type == "EdTech / Programming Education"
    assert cfg_hi.appointment_handling.enabled is False
    assert preview_hi["agent"]["language"] == "hi"
    assert preview_hi["agent"]["hinglish_mode"] is True
    assert preview_hi["tts"]["model_id"] == "eleven_turbo_v2_5"

    payload_hi = svc.build_create_payload(cfg_hi, voice_id="v_edtech_hi")
    assert payload_hi["conversation_config"]["agent"]["language"] == "hi"
    assert payload_hi["conversation_config"]["agent"]["hinglish_mode"] is True
    assert payload_hi["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2_5"




def build_fleet_dispatch_config(lang: str = "en") -> GeneratedAgentConfig:
    if lang == "hi":
        return GeneratedAgentConfig(
            agent_name="FleetDesk Sahayak",
            business_type="Truck Fleet Logistics & Dispatch",
            primary_language="hi",
            additional_languages=["en"],
            first_message="Namaste! FleetDesk Dispatch mein aapka swagat hai. Main aapke truck route ya fuel stop ki kya sahayata kar sakta hoon?",
            system_prompt=(
                "You are FleetDesk Sahayak, an AI logistics dispatch assistant for commercial truck drivers. "
                "Help drivers verify route assignments, report fuel stop locations, and log transit delays. "
                "Always require truck number or trip ID before confirming dispatch changes. "
                "Never invent route clearance without verified dispatcher log data. "
                "Speak naturally in Hindi and Hinglish."
            ),
            voice_preference=VoicePreference(
                gender="male",
                accent="Indian Hindi",
                description="Calm, clear radio dispatch tone",
            ),
            intended_capabilities=["route_verification", "fuel_stop_lookup", "delay_reporting"],
            faqs=[],
            appointment_handling=AppointmentHandling(enabled=False),
        )
    return GeneratedAgentConfig(
        agent_name="FleetDesk Agent",
        business_type="Truck Fleet Logistics & Dispatch",
        primary_language="en",
        additional_languages=["hi"],
        first_message="Hello! FleetDesk Dispatch, how can I assist your route or fuel stop today?",
        system_prompt=(
            "You are FleetDesk Agent, an automated dispatch communication assistant for freight drivers. "
            "Help drivers verify assigned routes, locate authorized fuel stops, and report cargo delays. "
            "Ask for the vehicle unit number or trip ID when missing. Never invent route clearance without dispatcher verification."
        ),
        voice_preference=VoicePreference(
            gender="male",
            accent="Indian English",
            description="Crisp, authoritative dispatch voice",
        ),
        intended_capabilities=["route_verification", "fuel_stop_lookup", "delay_reporting"],
        faqs=[],
        appointment_handling=AppointmentHandling(enabled=False),
    )


def test_fleet_dispatch_agent_english_and_hindi_configurations():
    # 1. English Fleet Dispatch Agent
    cfg_en = build_fleet_dispatch_config(lang="en")
    preview_en = cfg_en.to_elevenlabs_conversation_config_preview()
    assert cfg_en.business_type == "Truck Fleet Logistics & Dispatch"
    assert cfg_en.appointment_handling.enabled is False
    assert preview_en["agent"]["language"] == "en"
    assert "hinglish_mode" not in preview_en["agent"]
    assert preview_en["tts"]["model_id"] == "eleven_turbo_v2"

    svc = ElevenLabsAgentService(api_key="mock_key")
    payload_en = svc.build_create_payload(cfg_en, voice_id="v_fleet_en")
    assert payload_en["conversation_config"]["agent"]["language"] == "en"
    assert payload_en["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2"
    assert "hinglish_mode" not in payload_en["conversation_config"]["agent"]

    # 2. Hindi Fleet Dispatch Agent
    cfg_hi = build_fleet_dispatch_config(lang="hi")
    preview_hi = cfg_hi.to_elevenlabs_conversation_config_preview()
    assert cfg_hi.business_type == "Truck Fleet Logistics & Dispatch"
    assert cfg_hi.appointment_handling.enabled is False
    assert preview_hi["agent"]["language"] == "hi"
    assert preview_hi["agent"]["hinglish_mode"] is True
    assert preview_hi["tts"]["model_id"] == "eleven_turbo_v2_5"

    payload_hi = svc.build_create_payload(cfg_hi, voice_id="v_fleet_hi")
    assert payload_hi["conversation_config"]["agent"]["language"] == "hi"
    assert payload_hi["conversation_config"]["agent"]["hinglish_mode"] is True
    assert payload_hi["conversation_config"]["tts"]["model_id"] == "eleven_turbo_v2_5"


def test_distinct_use_cases_differ_appropriately_and_omit_receptionist_assumptions():
    ecommerce = build_ecommerce_config(lang="en")
    edtech = build_edtech_config(lang="en")
    fleet = build_fleet_dispatch_config(lang="en")

    # All three domains must have completely distinct names, business types, prompts, messages, and capabilities
    all_names = {ecommerce.agent_name, edtech.agent_name, fleet.agent_name}
    assert len(all_names) == 3

    all_domains = {ecommerce.business_type, edtech.business_type, fleet.business_type}
    assert len(all_domains) == 3

    all_prompts = {ecommerce.system_prompt, edtech.system_prompt, fleet.system_prompt}
    assert len(all_prompts) == 3

    all_messages = {ecommerce.first_message, edtech.first_message, fleet.first_message}
    assert len(all_messages) == 3

    # Verify domain-specific terms are not conflated
    for agent in [ecommerce, edtech, fleet]:
        assert "receptionist" not in agent.system_prompt.lower()
        assert "clinic" not in agent.system_prompt.lower()
        # Non-scheduling domains have appointment_handling disabled
        assert agent.appointment_handling.enabled is False


def test_multidomain_agents_create_and_persist_correctly():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_multidomain.db")
        temp_repo = AgentRepository(db_path=db_path)

        created_agents = []

        class MockMultiDomainService:
            def create_agent(self, config, explicit_voice_id=None):
                agent_id = f"agent_{config.agent_name.lower().replace(' ', '_')}_{config.primary_language}"
                created_agents.append((agent_id, config.agent_name, config.business_type, config.primary_language))
                return {
                    "agent_id": agent_id,
                    "voice_id": explicit_voice_id or "voice_default",
                }

        app.dependency_overrides[get_agent_repository] = lambda: temp_repo
        app.dependency_overrides[get_elevenlabs_service] = lambda: MockMultiDomainService()

        try:
            # 1. Create E-Commerce Agent (English)
            ecom_cfg = build_ecommerce_config(lang="en")
            res_ecom = client.post(
                "/api/agents",
                json={"configuration": ecom_cfg.model_dump(), "voice_id": "v_ecom"},
            )
            assert res_ecom.status_code == 201
            assert res_ecom.json()["name"] == "TrackFast Support"
            assert res_ecom.json()["business_type"] == "E-Commerce Logistics"

            # 2. Create EdTech Agent (Hindi)
            edtech_cfg = build_edtech_config(lang="hi")
            res_edtech = client.post(
                "/api/agents",
                json={"configuration": edtech_cfg.model_dump(), "voice_id": "v_edtech"},
            )
            assert res_edtech.status_code == 201
            assert res_edtech.json()["name"] == "CodeMitra Python Guru"
            assert res_edtech.json()["business_type"] == "EdTech / Programming Education"

            # 3. Verify SQLite DB persists both distinct records with accurate domain data
            records = temp_repo.list_agents()
            assert len(records) == 2
            business_types = {r["business_type"] for r in records}
            assert "E-Commerce Logistics" in business_types
            assert "EdTech / Programming Education" in business_types
        finally:
            app.dependency_overrides.clear()


def test_prompt_generator_passes_user_domain_context_to_gemini(monkeypatch):
    from google import genai
    from app.schemas.agent_config import AgentConfigureRequest, AgentPreferences

    captured_requests = []

    class MockModels:
        def generate_content(self, model, contents, config):
            captured_requests.append({
                "model": model,
                "contents": contents,
                "system_instruction": config.system_instruction,
            })
            # Return valid JSON matching schema
            return type("Resp", (), {
                "text": """{
                    "agent_name": "FleetDesk Agent",
                    "business_type": "Truck Fleet Dispatch",
                    "primary_language": "en",
                    "additional_languages": ["hi"],
                    "first_message": "Hello! FleetDesk Dispatch, how can I assist your route today?",
                    "system_prompt": "You are FleetDesk Agent, a logistics dispatch assistant.",
                    "voice_preference": {"gender": "male", "accent": "Neutral"},
                    "intended_capabilities": ["route_check", "fuel_report"],
                    "faqs": [],
                    "appointment_handling": {"enabled": false, "instructions": "No appointments."}
                }"""
            })()

    class MockGenaiClient:
        def __init__(self, *args, **kwargs):
            self.models = MockModels()

    monkeypatch.setattr(genai, "Client", MockGenaiClient)

    gen = GeminiAgentConfigGenerator(api_key="mock_key")
    req = AgentConfigureRequest(
        prompt="Create a truck fleet dispatch voice agent that answers drivers about routes and fuel stops.",
        preferences=AgentPreferences(
            business_type="Truck Fleet Dispatch",
            language="en",
            capabilities=["route_check", "fuel_report"],
        ),
    )
    res = gen.generate_config(req)
    assert res.agent_name == "FleetDesk Agent"
    assert res.business_type == "Truck Fleet Dispatch"
    assert res.appointment_handling.enabled is False

    # Verify context passed to Gemini includes user's specific domain
    assert len(captured_requests) == 1
    contents = captured_requests[0]["contents"]
    assert "truck fleet dispatch" in contents.lower()
    assert "Truck Fleet Dispatch" in contents
    assert "route_check" in contents

    # Verify system instruction contains general-purpose guidelines, not hardcoded receptionist clinic text
    sys_inst = captured_requests[0]["system_instruction"]
    assert "general-purpose voice-agent creation platform" in sys_inst
    assert "Dynamic Role & Domain Adaptation" in sys_inst
    assert "Do NOT force all agents into a receptionist or clinic pattern" in sys_inst
