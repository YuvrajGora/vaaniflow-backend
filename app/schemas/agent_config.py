from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


SUPPORTED_ENGLISH_VARIANTS = {
    "en",
    "english",
    "en-us",
    "en-in",
    "en-gb",
    "en_us",
    "en_in",
    "en_gb",
}

SUPPORTED_HINDI_VARIANTS = {
    "hi",
    "hindi",
    "hinglish",
    "hi-in",
    "hi_in",
}


def normalize_language_code(lang: str) -> str:
    """Validates and normalizes supported English and Hindi/Hinglish language inputs to canonical 'en' or 'hi'.

    Supported inputs: 'en', 'English', 'en-US', 'en-IN', 'hi', 'Hindi', 'hi-IN', 'Hinglish'.
    Raises ValueError for unsupported, empty, or malformed language values.
    """
    if not isinstance(lang, str):
        raise ValueError(f"Language must be a string, got {type(lang).__name__}.")
    clean = lang.strip().lower()
    if clean in SUPPORTED_ENGLISH_VARIANTS:
        return "en"
    if clean in SUPPORTED_HINDI_VARIANTS:
        return "hi"
    raise ValueError(
        f"Unsupported language '{lang}'. VaaniFlow AI supports English ('en', 'English', 'en-US') and Hindi/Hinglish ('hi', 'Hindi', 'hi-IN', 'Hinglish')."
    )


class AgentPreferences(BaseModel):
    language: Optional[str] = Field(default=None, description="Preferred primary language, e.g., 'hi' or 'en'")
    additional_languages: Optional[List[str]] = Field(default=None, description="Optional extra languages, e.g., ['en']")
    tone: Optional[str] = Field(default=None, description="Desired tone, e.g., 'warm, courteous, professional'")
    business_type: Optional[str] = Field(default=None, description="Domain, industry, organization, or use-case category, e.g., 'e-commerce', 'education', 'healthcare', 'fintech'")
    voice_gender: Optional[str] = Field(default=None, description="Voice gender preference, e.g., 'female' or 'male'")
    voice_accent: Optional[str] = Field(default=None, description="Accent preference, e.g., 'Indian English'")
    voice_id: Optional[str] = Field(default=None, description="Explicit ElevenLabs voice ID if known; never fabricated")
    capabilities: Optional[List[str]] = Field(
        default=None,
        description="Explicit capabilities requested, e.g., ['answer_faqs', 'appointment_requests']"
    )

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return normalize_language_code(str(v))

    @field_validator("additional_languages", mode="before")
    @classmethod
    def validate_additional_languages(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if not isinstance(v, list):
            return []
        res = []
        for item in v:
            norm = normalize_language_code(str(item))
            if norm not in res:
                res.append(norm)
        return res


    @model_validator(mode="after")
    def clean_additional_languages(self) -> "AgentPreferences":
        if self.additional_languages and self.language:
            self.additional_languages = [
                lang for lang in self.additional_languages if lang != self.language
            ]
        return self


class AgentConfigureRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Natural language description of the voice agent")
    preferences: Optional[AgentPreferences] = Field(default=None, description="Optional structured preferences")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        cleaned = v.strip() if isinstance(v, str) else ""
        if len(cleaned) < 3:
            raise ValueError("Prompt must contain at least 3 non-whitespace characters.")
        return cleaned


class VoicePreference(BaseModel):
    gender: Optional[str] = Field(default=None, description="Voice gender preference")
    accent: Optional[str] = Field(default=None, description="Accent or dialect style")
    description: Optional[str] = Field(default=None, description="Description of desired voice traits")
    voice_id: Optional[str] = Field(
        default=None,
        description="ElevenLabs voice ID only when explicitly supplied; never fabricated"
    )


class FAQItem(BaseModel):
    question: str = Field(..., description="Customer question")
    answer: str = Field(..., description="Verified answer grounded in input")


class AppointmentHandling(BaseModel):
    enabled: bool = Field(default=False, description="Whether the agent handles appointments, bookings, or reservations")
    instructions: str = Field(
        default="If scheduling or booking is enabled for this domain, collect necessary details and clarify confirmation policies appropriately.",
        description="Guidelines on appointment, booking, or reservation request collection vs final confirmation"
    )


class GeneratedAgentConfig(BaseModel):
    agent_name: str = Field(..., description="Name for the voice agent")
    business_type: str = Field(..., description="Domain, industry, organization, or primary use-case category")
    primary_language: str = Field(default="en", description="Primary language code, e.g. 'en' or 'hi'")
    additional_languages: List[str] = Field(default_factory=list, description="Supported secondary languages")
    first_message: str = Field(..., description="First greeting message the agent speaks when a call connects")
    system_prompt: str = Field(..., description="Comprehensive system prompt for the conversational voice agent")
    voice_preference: VoicePreference = Field(default_factory=VoicePreference, description="Voice style and characteristics")
    intended_capabilities: List[str] = Field(default_factory=list, description="List of capabilities supported by the agent")
    faqs: List[FAQItem] = Field(default_factory=list, description="Verified FAQs provided by the user; never fabricated")
    appointment_handling: AppointmentHandling = Field(default_factory=AppointmentHandling, description="Appointment, booking, or reservation request rules if applicable")

    @field_validator("primary_language", mode="before")
    @classmethod
    def validate_primary_language(cls, v: Any) -> str:
        return normalize_language_code(str(v) if v is not None else "en")

    @field_validator("additional_languages", mode="before")
    @classmethod
    def validate_additional_languages(cls, v: Any) -> List[str]:
        if not v or not isinstance(v, list):
            return []
        res = []
        for item in v:
            norm = normalize_language_code(str(item))
            if norm not in res:
                res.append(norm)
        return res

    @model_validator(mode="after")
    def clean_additional_languages(self) -> "GeneratedAgentConfig":
        if self.additional_languages and self.primary_language:
            self.additional_languages = [
                lang for lang in self.additional_languages if lang != self.primary_language
            ]
        return self

    def to_elevenlabs_conversation_config_preview(self) -> Dict[str, Any]:
        """Produces an informational preview of the ElevenLabs conversation_config payload.

        Unresolved voice IDs are represented as None rather than placeholder strings.
        Actual agent creation independently constructs and validates the provider payload.
        """
        primary_lang = normalize_language_code(self.primary_language)
        is_hinglish = primary_lang == "hi"
        tts_model_id = "eleven_turbo_v2" if primary_lang == "en" else "eleven_turbo_v2_5"

        agent_dict: Dict[str, Any] = {
            "prompt": {
                "prompt": self.system_prompt
            },
            "first_message": self.first_message,
            "language": primary_lang,
        }
        if is_hinglish:
            agent_dict["hinglish_mode"] = True

        return {
            "agent": agent_dict,
            "tts": {
                "voice_id": self.voice_preference.voice_id or None,
                "model_id": tts_model_id,
            },
            "conversation": {
                "max_duration_seconds": 600,
            }
        }


class AgentSummary(BaseModel):
    id: int
    agent_id: str
    name: str
    business_type: str
    voice_id: str
    created_at: str


class AgentListResponse(BaseModel):
    status: str = "ok"
    agents: List[AgentSummary] = Field(default_factory=list)
    total: int = 0


class AgentConfigureResponse(BaseModel):
    status: str = "ok"
    configuration: GeneratedAgentConfig
    elevenlabs_config_preview: Dict[str, Any]


class AgentCreateRequest(BaseModel):
    configuration: GeneratedAgentConfig = Field(..., description="Previously generated agent configuration")
    voice_id: Optional[str] = Field(default=None, description="Optional explicit voice ID to override configuration")


class AgentCreateResponse(BaseModel):
    status: str = "created"
    agent_id: str = Field(..., description="Real ElevenLabs agent ID")
    name: str = Field(..., description="Agent name")
    business_type: str = Field(..., description="Business or use-case category")
    voice_id: str = Field(..., description="Resolved ElevenLabs voice ID used for creation")
    created_at: str = Field(..., description="Creation timestamp in ISO 8601 UTC")


class AgentSessionResponse(BaseModel):
    status: str = "ok"
    agent_id: str = Field(..., description="The verified ElevenLabs agent ID")
    signed_url: str = Field(..., description="Temporary signed WebSocket URL for browser voice session")
    expires_in_seconds: int = Field(default=900, description="Validity period in seconds (15 minutes)")
