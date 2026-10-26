import logging
import re
from typing import Any, Dict, Optional
import httpx

from app.core.config import get_settings
from app.schemas.agent_config import GeneratedAgentConfig, normalize_language_code

logger = logging.getLogger(__name__)

ELEVENLABS_CREATE_AGENT_URL = "https://api.elevenlabs.io/v1/convai/agents/create"
ELEVENLABS_SIGNED_URL_ENDPOINT = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"

URL_PATTERN = re.compile(r"(?i)\b(?:https?|wss?)://\S+")
SAFE_FIELD_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*$")
SENSITIVE_KEYWORD_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|key|keys|token|tokens|bearer|secret|secrets|credential|credentials|password|passwords|auth|authorization|signature|signatures|session_id|cookie|cookies)\b"
)
QUOTED_VALUE_PATTERN = re.compile(r"""['"][^'"]*['"]""")
TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_-]{16,}\b")
# Matches short (8-15 char) mixed alphanumeric tokens that look opaque (contain both
# a letter and a digit). Pure alphabetic words are not affected.
SHORT_OPAQUE_TOKEN_PATTERN = re.compile(
    r"\b(?=[a-zA-Z0-9_-]{8,15}\b)(?=[a-zA-Z0-9_-]*[a-zA-Z])(?=[a-zA-Z0-9_-]*[0-9])[a-zA-Z0-9_-]{8,15}\b"
)
SAFE_MSG_CHARS_PATTERN = re.compile(r"^[a-zA-Z0-9_.,:; '\-\[\]\(\)]+$")

# Maximum allowed length for a provider-supplied field path (dot-joined loc segments).
MAX_FIELD_PATH_LEN = 200

ALLOWLISTED_ERROR_TYPES = {
    "value_error",
    "type_error",
    "missing",
    "extra_forbidden",
    "string_type",
    "bool_parsing",
    "int_parsing",
    "float_parsing",
    "list_type",
    "dict_type",
    "string_too_short",
    "string_too_long",
    "validation_error",
    "provider_validation_error",
}

ALLOWLISTED_ERROR_CATEGORIES = {
    "voice_not_found": "voice_not_found",
    "invalid_voice_id": "invalid_voice_id",
    "voice_id_is_not_valid": "invalid_voice_id",
    "agent_not_found": "agent_not_found",
    "invalid_agent_id": "invalid_agent_id",
    "rate_limit_exceeded": "rate_limit_exceeded",
    "too_many_requests": "rate_limit_exceeded",
    "quota_exceeded": "quota_exceeded",
    "model_not_found": "model_not_found",
    "invalid_model_id": "invalid_model_id",
    "unauthorized": "unauthorized",
    "bad_request": "bad_request",
    "concurrency_limit_reached": "concurrency_limit_reached",
    "validation_error": "validation_error",
    "provider_validation_error": "provider_validation_error",
    "invalid_request": "bad_request",
    "invalid_parameters": "bad_request",
    "invalid_parameter": "bad_request",
    "missing_parameter": "bad_request",
    "unprocessable_entity": "validation_error",
    "request_validation_error": "validation_error",
}


class ElevenLabsServiceError(Exception):
    """Base exception for ElevenLabs service failures."""
    pass


class MissingElevenLabsApiKeyError(ElevenLabsServiceError):
    """Raised when ElevenLabs API key is missing."""
    pass


class MissingVoiceIdError(ElevenLabsServiceError):
    """Raised when no voice ID can be resolved from request, configuration, or environment."""
    pass


class ElevenLabsProviderError(ElevenLabsServiceError):
    """Raised when the ElevenLabs API request fails."""
    def __init__(self, message: str, provider_detail: Optional[str] = None):
        super().__init__(message)
        self.provider_detail = provider_detail


class InvalidProviderResponseError(ElevenLabsServiceError):
    """Raised when ElevenLabs returns an unexpected response payload."""
    pass


class ElevenLabsAgentService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_voice_id: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        base_url: str = ELEVENLABS_CREATE_AGENT_URL,
        signed_url_endpoint: str = ELEVENLABS_SIGNED_URL_ENDPOINT,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.elevenlabs_api_key
        self.default_voice_id = default_voice_id if default_voice_id is not None else settings.elevenlabs_default_voice_id
        self._http_client = http_client
        self.base_url = base_url
        self.signed_url_endpoint = signed_url_endpoint

    @staticmethod
    def _normalize_error_type(err_type: str) -> str:
        """Returns the error type if strictly present in the allowlist, otherwise 'validation_error'."""
        if isinstance(err_type, str):
            cleaned = err_type.strip().lower()
            if cleaned in ALLOWLISTED_ERROR_TYPES:
                return cleaned
        return "validation_error"

    @classmethod
    def _sanitize_validation_message(cls, msg: str, err_type: str = "") -> str:
        """Sanitizes a validation message for server diagnostics.

        If sensitive keywords or URLs are present, suppresses the text and falls back to
        the allowlisted safe error type. Masks quoted values to protect short secrets.
        """
        safe_fallback = cls._normalize_error_type(err_type)
        if not msg or not isinstance(msg, str):
            return safe_fallback

        cleaned = re.sub(r"[\r\n\t]+", " ", msg).strip()

        # 1. Suppress if sensitive keywords (key, token, secret, auth, password, etc.) are present
        if SENSITIVE_KEYWORD_PATTERN.search(cleaned):
            return safe_fallback

        # 2. Suppress if URLs are present
        if URL_PATTERN.search(cleaned):
            return safe_fallback

        # 3. Mask quoted values (e.g. 'short_secret' -> '[VALUE]') to protect short credentials
        cleaned = QUOTED_VALUE_PATTERN.sub("'[VALUE]'", cleaned)

        # 4. Mask short opaque tokens: 8-15 char values that mix letters and digits.
        #    Pure alphabetic words are not affected; only mixed letter+digit tokens are
        #    considered suspicious and masked as [REDACTED].
        cleaned = SHORT_OPAQUE_TOKEN_PATTERN.sub("[REDACTED]", cleaned)

        # 5. Mask long opaque tokens (16+ chars)
        cleaned = TOKEN_PATTERN.sub("[REDACTED_TOKEN]", cleaned)

        # 6. Validate characters to prevent formatting or log injection
        if not SAFE_MSG_CHARS_PATTERN.match(cleaned):
            return safe_fallback

        return cleaned[:150].strip()

    @staticmethod
    def _match_allowlisted_category(text: str) -> str:
        """Matches unvalidated provider text against an allowlisted set of known error categories

        using exact matching after conservative normalization.
        """
        if not text or not isinstance(text, str):
            return ""
        cleaned = text.strip().lower().strip(".:;,!?'\"")
        normalized = re.sub(r"[\s\-_]+", "_", cleaned)
        return ALLOWLISTED_ERROR_CATEGORIES.get(normalized, "")

    @classmethod
    def _extract_error_summary(cls, response: httpx.Response) -> str:
        """Safely extracts a concise provider validation/error message from the response body.

        Recognizes structured validation lists, allowlisted categories, and sanitized error messages
        from `detail`, `error`, `message`, or `errors`.
        Always prevents secret leakage and log injection. If a safe specific diagnostic cannot be
        extracted, returns 'provider_validation_error' rather than an empty string.
        """
        try:
            data = response.json()
        except Exception:
            return ""

        if not isinstance(data, dict):
            return ""

        try:
            # 1. Structured validation list in detail, errors, error, or error.details
            raw_list = None
            if isinstance(data.get("detail"), list):
                raw_list = data.get("detail")
            elif isinstance(data.get("errors"), list):
                raw_list = data.get("errors")
            elif isinstance(data.get("error"), list):
                raw_list = data.get("error")
            elif isinstance(data.get("error"), dict) and isinstance(data.get("error").get("details"), list):
                raw_list = data.get("error").get("details")

            if raw_list is not None:
                parts = []
                for item in raw_list:
                    if isinstance(item, dict):
                        loc_val = item.get("loc")
                        if isinstance(loc_val, (list, tuple)):
                            loc_raw = [str(x) for x in loc_val if x is not None and str(x).lower() != "body"]
                        elif isinstance(loc_val, str):
                            loc_raw = [s for s in loc_val.split(".") if s and s.lower() != "body"]
                        else:
                            loc_raw = []

                        loc_candidate = ".".join(loc_raw)
                        # Reject paths exceeding the length limit or containing unsafe characters.
                        loc = (
                            loc_candidate
                            if len(loc_candidate) <= MAX_FIELD_PATH_LEN
                            and SAFE_FIELD_PATH_PATTERN.match(loc_candidate)
                            else ""
                        )
                        raw_type = item.get("type")
                        err_type = str(raw_type).strip() if raw_type is not None else ""

                        raw_msg = item.get("msg") if item.get("msg") is not None else item.get("message")
                        msg_str = "" if raw_msg is None else str(raw_msg)

                        msg = cls._sanitize_validation_message(msg_str, err_type)
                        if loc and msg:
                            parts.append(f"{loc}: {msg}")
                        elif loc:
                            parts.append(f"{loc}: validation_error")
                        elif msg:
                            parts.append(msg)
                    elif isinstance(item, str):
                        cat = cls._match_allowlisted_category(item)
                        if cat:
                            parts.append(cat)
                        else:
                            sanitized = cls._sanitize_validation_message(item, "validation_error")
                            if sanitized:
                                parts.append(sanitized)
                if parts:
                    return "; ".join(parts)[:250]
                return "provider_validation_error"

            # 2. String or nested error representations in detail, error, or message
            candidate_category = None
            candidate_text = None

            # Check detail
            detail = data.get("detail")
            if isinstance(detail, str) and detail.strip():
                candidate_category = cls._match_allowlisted_category(detail)
                candidate_text = detail.strip()
            elif isinstance(detail, dict):
                candidate_category = (
                    cls._match_allowlisted_category(str(detail.get("status", "") or ""))
                    or cls._match_allowlisted_category(str(detail.get("code", "") or ""))
                    or cls._match_allowlisted_category(str(detail.get("message", "") or ""))
                )
                for k in ["message", "msg", "status", "detail", "description"]:
                    val = detail.get(k)
                    if isinstance(val, str) and val.strip():
                        candidate_text = val.strip()
                        break

            # Check error
            error_val = data.get("error")
            if not candidate_category and not candidate_text:
                if isinstance(error_val, str) and error_val.strip():
                    candidate_category = cls._match_allowlisted_category(error_val)
                    candidate_text = error_val.strip()
                elif isinstance(error_val, dict):
                    candidate_category = (
                        cls._match_allowlisted_category(str(error_val.get("status", "") or ""))
                        or cls._match_allowlisted_category(str(error_val.get("code", "") or ""))
                        or cls._match_allowlisted_category(str(error_val.get("message", "") or ""))
                        or cls._match_allowlisted_category(str(error_val.get("type", "") or ""))
                    )
                    for k in ["message", "msg", "status", "detail", "description"]:
                        val = error_val.get(k)
                        if isinstance(val, str) and val.strip():
                            candidate_text = val.strip()
                            break

            # Check top-level message
            msg_val = data.get("message")
            if not candidate_category and not candidate_text:
                if isinstance(msg_val, str) and msg_val.strip():
                    candidate_category = cls._match_allowlisted_category(msg_val)
                    candidate_text = msg_val.strip()

            if candidate_category:
                # If the matched category is generic and a specific message exists, prefer the sanitized message
                if candidate_category in {"bad_request", "validation_error", "provider_validation_error"} and candidate_text:
                    sanitized = cls._sanitize_validation_message(candidate_text, candidate_category)
                    if sanitized and sanitized not in {"validation_error", "provider_validation_error", "bad_request"}:
                        return sanitized
                return candidate_category

            if candidate_text:
                sanitized = cls._sanitize_validation_message(candidate_text, "provider_validation_error")
                return sanitized if sanitized else "provider_validation_error"

            return "provider_validation_error"
        except Exception:
            return "provider_validation_error"

    def resolve_voice_id(
        self,
        config: GeneratedAgentConfig,
        explicit_voice_id: Optional[str] = None,
    ) -> str:
        """Resolves the voice ID in deterministic priority order:

        1. Explicitly supplied voice_id in the creation request.
        2. Voice ID in the generated configuration.
        3. ELEVENLABS_DEFAULT_VOICE_ID from backend environment settings.
        """
        if explicit_voice_id and explicit_voice_id.strip():
            return explicit_voice_id.strip()

        if config.voice_preference and config.voice_preference.voice_id and config.voice_preference.voice_id.strip():
            return config.voice_preference.voice_id.strip()

        if self.default_voice_id and self.default_voice_id.strip():
            return self.default_voice_id.strip()

        raise MissingVoiceIdError(
            "A voice ID must be provided in the request, specified in the agent configuration, or configured via ELEVENLABS_DEFAULT_VOICE_ID."
        )

    def build_create_payload(
        self,
        config: GeneratedAgentConfig,
        voice_id: str,
    ) -> Dict[str, Any]:
        """Builds the official ElevenLabs conversation_config payload."""
        primary_lang = normalize_language_code(config.primary_language)
        # ElevenLabs requires hinglish_mode to only be enabled when primary language is Hindi ('hi').
        # When primary language is English or other languages, hinglish_mode is omitted entirely.
        is_hinglish = primary_lang == "hi"

        # ElevenLabs requires English agents to use turbo or flash v2 ('eleven_turbo_v2').
        # Multilingual agents (e.g. Hindi) use 'eleven_turbo_v2_5'.
        tts_model_id = "eleven_turbo_v2" if primary_lang == "en" else "eleven_turbo_v2_5"

        agent_dict: Dict[str, Any] = {
            "prompt": {
                "prompt": config.system_prompt
            },
            "first_message": config.first_message,
            "language": primary_lang,
        }
        if is_hinglish:
            agent_dict["hinglish_mode"] = True

        return {
            "name": config.agent_name,
            "conversation_config": {
                "agent": agent_dict,
                "tts": {
                    "voice_id": voice_id,
                    "model_id": tts_model_id,
                },
                "conversation": {
                    "max_duration_seconds": 600,
                },
            },
            "platform_settings": {
                "auth": {
                    "enable_auth": True,
                },
            },
        }

    def create_agent(
        self,
        config: GeneratedAgentConfig,
        explicit_voice_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Creates an agent in ElevenLabs using the official REST API and returns agent_id and resolved voice_id."""
        if not self.api_key or not self.api_key.strip():
            logger.error("ElevenLabs API key is not configured.")
            raise MissingElevenLabsApiKeyError("ElevenLabs API key is not configured on the server.")

        resolved_voice = self.resolve_voice_id(config, explicit_voice_id)
        payload = self.build_create_payload(config, resolved_voice)

        headers = {
            "xi-api-key": self.api_key.strip(),
            "Content-Type": "application/json",
        }

        try:
            if self._http_client:
                response = self._http_client.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=20.0,
                )
            else:
                with httpx.Client(timeout=20.0) as client:
                    response = client.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                    )
        except Exception as e:
            logger.error("ElevenLabs network connection failed: %s", type(e).__name__)
            raise ElevenLabsProviderError("Failed to communicate with ElevenLabs voice provider.") from e

        if response.status_code >= 400:
            error_detail = self._extract_error_summary(response)
            if error_detail:
                logger.error(
                    "ElevenLabs returned HTTP error status %s: %s",
                    response.status_code,
                    error_detail,
                )
            else:
                logger.error("ElevenLabs returned HTTP error status %s", response.status_code)

            raise ElevenLabsProviderError(
                f"ElevenLabs agent creation failed with status {response.status_code}.",
                provider_detail=error_detail or None,
            )

        try:
            body = response.json()
        except Exception as e:
            logger.error("ElevenLabs returned non-JSON response.")
            raise InvalidProviderResponseError("ElevenLabs returned an invalid response body.") from e

        agent_id = body.get("agent_id")
        if not agent_id or not isinstance(agent_id, str) or not agent_id.strip():
            logger.error("ElevenLabs response did not contain a valid agent_id.")
            raise InvalidProviderResponseError("ElevenLabs did not return a valid agent_id.")

        return {
            "agent_id": agent_id.strip(),
            "voice_id": resolved_voice,
        }

    def get_signed_url(self, agent_id: str) -> str:
        """Requests a temporary signed WebSocket URL from ElevenLabs for live browser voice sessions."""
        if not self.api_key or not self.api_key.strip():
            logger.error("ElevenLabs API key is not configured.")
            raise MissingElevenLabsApiKeyError("ElevenLabs API key is not configured on the server.")

        clean_agent_id = agent_id.strip() if agent_id else ""
        if not clean_agent_id:
            raise ValueError("agent_id must not be empty.")

        headers = {
            "xi-api-key": self.api_key.strip(),
        }
        params = {
            "agent_id": clean_agent_id,
        }

        try:
            if self._http_client:
                response = self._http_client.get(
                    self.signed_url_endpoint,
                    params=params,
                    headers=headers,
                    timeout=20.0,
                )
            else:
                with httpx.Client(timeout=20.0) as client:
                    response = client.get(
                        self.signed_url_endpoint,
                        params=params,
                        headers=headers,
                    )
        except Exception as e:
            logger.error("ElevenLabs signed URL request failed: %s", type(e).__name__)
            raise ElevenLabsProviderError("Failed to communicate with ElevenLabs voice provider.") from e

        if response.status_code >= 400:
            error_detail = self._extract_error_summary(response)
            if error_detail:
                logger.error(
                    "ElevenLabs get-signed-url returned HTTP error status %s: %s",
                    response.status_code,
                    error_detail,
                )
            else:
                logger.error("ElevenLabs get-signed-url returned HTTP error status %s", response.status_code)

            raise ElevenLabsProviderError(
                f"ElevenLabs signed URL generation failed with status {response.status_code}.",
                provider_detail=error_detail or None,
            )

        try:
            body = response.json()
        except Exception as e:
            logger.error("ElevenLabs returned non-JSON response.")
            raise InvalidProviderResponseError("ElevenLabs returned an invalid response body.") from e

        signed_url = body.get("signed_url")
        if not signed_url or not isinstance(signed_url, str) or not signed_url.strip():
            logger.error("ElevenLabs response did not contain a valid signed_url.")
            raise InvalidProviderResponseError("ElevenLabs did not return a valid signed_url.")

        return signed_url.strip()


def get_elevenlabs_service() -> ElevenLabsAgentService:
    settings = get_settings()
    return ElevenLabsAgentService(
        api_key=settings.elevenlabs_api_key,
        default_voice_id=settings.elevenlabs_default_voice_id,
    )