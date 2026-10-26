from datetime import datetime, timezone
import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.repositories.agent_repository import (
    AgentRepository,
    DatabasePersistenceError,
    get_agent_repository,
)
from app.schemas.agent_config import (
    AgentConfigureRequest,
    AgentConfigureResponse,
    AgentCreateRequest,
    AgentCreateResponse,
    AgentSessionResponse,
    AgentSummary,
    AgentListResponse,
)
from app.services.agent_config_generator import (
    GeminiAgentConfigGenerator,
    InvalidModelOutputError,
    MissingApiKeyError,
    ProviderError,
    get_config_generator,
)
from app.services.elevenlabs_agent_service import (
    ElevenLabsAgentService,
    ElevenLabsProviderError,
    InvalidProviderResponseError,
    MissingElevenLabsApiKeyError,
    MissingVoiceIdError,
    get_elevenlabs_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/configure",
    response_model=AgentConfigureResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate voice agent configuration from prompt using Gemini",
)
def configure_agent(
    request: AgentConfigureRequest,
    generator: GeminiAgentConfigGenerator = Depends(get_config_generator),
) -> AgentConfigureResponse:
    """Interprets a natural-language description and optional preferences into a validated

    intermediate voice agent configuration using Google Gemini structured outputs.
    """
    try:
        config = generator.generate_config(request)
    except MissingApiKeyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    except (ProviderError, InvalidModelOutputError) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e

    return AgentConfigureResponse(
        status="ok",
        configuration=config,
        elevenlabs_config_preview=config.to_elevenlabs_conversation_config_preview(),
    )


@router.get(
    "",
    response_model=AgentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all registered voice agents",
)
def list_agents(
    agent_repository: AgentRepository = Depends(get_agent_repository),
) -> AgentListResponse:
    """Retrieves all registered agents stored in the local SQLite database."""
    try:
        records = agent_repository.list_agents()
    except Exception as e:
        logger.error("Failed to query agents list: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agents from database.",
        ) from e

    agents = [
        AgentSummary(
            id=r["id"],
            agent_id=r["agent_id"],
            name=r["name"],
            business_type=r["business_type"],
            voice_id=r["voice_id"],
            created_at=r["created_at"],
        )
        for r in records
    ]
    return AgentListResponse(
        status="ok",
        agents=agents,
        total=len(agents),
    )


@router.post(
    "",
    response_model=AgentCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Provision a real voice agent with ElevenLabs and persist to SQLite",
)
def create_agent(
    request: AgentCreateRequest,
    elevenlabs_service: ElevenLabsAgentService = Depends(get_elevenlabs_service),
    agent_repository: AgentRepository = Depends(get_agent_repository),
) -> AgentCreateResponse:
    """Validates configuration, resolves voice ID, provisions the voice agent via ElevenLabs API,

    and stores the created agent record in local SQLite database.
    """
    try:
        result = elevenlabs_service.create_agent(
            config=request.configuration,
            explicit_voice_id=request.voice_id,
        )
    except MissingVoiceIdError as e:
        raise HTTPException(
            status_code=getattr(status, 'HTTP_422_UNPROCESSABLE_CONTENT', 422),
            detail=str(e),
        ) from e
    except MissingElevenLabsApiKeyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    except (ElevenLabsProviderError, InvalidProviderResponseError) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e

    created_at_utc = datetime.now(timezone.utc).isoformat()
    config_json = request.configuration.model_dump_json()

    try:
        agent_repository.save_agent(
            agent_id=result["agent_id"],
            name=request.configuration.agent_name,
            business_type=request.configuration.business_type,
            voice_id=result["voice_id"],
            configuration_json=config_json,
            created_at=created_at_utc,
        )
    except Exception as e:
        logger.critical(
            "PARTIAL FAILURE: ElevenLabs agent %s was provisioned successfully, but local SQLite persistence failed: %s",
            result["agent_id"],
            type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent was provisioned with provider but failed to persist locally.",
        ) from e

    return AgentCreateResponse(
        status="created",
        agent_id=result["agent_id"],
        name=request.configuration.agent_name,
        business_type=request.configuration.business_type,
        voice_id=result["voice_id"],
        created_at=created_at_utc,
    )


@router.post(
    "/{agent_id}/session",
    response_model=AgentSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Authorize a live voice session and retrieve a signed WebSocket URL",
)
def create_agent_session(
    agent_id: str,
    agent_repository: AgentRepository = Depends(get_agent_repository),
    elevenlabs_service: ElevenLabsAgentService = Depends(get_elevenlabs_service),
) -> AgentSessionResponse:
    """Verifies the agent exists in the local registry, requests an authorized signed

    WebSocket URL from ElevenLabs, and returns it to the client without exposing API keys.
    """
    clean_agent_id = agent_id.strip() if agent_id else ""
    if not clean_agent_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found in local registry.",
        )

    # 1. Restrict session authorization strictly to locally stored agents
    stored_agent = agent_repository.get_agent_by_id(clean_agent_id)
    if not stored_agent:
        logger.warning("Session authorization denied for unknown local agent_id: %s", clean_agent_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found in local registry.",
        )

    # 2. Request signed WebSocket URL from ElevenLabs
    try:
        signed_url = elevenlabs_service.get_signed_url(clean_agent_id)
    except MissingElevenLabsApiKeyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    except (ElevenLabsProviderError, InvalidProviderResponseError) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e

    # 3. Return signed URL and validity window to frontend
    return AgentSessionResponse(
        status="ok",
        agent_id=clean_agent_id,
        signed_url=signed_url,
        expires_in_seconds=900,
    )
