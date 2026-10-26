# VaaniFlow AI ? Backend API Contract

**Status:** Milestone 6 (Signed Voice Session Authorization Implemented)  
**Workspace:** `C:\Users\a\Documents\Hackathon\h2\backend h2`  
**Base URL (Local Development):** `http://127.0.0.1:8000`  
**Interactive Docs (Swagger UI):** `http://127.0.0.1:8000/docs`  

---

## 1. Endpoint Status Matrix

| Method | Path | Status | Tested | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **GET** | `/api/health` | **Implemented** | Yes | Service health check returning operational status. |
| **GET** | `/` | **Implemented** | Yes | Root service discovery and identity check. |
| **POST** | `/api/agents/configure` | **Implemented** | Yes | Generates structured voice agent configuration using Google Gemini structured outputs. |
| **POST** | `/api/agents` | **Implemented** | Yes | Provisions conversational voice agent via ElevenLabs REST API and persists record to SQLite. |
| **POST** | `/api/agents/{agent_id}/session` | **Implemented** | Yes | Generates authorized, temporary signed WebSocket URL for browser client voice sessions. |
| **GET** | `/api/agents` | **PLANNED (NOT IMPLEMENTED)** | No | Lists configured receptionist agents for the business. |
| **GET** | `/api/sessions` | **PLANNED (NOT IMPLEMENTED)** | No | Lists past voice interaction sessions and summaries. |
| **GET** | `/api/sessions/{session_id}` | **PLANNED (NOT IMPLEMENTED)** | No | Retrieves full call record, transcript, and appointment requests. |

> **IMPORTANT:** Endpoints marked **PLANNED (NOT IMPLEMENTED)** are design specifications for subsequent milestones. Client applications must not call them until implementation milestones are delivered.

---

## 2. Implemented Endpoints

### 2.1 `GET /api/health`
Checks backend readiness and container health.

* **HTTP Method:** `GET`
* **Path:** `/api/health`
* **Status Code:** `200 OK`
* **Response JSON:**
  ```json
  {
    "status": "ok",
    "service": "vaaniflow-backend"
  }
  ```

---

### 2.2 `GET /`
Provides top-level service metadata.

* **HTTP Method:** `GET`
* **Path:** `/`
* **Status Code:** `200 OK`
* **Response JSON:**
  ```json
  {
    "service": "vaaniflow-backend",
    "version": "0.1.0",
    "status": "running"
  }
  ```

---

### 2.3 `POST /api/agents/configure`
Interprets a natural-language description and optional structured preferences into a validated intermediate voice agent configuration using Google Gemini (`gemini-2.5-flash`).

* **HTTP Method:** `POST`
* **Path:** `/api/agents/configure`
* **Request Schema (`AgentConfigureRequest`):**
  ```json
  {
    "prompt": "Create a friendly bilingual receptionist for a dental clinic who answers common questions and helps patients request appointments.",
    "preferences": {
      "language": "hi",
      "additional_languages": ["en"],
      "tone": "warm and professional",
      "business_type": "Dental Clinic",
      "voice_gender": "female",
      "voice_accent": "Indian English",
      "voice_id": null,
      "capabilities": ["answer_faqs", "appointment_requests"]
    }
  }
  ```
* **Status Code:** `200 OK`
* **Response Schema (`AgentConfigureResponse`):**
  ```json
  {
    "status": "ok",
    "configuration": {
      "agent_name": "Aanya",
      "business_type": "Dental Clinic",
      "primary_language": "hi",
      "additional_languages": ["en"],
      "first_message": "Namaste! Sharma Dental Clinic mein aapka swagat hai. Main Aanya hoon. Main aapki kya sahayata kar sakti hoon?",
      "system_prompt": "You are Aanya, a courteous receptionist for Sharma Dental Clinic...",
      "voice_preference": {
        "gender": "female",
        "accent": "Indian English / Hindi",
        "description": "Warm, respectful, and articulate receptionist tone",
        "voice_id": null
      },
      "intended_capabilities": [
        "answer_faqs",
        "appointment_requests"
      ],
      "faqs": [],
      "appointment_handling": {
        "enabled": true,
        "instructions": "Collect requested date, time, and service. Inform caller staff will confirm via call/SMS."
      }
    },
    "elevenlabs_config_preview": {
      "agent": {
        "prompt": {
          "prompt": "You are Aanya, a courteous receptionist for Sharma Dental Clinic..."
        },
        "first_message": "Namaste! Sharma Dental Clinic mein aapka swagat hai. Main Aanya hoon. Main aapki kya sahayata kar sakti hoon?",
        "language": "hi",
        "hinglish_mode": true
      },
      "tts": {
        "voice_id": null,
        "model_id": "eleven_turbo_v2_5"
      },
      "conversation": {
        "max_duration_seconds": 600
      }
    }
  }
  ```

---

### 2.4 `POST /api/agents`
Provisions an actual conversational voice agent with ElevenLabs via `POST https://api.elevenlabs.io/v1/convai/agents/create` and persists the created agent entity into a local SQLite database.

* **HTTP Method:** `POST`
* **Path:** `/api/agents`
* **Request Schema (`AgentCreateRequest`):**
  ```json
  {
    "configuration": {
      "agent_name": "Aanya",
      "business_type": "Dental Clinic",
      "primary_language": "hi",
      "additional_languages": ["en"],
      "first_message": "Namaste! Sharma Dental Clinic mein aapka swagat hai.",
      "system_prompt": "You are Aanya, a courteous receptionist for Sharma Dental Clinic...",
      "voice_preference": {
        "gender": "female",
        "accent": "Indian English / Hindi",
        "description": "Warm and professional tone",
        "voice_id": null
      },
      "intended_capabilities": ["answer_faqs", "appointment_requests"],
      "faqs": [],
      "appointment_handling": {
        "enabled": true,
        "instructions": "Collect requested date, time, and service. Inform caller staff will confirm via call/SMS."
      }
    },
    "voice_id": "21m00Tcm4TlvDq8ikWAM"
  }
  ```
* **Status Code:** `201 Created`
* **Response Schema (`AgentCreateResponse`):**
  ```json
  {
    "status": "created",
    "agent_id": "agent_7101k5zvyjhmfg983brhmhkd98n6",
    "name": "Aanya",
    "business_type": "Dental Clinic",
    "voice_id": "21m00Tcm4TlvDq8ikWAM",
    "created_at": "2026-10-25T23:45:00.000000+00:00"
  }
  ```
* **Voice ID Resolution Order:**
  1. Explicitly supplied `voice_id` in the creation request.
  2. `configuration.voice_preference.voice_id` in the generated configuration.
  3. `ELEVENLABS_DEFAULT_VOICE_ID` from backend environment settings.
  4. If none configured &rarr; Returns `422 Unprocessable Content`.
* **Platform Security Settings:**
  * Outgoing agent creation payload sets `"platform_settings": {"auth": {"enable_auth": true}}`, ensuring the created agent requires signed token authorization for live sessions and works seamlessly with the signed URL flow.

---

### 2.5 `POST /api/agents/{agent_id}/session`
Requests an authorized temporary signed WebSocket URL from ElevenLabs (`GET https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?agent_id={agent_id}`) for live client browser conversations.

* **HTTP Method:** `POST`
* **Path:** `/api/agents/{agent_id}/session`
* **Path Parameter:**
  * `agent_id` (string, required): The ElevenLabs agent identifier.
* **Request Body:** None required.
* **Status Code:** `200 OK`
* **Response Content-Type:** `application/json`
* **Response Schema (`AgentSessionResponse`):**
  ```json
  {
    "status": "ok",
    "agent_id": "agent_7101k5zvyjhmfg983brhmhkd98n6",
    "signed_url": "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=agent_7101k5zvyjhmfg983brhmhkd98n6&conversation_signature=example_signed_token_456",
    "expires_in_seconds": 900
  }
  ```
* **Operational & Security Behavior:**
  * **Strict Registry Check:** The endpoint verifies that `agent_id` exists in the backend's local SQLite database before requesting authorization from ElevenLabs. Unknown agent IDs return `404 Not Found` immediately without issuing upstream requests.
  * **Zero Credential Leakage:** `ELEVENLABS_API_KEY` is kept strictly on the backend. It is never returned in client responses or printed in logs.
  * **No Secret/URL Persistence:** Signed URLs and temporary tokens are ephemeral and are never stored in SQLite or application log files.
  * **Validity Window:** Signed URLs expire after **15 minutes (900 seconds)** for connection initialization. Once established, active conversations continue uninterrupted.
* **Error Responses:**
  * `404 Not Found`: Agent ID not found in local registry (`{"detail": "Agent not found in local registry."}`).
  * `500 Internal Server Error`: Server missing `ELEVENLABS_API_KEY` configuration.
  * `502 Bad Gateway`: Upstream ElevenLabs API failure, network timeout, or invalid provider response.

---

### 2.6 `GET /api/agents`
Lists all configured voice agents stored in the local SQLite database in descending order of creation.

* **HTTP Method:** `GET`
* **Path:** `/api/agents`
* **Status Code:** `200 OK`
* **Response Content-Type:** `application/json`
* **Response Schema (`AgentListResponse`):**
  ```json
  {
    "status": "ok",
    "agents": [
      {
        "id": 1,
        "agent_id": "agent_0401m5td8ncmf1nbhw85af8c04a2",
        "name": "Dr. Smile Receptionist",
        "business_type": "Dental Clinic",
        "voice_id": "SZfY4K69FwXus87eayHK",
        "created_at": "2026-10-25T20:50:36.710735+00:00"
      }
    ],
    "total": 1
  }
  ```
* **Error Responses:**
  * `500 Internal Server Error`: Database query failure (`{"detail": "Failed to retrieve agents from database."}`).

---

## 3. Proposed Endpoints (Design Specifications)

### 3.1 `GET /api/sessions` *(PLANNED, NOT IMPLEMENTED)*
* **Purpose:** Lists past interaction sessions, caller contact details, and appointment request flags.
* **Status:** PLANNED.

### 3.2 `GET /api/sessions/{session_id}` *(PLANNED, NOT IMPLEMENTED)*
* **Purpose:** Fetches full call details, transcript (`available`, `unavailable`, `empty`), and appointment request info.
* **Status:** PLANNED.


---

## 4. Environment & Secrets Management
* `GEMINI_API_KEY`: Kept private on backend in `.env`.
* `GEMINI_MODEL`: Defaults to `gemini-2.5-flash`.
* `ELEVENLABS_API_KEY`: Kept private on backend in `.env`. Never exposed to client.
* `ELEVENLABS_DEFAULT_VOICE_ID`: Optional default voice ID fallback for agent creation.
* `DATABASE_PATH`: Defaults to `data/vaaniflow.db`.
