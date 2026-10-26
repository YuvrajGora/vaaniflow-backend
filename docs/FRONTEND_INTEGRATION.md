# VaaniFlow AI ? Frontend Integration Guide

**Target Audience:** Frontend Developer (React + TypeScript + Vite)  
**Workspace Root:** `C:\Users\a\Documents\Hackathon\h2\backend h2`  

---

## 1. Quick Reference & URLs

| Resource | URL | Description |
| :--- | :--- | :--- |
| **Local Backend Base URL** | `http://127.0.0.1:8000` | Local development server |
| **Interactive Swagger Docs** | `http://127.0.0.1:8000/docs` | Test live endpoints & inspect schemas |
| **OpenAPI Schema** | `http://127.0.0.1:8000/openapi.json` | JSON format OpenAPI v3 specification |
| **Full API Contract** | `docs/API_CONTRACT.md` | Complete specifications for all endpoints |

---

## 2. Environment Configuration (Vite)

In your Vite frontend project root, configure the backend URL in `.env`:

```bash
# Frontend .env (Vite)
VITE_BACKEND_URL=http://127.0.0.1:8000
```

> **Note:** Vite exposes client environment variables via `import.meta.env.VITE_*`. Do not use `process.env.REACT_APP_*`.

---

## 3. End-to-End User & API Workflow

```
[ User Prompt ]
      ?
      ?
1. POST /api/agents/configure  ???????  Review Draft Configuration
                                                ?
                                                ?
                                        2. POST /api/agents  ???????  Agent Created & Saved
                                                                              ?
                                                                              ?
                                                                3. POST /api/agents/{agent_id}/session
                                                                              ?
                                                                              ?
                                                                Signed WebSocket URL (15m expiry)
                                                                              ?
                                                                              ?
                                                                @elevenlabs/react Voice Call
```

---

### 3.1 Step 1: Generate Configuration (`POST /api/agents/configure`)

```typescript
import { AgentConfigureRequest, AgentConfigureResponse } from "../types/agentConfig";

const BACKEND_BASE_URL: string =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) || "http://127.0.0.1:8000";

export async function generateAgentConfig(
  request: AgentConfigureRequest
): Promise<AgentConfigureResponse> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/agents/configure`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Configuration failed: HTTP ${response.status}`);
  }

  return response.json();
}
```

---

### 3.2 Step 2: Create Agent (`POST /api/agents`)

```typescript
import { AgentCreateRequest, AgentCreateResponse } from "../types/agentCreate";

export async function createAgent(
  request: AgentCreateRequest
): Promise<AgentCreateResponse> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/agents`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Agent creation failed: HTTP ${response.status}`);
  }

  return response.json();
}
```

---

### 3.3 Step 3: Authorize Voice Session (`POST /api/agents/{agent_id}/session`)

Requests an authorized signed WebSocket URL to start a live voice call in the browser.

#### TypeScript Types (`types/agentSession.ts`)
```typescript
export interface AgentSessionResponse {
  status: "ok";
  agent_id: string;
  signed_url: string;        // Temporary signed WebSocket URL
  expires_in_seconds: number; // 900 seconds (15 minutes)
}
```

#### Fetch Function (`services/sessionService.ts`)
```typescript
export async function requestAgentSession(
  agentId: string
): Promise<AgentSessionResponse> {
  const response = await fetch(
    `${BACKEND_BASE_URL}/api/agents/${encodeURIComponent(agentId)}/session`,
    {
      method: "POST",
      headers: {
        "Accept": "application/json",
      },
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Session authorization failed: HTTP ${response.status}`);
  }

  return response.json();
}
```

#### Connecting via the Official ElevenLabs React SDK (`@elevenlabs/react`)
```tsx
import React, { useState } from "react";
import { useConversation } from "@elevenlabs/react";
import { requestAgentSession } from "../services/sessionService";

export const VoiceSessionButton: React.FC<{ agentId: string }> = ({ agentId }) => {
  const conversation = useConversation();
  const [connecting, setConnecting] = useState(false);

  const startVoiceCall = async () => {
    setConnecting(true);
    try {
      // 1. Request temporary signed URL from your backend (no API key exposed)
      const sessionData = await requestAgentSession(agentId);

      // 2. Request microphone permission
      await navigator.mediaDevices.getUserMedia({ audio: true });

      // 3. Connect browser directly to ElevenLabs WebSocket using the signed URL
      await conversation.startSession({
        signedUrl: sessionData.signed_url,
      });
    } catch (err: any) {
      alert("Failed to start voice call: " + err.message);
    } finally {
      setConnecting(false);
    }
  };

  const endVoiceCall = async () => {
    await conversation.endSession();
  };

  return (
    <div>
      {conversation.status === "connected" ? (
        <button onClick={endVoiceCall} style={{ background: "red", color: "white" }}>
          End Call
        </button>
      ) : (
        <button onClick={startVoiceCall} disabled={connecting}>
          {connecting ? "Connecting..." : "Start Voice Call"}
        </button>
      )}
      <p>Status: {conversation.status}</p>
    </div>
  );
};
```

---

### 3.4 Step 4: List Saved Agents (`GET /api/agents`)

Fetches all configured receptionist agents persisted in the local SQLite database.

#### TypeScript Types (`types/agentList.ts`)
```typescript
export interface AgentSummary {
  id: number;
  agent_id: string;
  name: string;
  business_type: string;
  voice_id: string;
  created_at: string;
}

export interface AgentListResponse {
  status: "ok";
  agents: AgentSummary[];
  total: number;
}
```

#### Fetch Function (`services/agentService.ts`)
```typescript
export async function listAgents(): Promise<AgentListResponse> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/agents`, {
    method: "GET",
    headers: {
      "Accept": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch agents: HTTP ${response.status}`);
  }

  return response.json();
}
```

---

## 4. Pending Subsequent Milestones (Do Not Call Yet)

* `GET /api/sessions` & `GET /api/sessions/{session_id}` (Call logs & transcripts)


---

## 5. Security Directives for Frontend Developers
1. **Never request or expose `ELEVENLABS_API_KEY` on the client.** The frontend only communicates with your FastAPI backend.
2. The `signed_url` received from `POST /api/agents/{agent_id}/session` is ephemeral and must only be passed into `@elevenlabs/react` `startSession({ signedUrl })`. Do not log or inspect signed tokens in frontend analytics.
3. The frontend must never attempt to call ElevenLabs management APIs directly.
