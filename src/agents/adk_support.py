"""Shared ADK plumbing: model selection, a synchronous runner, tool-event capture.

The ADK server is not used; `api.py` stays the HTTP surface. These helpers run an
ADK agent once, synchronously, and hand back every structured output it produced.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from src.utils.env_helper import load_env_file

load_env_file()

APP_NAME = "lumen-premortem"

# ADK reads GOOGLE_API_KEY; the repo's .env has always carried GEMINI_API_KEY.
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def agent_model() -> str:
    """Gemini model for both subagents.

    Defaults to gemini-3.7-flash, overridable via LUMEN_AGENT_MODEL.
    """
    return os.environ.get("LUMEN_AGENT_MODEL", "gemini-3.7-flash")


def is_vertex_ai() -> bool:
    """Return True if Vertex AI backend is configured."""
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1") or \
           os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in ("true", "1")


def has_gemini_key() -> bool:
    """Return True if Gemini credentials are present (either via Vertex AI or API key)."""
    if is_vertex_ai():
        return bool(
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def create_adk_model(model_name: Optional[str] = None):
    """Instantiate an ADK Gemini model configured with retry options."""
    from google.adk.models import Gemini
    from google.genai import types

    name = model_name or agent_model()
    retry_options = types.HttpRetryOptions(initial_delay=2.0, attempts=3)
    return Gemini(model=name, retry_options=retry_options)


class ToolEventLog:
    """Collects per-tool-call events via ADK's before/after tool callbacks and optionally emits to live stream."""

    def __init__(self, on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> None:
        self.events: List[Dict[str, Any]] = []
        self.on_event = on_event

    def before(self, tool, args, tool_context):  # ADK callback signature
        event = {"phase": "tool_start", "tool": tool.name, "args": dict(args)}
        self.events.append(event)
        if self.on_event:
            try:
                self.on_event("tool_start", event)
            except Exception:
                pass
        return None

    def after(self, tool, args, tool_context, tool_response):
        event = {
            "phase": "tool_end",
            "tool": tool.name,
            "args": dict(args),
            "result_count": len((tool_response or {}).get("results", []))
            if isinstance(tool_response, dict) else None,
        }
        self.events.append(event)
        if self.on_event:
            try:
                self.on_event("tool_end", event)
            except Exception:
                pass
        return None


def run_agent(agent, prompt: str, output_key: str) -> List[Any]:
    """Run an ADK agent to completion and return every value it wrote to `output_key`.

    A LoopAgent overwrites session state on each iteration, so the outputs are
    read off the event stream instead of out of the final session.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    max_retries = max(0, int(os.environ.get("LUMEN_LLM_MAX_RETRIES", "3")))
    base_delay = max(0.1, float(os.environ.get("LUMEN_LLM_RETRY_BASE_SEC", "1.0")))

    def retryable(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(token in text for token in (
            "429", "resource exhausted", "rate limit", "503", "service unavailable",
            "temporarily unavailable", "deadline exceeded", "unavailable",
        ))

    for attempt in range(max_retries + 1):
        try:
            runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
            session_id = str(uuid.uuid4())
            asyncio.run(
                runner.session_service.create_session(
                    app_name=APP_NAME, user_id="lumen", session_id=session_id
                )
            )
            outputs: List[Any] = []
            message = types.Content(role="user", parts=[types.Part(text=prompt)])
            for event in runner.run(user_id="lumen", session_id=session_id, new_message=message):
                delta = getattr(event.actions, "state_delta", None) if event.actions else None
                if delta and output_key in delta:
                    outputs.append(delta[output_key])
            return outputs
        except Exception as exc:
            if attempt >= max_retries or not retryable(exc):
                raise
            # Exponential backoff with jitter prevents an agent burst from
            # synchronizing its retries and amplifying Vertex AI 429s.
            delay = min(20.0, base_delay * (2 ** attempt)) * random.uniform(0.5, 1.5)
            time.sleep(delay)

    raise RuntimeError("agent execution exhausted retry policy")


def stop_after_output(state_key: str) -> Callable:
    """after_agent_callback that ends a LoopAgent once the agent has answered.

    ADK only honours `escalate` when it rides on a yielded event, and the
    callback yields one only if it returns content or leaves a state delta — so
    this writes a marker as well as setting the flag.
    """

    def _callback(callback_context):
        callback_context.state[state_key] = True
        callback_context.actions.escalate = True
        return None

    return _callback
