"""Pydantic models matching OpenAI v1 chat completions (minimal subset)."""
from __future__ import annotations

import time
import uuid
from typing import Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    user: str | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def build_suppressed_response(
    *, request_id: str, model: str, refusal: str, prompt_tokens: int,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=_completion_id(),
        created=int(time.time()),
        model=model,
        choices=[Choice(
            index=0,
            message=ChatMessage(role="assistant", content=refusal),
            finish_reason="content_filter",
        )],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens),
    )


def build_pass_response(
    *, request_id: str, model: str, completion: str,
    prompt_tokens: int, completion_tokens: int,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=_completion_id(),
        created=int(time.time()),
        model=model,
        choices=[Choice(
            index=0,
            message=ChatMessage(role="assistant", content=completion),
            finish_reason="stop",
        )],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
