"""Chat model construction, including local open-weight model servers (Ollama, LM Studio)."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from config import Settings

LM_STUDIO = "lmstudio"
OLLAMA = "ollama"
OPENAI = "openai"
LM_STUDIO_DEFAULT_URL = "http://localhost:1234/v1"
LOCAL_SERVER_PLACEHOLDER_KEY = "not-needed"
REASONING_FIELD = "reasoning_content"

logger = logging.getLogger("models")


def split_model_id(model_id: str) -> tuple[str, str]:
    provider, separator, model_name = model_id.partition(":")
    if not separator or not provider.strip() or not model_name.strip():
        raise ValueError(f"CHAT_MODEL must look like 'provider:model', got {model_id!r}")
    return provider.strip().lower(), model_name.strip()


def build_chat_model(settings: Settings) -> BaseChatModel:
    provider, model_name = split_model_id(settings.chat_model)
    _warn_about_ignored_options(settings, provider)
    options: dict[str, Any] = {}
    if settings.chat_temperature is not None:
        options["temperature"] = settings.chat_temperature

    if provider == LM_STUDIO:
        options |= _openai_compatible_options(settings, LM_STUDIO_DEFAULT_URL)
        return ChatOpenAICompatible(model=model_name, **options)
    if provider == OPENAI and settings.chat_base_url:
        options |= _openai_compatible_options(settings, settings.chat_base_url)
        return ChatOpenAICompatible(model=model_name, **options)
    if provider == OLLAMA:
        options |= _ollama_options(settings)
    return init_chat_model(f"{provider}:{model_name}", **options)


def _openai_compatible_options(settings: Settings, default_url: str) -> dict[str, Any]:
    return {
        "base_url": settings.chat_base_url or default_url,
        "api_key": settings.chat_api_key or LOCAL_SERVER_PLACEHOLDER_KEY,
        # Local servers implement Chat Completions; the Responses API is not always there.
        "use_responses_api": False,
        "stream_usage": True,
    }


def _field(container: Any, name: str) -> Any:
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _attach_reasoning(message: BaseMessage, reasoning: Any) -> None:
    if isinstance(reasoning, str) and reasoning:
        message.additional_kwargs[REASONING_FIELD] = reasoning


class ChatOpenAICompatible(ChatOpenAI):
    """ChatOpenAI that keeps the `reasoning_content` field added by LM Studio, vLLM and similar
    servers, in `additional_kwargs` where ChatOllama and ChatDeepSeek put it too."""

    def _create_chat_result(
        self, response: Any, generation_info: dict | None = None
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        for generation, choice in zip(result.generations, _field(response, "choices") or []):
            _attach_reasoning(
                generation.message, _field(_field(choice, "message"), REASONING_FIELD)
            )
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices") or []
        if generation_chunk is not None and choices:
            _attach_reasoning(
                generation_chunk.message, _field(choices[0].get("delta"), REASONING_FIELD)
            )
        return generation_chunk


def _ollama_options(settings: Settings) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if settings.chat_base_url:
        options["base_url"] = settings.chat_base_url
    if settings.chat_context_window:
        options["num_ctx"] = settings.chat_context_window
    if settings.chat_reasoning is not None:
        options["reasoning"] = settings.chat_reasoning
    return options


def _warn_about_ignored_options(settings: Settings, provider: str) -> None:
    if provider == OLLAMA:
        return
    if settings.chat_context_window:
        logger.warning(
            "CHAT_CONTEXT_WINDOW only applies to Ollama; set the context length "
            "where the model is loaded (e.g. LM Studio's load settings)."
        )
    if settings.chat_reasoning is not None:
        logger.warning("CHAT_REASONING only applies to Ollama and is ignored.")


ToolResult = ToolMessage | Command[Any]


def as_plain_text_result(result: ToolResult) -> ToolResult:
    if isinstance(result, ToolMessage) and isinstance(result.content, list):
        return result.model_copy(update={"content": result.text})
    return result


@wrap_tool_call
async def plain_text_tool_results(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolResult]]
) -> ToolResult:
    """Send tool results to the model as plain text, which every local server accepts."""
    return as_plain_text_result(await handler(request))
