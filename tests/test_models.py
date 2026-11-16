from dataclasses import replace

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import Settings
from models import LM_STUDIO_DEFAULT_URL, as_plain_text_result, build_chat_model, split_model_id


@pytest.fixture
def settings():
    return replace(
        Settings.from_env(),
        chat_base_url=None,
        chat_api_key=None,
        chat_temperature=None,
        chat_context_window=None,
        chat_reasoning=None,
    )


def test_lmstudio_uses_openai_compatible_chat_completions(settings):
    model = build_chat_model(replace(settings, chat_model="lmstudio:qwen/qwen3-8b"))

    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == LM_STUDIO_DEFAULT_URL
    assert model.use_responses_api is False


def test_lmstudio_base_url_can_be_overridden(settings):
    custom_url = "http://gpu-box:1234/v1"
    model = build_chat_model(
        replace(settings, chat_model="lmstudio:qwen/qwen3-8b", chat_base_url=custom_url)
    )

    assert model.openai_api_base == custom_url


def test_ollama_receives_context_window_and_reasoning(settings):
    model = build_chat_model(
        replace(
            settings,
            chat_model="ollama:qwen3:8b",
            chat_context_window=16384,
            chat_reasoning=True,
        )
    )

    assert isinstance(model, ChatOllama)
    assert model.model == "qwen3:8b"
    assert model.num_ctx == 16384
    assert model.reasoning is True


@pytest.mark.parametrize("model_id", ["qwen3", ":qwen3", "ollama:"])
def test_model_id_without_provider_is_rejected(model_id):
    with pytest.raises(ValueError):
        split_model_id(model_id)


@pytest.fixture
def lmstudio_model(settings):
    return build_chat_model(replace(settings, chat_model="lmstudio:qwen/qwen3-8b"))


def test_lmstudio_keeps_reasoning_from_full_response(lmstudio_model):
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "51", "reasoning_content": "17*3"},
                "finish_reason": "stop",
            }
        ]
    }

    message = lmstudio_model._create_chat_result(response).generations[0].message

    assert message.additional_kwargs["reasoning_content"] == "17*3"


def test_lmstudio_keeps_reasoning_from_streamed_chunks(lmstudio_model):
    chunk = {"choices": [{"index": 0, "delta": {"reasoning_content": "17*3"}}]}

    generation = lmstudio_model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})

    assert generation.message.additional_kwargs["reasoning_content"] == "17*3"


def test_list_tool_content_becomes_plain_text():
    message = ToolMessage(
        content=[{"type": "text", "text": "passage"}], tool_call_id="1", name="search"
    )

    assert isinstance(as_plain_text_result(message).content, str)
