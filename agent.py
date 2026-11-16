"""Agentic RAG: a LangChain agent that answers from the knowledge base and acts through MCP tools."""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, ToolCallInterceptor
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from mcp.types import CallToolResult, TextContent
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from config import PROJECT_DIR, Settings, configure_logging
from knowledge_base import knowledge_base_exists
from models import REASONING_FIELD, build_chat_model, plain_text_tool_results

SYSTEM_PROMPT = """You are a customer support assistant for Acme Co.

- For any question about policies, shipping or products, call search_knowledge_base first.
- Answer only from the passages it returns and cite them inline, e.g. [refund_policy.md#0].
- If nothing relevant comes back, say the knowledge base doesn't cover it. Never guess.
- Before opening a ticket, confirm from the policy that the request qualifies.
- Keep answers short and concrete."""

MAX_AGENT_STEPS = 25
TOOL_PREVIEW_CHARS = 160

logger = logging.getLogger("agent")


def stdio_server(script_name: str) -> dict[str, Any]:
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(PROJECT_DIR / script_name)],
    }


def build_server_connections(settings: Settings) -> dict[str, dict[str, Any]]:
    knowledge_base = (
        {"transport": "streamable_http", "url": settings.rag_server_url}
        if settings.rag_server_url
        else stdio_server("rag_mcp_server.py")
    )
    return {"knowledge_base": knowledge_base, "tickets": stdio_server("ticket_mcp_server.py")}


async def load_tools_with_live_sessions(
    client: MultiServerMCPClient,
    stack: AsyncExitStack,
    tool_interceptors: list[ToolCallInterceptor],
) -> list[BaseTool]:
    """Keep one session per server open, so each server starts once instead of per tool call."""
    tools: list[BaseTool] = []
    for server_name in client.connections:
        session = await stack.enter_async_context(client.session(server_name))
        server_tools = await load_mcp_tools(
            session, server_name=server_name, tool_interceptors=tool_interceptors
        )
        logger.info("%s tools: %s", server_name, [tool.name for tool in server_tools])
        tools.extend(server_tools)
    return tools


def reasoning_text(message: AIMessage) -> str:
    """The model's reasoning, whether the provider puts it in a field or in content blocks."""
    if reasoning := message.additional_kwargs.get(REASONING_FIELD):
        return reasoning
    return "".join(
        block.get("reasoning", "")
        for block in message.content_blocks
        if block["type"] == "reasoning"
    )


def one_line_preview(text: str) -> str:
    return " ".join(text.split())[:TOOL_PREVIEW_CHARS]


def describe_model_call(message: AIMessage) -> str:
    usage = message.usage_metadata or {}
    reasoning_tokens = (usage.get("output_token_details") or {}).get("reasoning")
    finish_reason = message.response_metadata.get("finish_reason", "?")
    return (
        f"  [model] tokens in={usage.get('input_tokens', '?')} "
        f"out={usage.get('output_tokens', '?')} reasoning={reasoning_tokens or 0} "
        f"finish_reason={finish_reason}"
    )


def describe_mcp_result(result: Any) -> str:
    if not isinstance(result, CallToolResult):
        return type(result).__name__
    block_types = ", ".join(block.type for block in result.content) or "no content"
    text_chars = sum(len(block.text) for block in result.content if isinstance(block, TextContent))
    summary = f"{'error' if result.isError else 'ok'}: {block_types}, {text_chars} chars"
    return summary + (", structured content" if result.structuredContent else "")


class ConsoleView:
    """Prints the agent's progress. Debug mode also streams everything the model generates
    and shows token usage, each MCP round trip and full tool results."""

    THINKING_HEADER = "  [thinking] "
    ANSWER_HEADER = "\nAssistant: "

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self._live_header: str | None = None
        self._answer_streamed = False

    @property
    def stream_modes(self) -> list[str]:
        return ["updates", "messages"] if self.debug else ["updates"]

    @property
    def tool_interceptors(self) -> list[ToolCallInterceptor]:
        return [self.trace_mcp_call] if self.debug else []

    def show_token(self, chunk: BaseMessage) -> None:
        if not isinstance(chunk, AIMessage):
            return
        self._stream(self.THINKING_HEADER, reasoning_text(chunk))
        if self._stream(self.ANSWER_HEADER, chunk.text):
            self._answer_streamed = True

    def show_update(self, update: dict[str, Any]) -> None:
        self.end_live_output()
        for node_output in update.values():
            if isinstance(node_output, dict):
                self._show_messages(node_output.get("messages", []))
        self._answer_streamed = False

    def end_live_output(self) -> None:
        if self._live_header is not None:
            print()
            self._live_header = None

    async def trace_mcp_call(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Print each MCP request before it is sent and its outcome with the round-trip time."""
        label = f"  [mcp] {request.server_name}.{request.name}"
        print(f"{label} request: {json.dumps(request.args, ensure_ascii=False, default=str)}")
        started = time.perf_counter()
        try:
            result = await handler(request)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"{label} failed after {elapsed:.2f} s: {type(exc).__name__}: {exc}")
            raise
        elapsed = time.perf_counter() - started
        print(f"{label} response after {elapsed:.2f} s ({describe_mcp_result(result)})")
        return result

    def _stream(self, header: str, text: str) -> bool:
        if not text:
            return False
        if self._live_header != header:
            self.end_live_output()
            print(header, end="")
            self._live_header = header
        print(text, end="", flush=True)
        return True

    def _show_messages(self, messages: Iterable[BaseMessage]) -> None:
        for message in messages:
            if isinstance(message, AIMessage):
                self._show_ai_message(message)
            elif isinstance(message, ToolMessage):
                result = message.text if self.debug else one_line_preview(message.text)
                print(f"  <- {message.name} [{message.status}]: {result}")

    def _show_ai_message(self, message: AIMessage) -> None:
        if self.debug:
            print(describe_model_call(message))
        for call in message.tool_calls:
            print(f"  -> {call['name']}({call['args']})")
        for call in message.invalid_tool_calls:
            error = f": {call['error']}" if call.get("error") else ""
            print(f"  !! malformed tool call {call.get('name')}({call.get('args')}){error}")
        if message.tool_calls or message.invalid_tool_calls or self._answer_streamed:
            return
        if message.text:
            print(f"\nAssistant: {message.text}\n")
        else:
            self._show_empty_response()

    def _show_empty_response(self) -> None:
        hint = "" if self.debug else " Run with --debug to see its reasoning."
        print(f"  !! The model returned neither an answer nor a tool call.{hint}")


async def ask(agent: CompiledStateGraph, question: str, thread_id: str, view: ConsoleView) -> None:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_AGENT_STEPS}
    inputs = {"messages": [{"role": "user", "content": question}]}
    try:
        async for mode, payload in agent.astream(
            inputs, config=config, stream_mode=view.stream_modes
        ):
            if mode == "messages":
                view.show_token(payload[0])
            else:
                view.show_update(payload)
    finally:
        view.end_live_output()


async def ask_safely(
    agent: CompiledStateGraph, question: str, thread_id: str, view: ConsoleView
) -> bool:
    """Answer one question, reporting failures (e.g. model server not running) without crashing."""
    try:
        await ask(agent, question, thread_id, view)
        return True
    except GraphRecursionError:
        print("Stopped: the agent needed too many steps. Try a more specific question.")
    except Exception as exc:
        logger.error("Request failed: %s: %s", type(exc).__name__, exc, exc_info=view.debug)
    return False


async def chat_loop(agent: CompiledStateGraph, thread_id: str, view: ConsoleView) -> None:
    print("Ask a question (empty line or Ctrl+D to quit).")
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            return
        if not question:
            return
        await ask_safely(agent, question, thread_id, view)


async def run(
    settings: Settings, model: BaseChatModel, question: str | None, view: ConsoleView
) -> int:
    if not settings.rag_server_url and not knowledge_base_exists(settings):
        logger.error("Knowledge base not found. Run `python ingest.py` first.")
        return 1

    client = MultiServerMCPClient(build_server_connections(settings))
    async with AsyncExitStack() as stack:
        logger.info("Starting MCP servers (the knowledge base loads its embedding model first)")
        try:
            tools = await load_tools_with_live_sessions(client, stack, view.tool_interceptors)
        except Exception as exc:
            logger.error(
                "Could not start the MCP servers (see their errors above): %s", exc,
                exc_info=view.debug,
            )
            return 1
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[plain_text_tool_results],
            checkpointer=InMemorySaver(),
        )
        thread_id = uuid.uuid4().hex
        if question:
            return 0 if await ask_safely(agent, question, thread_id, view) else 1
        await chat_loop(agent, thread_id, view)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="stream the model's reasoning and show token usage and full tool results",
    )
    args = parser.parse_args()
    configure_logging()

    try:
        settings = Settings.from_env()
        model = build_chat_model(settings)
    except (ValueError, ImportError) as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2
    logger.info("Using chat model %s", settings.chat_model)

    try:
        return asyncio.run(run(settings, model, args.question, ConsoleView(debug=args.debug)))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
