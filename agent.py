"""Agentic RAG: a LangChain agent that answers from the knowledge base and acts through MCP tools."""

import argparse
import asyncio
import logging
import sys
import uuid
from collections.abc import Iterable
from contextlib import AsyncExitStack
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from config import PROJECT_DIR, Settings, configure_logging
from knowledge_base import knowledge_base_exists
from models import build_chat_model, plain_text_tool_results

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
    client: MultiServerMCPClient, stack: AsyncExitStack
) -> list[BaseTool]:
    """Keep one session per server open, so each server starts once instead of per tool call."""
    tools: list[BaseTool] = []
    for server_name in client.connections:
        session = await stack.enter_async_context(client.session(server_name))
        server_tools = await load_mcp_tools(session, server_name=server_name)
        logger.info("%s tools: %s", server_name, [tool.name for tool in server_tools])
        tools.extend(server_tools)
    return tools


def print_messages(messages: Iterable[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  -> {call['name']}({call['args']})")
        elif isinstance(message, AIMessage) and message.text:
            print(f"\nAssistant: {message.text}\n")
        elif isinstance(message, ToolMessage):
            preview = " ".join(message.text.split())[:TOOL_PREVIEW_CHARS]
            print(f"  <- {message.name} [{message.status}]: {preview}")


async def ask(agent: CompiledStateGraph, question: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_AGENT_STEPS}
    inputs = {"messages": [{"role": "user", "content": question}]}
    async for update in agent.astream(inputs, config=config, stream_mode="updates"):
        for node_output in update.values():
            if isinstance(node_output, dict):
                print_messages(node_output.get("messages", []))


async def ask_safely(agent: CompiledStateGraph, question: str, thread_id: str) -> bool:
    """Answer one question, reporting failures (e.g. model server not running) without crashing."""
    try:
        await ask(agent, question, thread_id)
        return True
    except GraphRecursionError:
        print("Stopped: the agent needed too many steps. Try a more specific question.")
    except Exception as exc:
        logger.error("Request failed: %s: %s", type(exc).__name__, exc)
    return False


async def chat_loop(agent: CompiledStateGraph, thread_id: str) -> None:
    print("Ask a question (empty line or Ctrl+D to quit).")
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            return
        if not question:
            return
        await ask_safely(agent, question, thread_id)


async def run(settings: Settings, model: BaseChatModel, question: str | None) -> int:
    if not settings.rag_server_url and not knowledge_base_exists(settings):
        logger.error("Knowledge base not found. Run `python ingest.py` first.")
        return 1

    client = MultiServerMCPClient(build_server_connections(settings))
    async with AsyncExitStack() as stack:
        tools = await load_tools_with_live_sessions(client, stack)
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[plain_text_tool_results],
            checkpointer=InMemorySaver(),
        )
        thread_id = uuid.uuid4().hex
        if question:
            return 0 if await ask_safely(agent, question, thread_id) else 1
        await chat_loop(agent, thread_id)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="ask one question and exit")
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
        return asyncio.run(run(settings, model, args.question))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
