"""MCP server that lets the agent open and look up support tickets."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from config import Settings

Priority = Literal["low", "normal", "high"]

settings = Settings.from_env()
mcp = FastMCP(
    "support-tickets",
    instructions="Create and look up customer support tickets.",
    log_level="WARNING",
)


def load_tickets() -> dict[str, dict[str, Any]]:
    if not settings.tickets_file.exists():
        return {}
    try:
        return json.loads(settings.tickets_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ticket store {settings.tickets_file} is corrupted") from exc


def save_tickets(tickets: dict[str, dict[str, Any]]) -> None:
    settings.tickets_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = settings.tickets_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    temp_file.replace(settings.tickets_file)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False))
def create_ticket(
    customer_email: str, subject: str, description: str, priority: Priority = "normal"
) -> str:
    """Open a support ticket for a customer and return its id.

    Only call this after checking the relevant policy in the knowledge base.
    Put the policy citation and your eligibility reasoning in the description.
    """
    customer_email, subject, description = (
        customer_email.strip(), subject.strip(), description.strip()
    )
    if "@" not in customer_email:
        raise ValueError("customer_email must be a valid email address")
    if not subject or not description:
        raise ValueError("subject and description must not be empty")

    tickets = load_tickets()
    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
    tickets[ticket_id] = {
        "customer_email": customer_email,
        "subject": subject,
        "description": description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_tickets(tickets)
    return f"Created ticket {ticket_id} with {priority} priority."


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_ticket(ticket_id: str) -> dict[str, Any]:
    """Look up a support ticket by id, e.g. TCK-1A2B3C4D."""
    normalized_id = ticket_id.strip().upper()
    ticket = load_tickets().get(normalized_id)
    if ticket is None:
        raise ValueError(f"No ticket found with id {normalized_id}")
    return {"id": normalized_id, **ticket}


if __name__ == "__main__":
    mcp.run(transport="stdio")
