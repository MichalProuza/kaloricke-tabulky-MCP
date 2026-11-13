"""Network-error reporting, public health payload and MCP tool metadata."""

from __future__ import annotations

from typing import Any

from aiohttp import ClientConnectionError
from mcp.server.fastmcp import FastMCP
import pytest
from starlette.testclient import TestClient

from kaloricke_tabulky_mcp.account_registry import AccountRegistry
from kaloricke_tabulky_mcp.client import (
    AccountCredentials,
    KalorickeTabulkyClient,
    KalorickeTabulkyNetworkError,
)
from kaloricke_tabulky_mcp.config import load_settings
from kaloricke_tabulky_mcp.main import _add_health_route
from kaloricke_tabulky_mcp.tools import _error, setup_tools

ENV = {
    "KT_ACCOUNTS": "personal",
    "KT_ACCOUNT_PERSONAL_EMAIL": "fixture@example.com",
    "KT_ACCOUNT_PERSONAL_PASSWORD": "fixture-secret",
    "KT_DEFAULT_ACCOUNT": "personal",
}

READ_ONLY = {
    "health_check",
    "list_accounts",
    "check_account_auth",
    "search_food",
    "list_custom_recipes",
    "prepare_recipe_import",
    "get_diary_range",
    "get_diary_entries",
    "get_diary_summary",
}


class FailingSession:
    closed = False

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise self.error


def _client(error: Exception) -> tuple[KalorickeTabulkyClient, FailingSession]:
    session = FailingSession(error)
    client = KalorickeTabulkyClient(
        AccountCredentials(
            alias="personal", email="fixture@example.com", password="fixture-secret"
        ),
        session=session,  # type: ignore[arg-type]
    )
    client._cookies = "session=fixture-cookie"  # noqa: SLF001 - skip login
    return client, session


def _app() -> tuple[FastMCP, AccountRegistry]:
    registry = AccountRegistry(load_settings(ENV))
    mcp = FastMCP(name="test")
    setup_tools(mcp, registry)
    return mcp, registry


@pytest.mark.parametrize("error", [ClientConnectionError("down"), TimeoutError()])
async def test_read_network_failure_is_a_reported_tool_error(error) -> None:
    client, session = _client(error)

    with pytest.raises(KalorickeTabulkyNetworkError) as caught:
        await client._request_with_reauth("GET", "https://kt.invalid/x")  # noqa: SLF001

    assert caught.value.outcome_unknown is False
    assert session.calls == 1
    assert "outcome_unknown" not in _error(caught.value)


async def test_write_network_failure_reports_unknown_outcome() -> None:
    client, _ = _client(TimeoutError())

    with pytest.raises(KalorickeTabulkyNetworkError) as caught:
        await client._request_with_reauth("POST", "https://kt.invalid/x", json={})  # noqa: SLF001

    result = _error(caught.value)
    assert result["outcome_unknown"] is True
    assert "check the diary before retrying" in result["error"]
    assert "fixture-cookie" not in result["error"]


def test_public_health_route_omits_auth_diagnostics() -> None:
    mcp, registry = _app()
    registry._auth_diagnostics["personal"] = {  # noqa: SLF001 - contract test
        "status": "error",
        "message": "Invalid credentials; email=f***@example.com, password_len=14",
    }
    _add_health_route(mcp, registry)

    response = TestClient(mcp.streamable_http_app()).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["configured_accounts"] == [
        {"alias": "personal", "configured": True}
    ]
    assert "auth_diagnostics" not in response.json()
    assert "password_len" not in response.text
    # The authenticated MCP tool keeps the diagnostics.
    assert "personal" in registry.safe_health()["auth_diagnostics"]


async def test_tools_declare_read_only_and_destructive_hints() -> None:
    mcp, _ = _app()
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    for name, tool in tools.items():
        assert tool.annotations is not None, name
        assert tool.annotations.readOnlyHint is (name in READ_ONLY), name
    for name in ("record_food", "record_food_batch", "record_recipe_serving"):
        assert tools[name].annotations.destructiveHint is False
    for name in ("delete_diary_entry", "create_custom_recipe", "record_weight"):
        assert tools[name].annotations.destructiveHint is True


async def test_batch_and_recipe_items_expose_their_fields_in_the_schema() -> None:
    mcp, _ = _app()
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    for name, argument in (
        ("record_food_batch", "items"),
        ("prepare_recipe_import", "ingredients"),
        ("create_custom_recipe", "ingredients"),
    ):
        schema = tools[name].inputSchema
        reference = schema["properties"][argument]["items"]["$ref"].split("/")[-1]
        item = schema["$defs"][reference]
        assert {"query", "food_guid", "amount", "unit", "meal_type"} <= set(
            item["properties"]
        )
        assert item["required"] == ["amount"]
        assert item["additionalProperties"] is False
