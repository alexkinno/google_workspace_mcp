"""Unit tests for the Google Chat list_members tool."""

from unittest.mock import Mock

import pytest
from googleapiclient.errors import HttpError

from gchat import chat_helpers
from gchat.chat_tools import list_members


def _unwrap(tool):
    fn = getattr(tool, "fn", tool)
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


@pytest.fixture(autouse=True)
def _clear_sender_cache():
    chat_helpers._sender_name_cache.clear()
    yield
    chat_helpers._sender_name_cache.clear()


def _chat_service(*pages):
    service = Mock()
    service.spaces().members().list().execute.side_effect = list(pages)
    return service


def _people_service(names):
    service = Mock()
    service.people().getBatchGet().execute.return_value = {
        "responses": [
            {
                "requestedResourceName": resource,
                "person": {"names": [{"displayName": name}]},
            }
            for resource, name in names.items()
        ]
    }
    return service


@pytest.mark.asyncio
async def test_list_members_returns_mention_tokens_for_humans_across_pages():
    chat_service = _chat_service(
        {
            "memberships": [
                {"member": {"name": "users/1", "type": "HUMAN"}},
                {"member": {"name": "users/app", "type": "BOT"}},
            ],
            "nextPageToken": "next",
        },
        {"memberships": [{"member": {"name": "users/2", "type": "HUMAN"}}]},
    )
    people_service = _people_service({"people/1": "Alice Smith"})

    result = await _unwrap(list_members)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        space_id="spaces/AAA",
    )

    assert "Found 2 human members in space 'spaces/AAA'" in result
    assert "- Alice Smith (mention: <users/1>)" in result
    # A member without a name gets no token, so it cannot be mentioned by guess.
    assert "<users/2>" not in result
    assert "1 other member could not be named and cannot be mentioned." in result
    assert "users/app" not in result


@pytest.mark.asyncio
async def test_list_members_offers_no_tokens_when_no_name_resolves():
    chat_service = _chat_service(
        {"memberships": [{"member": {"name": "users/1", "type": "HUMAN"}}]}
    )
    people_service = Mock()
    people_service.people().getBatchGet().execute.side_effect = Exception(
        "People API has not been used in project"
    )

    result = await _unwrap(list_members)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        space_id="spaces/AAA",
    )

    assert "Could not resolve the names of the 1 human member of space" in result
    assert "Send the message without mentions." in result
    assert "<users/" not in result


@pytest.mark.asyncio
async def test_list_members_reports_space_without_human_members():
    chat_service = _chat_service({"memberships": []})

    result = await _unwrap(list_members)(
        chat_service=chat_service,
        people_service=Mock(),
        user_google_email="me@example.com",
        space_id="spaces/AAA",
    )

    assert result == "No human members found in space 'spaces/AAA'."


@pytest.mark.asyncio
async def test_list_members_propagates_api_errors_instead_of_reporting_empty():
    chat_service = Mock()
    chat_service.spaces().members().list().execute.side_effect = HttpError(
        Mock(status=403), b"forbidden"
    )

    with pytest.raises(HttpError):
        await _unwrap(list_members)(
            chat_service=chat_service,
            people_service=Mock(),
            user_google_email="me@example.com",
            space_id="spaces/AAA",
        )
