"""
Unit tests for Google Chat MCP tools — attachment support
"""

import asyncio
import inspect
import ssl
from urllib.parse import urlparse

import pytest
from unittest.mock import AsyncMock, Mock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _make_message(text="Hello", attachments=None, msg_name="spaces/S/messages/M"):
    """Build a minimal Chat API message dict for testing."""
    msg = {
        "name": msg_name,
        "text": text,
        "createTime": "2025-01-01T00:00:00Z",
        "sender": {"name": "users/123", "displayName": "Test User"},
    }
    if attachments is not None:
        msg["attachment"] = attachments
    return msg


def _make_attachment(
    name="spaces/S/messages/M/attachments/A",
    content_name="image.png",
    content_type="image/png",
    resource_name="spaces/S/attachments/A",
):
    att = {
        "name": name,
        "contentName": content_name,
        "contentType": content_type,
        "source": "UPLOADED_CONTENT",
    }
    if resource_name:
        att["attachmentDataRef"] = {"resourceName": resource_name}
    return att


def _unwrap(tool):
    """Unwrap a FunctionTool + decorator chain to the original async function."""
    fn = getattr(tool, "fn", tool)
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


# ---------------------------------------------------------------------------
# get_messages: attachment metadata appears in output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_get_messages_shows_attachment_metadata(mock_resolve):
    """When a message has attachments, get_messages should surface their metadata."""
    mock_resolve.return_value = "Test User"

    att = _make_attachment()
    msg = _make_message(attachments=[att])

    chat_service = Mock()
    chat_service.spaces().get().execute.return_value = {"displayName": "Test Space"}
    chat_service.spaces().messages().list().execute.return_value = {"messages": [msg]}

    people_service = Mock()

    from gchat.chat_tools import get_messages

    result = await _unwrap(get_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        space_id="spaces/S",
    )

    assert "[attachment 0: image.png (image/png)]" in result
    assert "download_chat_attachment" in result


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_get_messages_no_attachments_unchanged(mock_resolve):
    """Messages without attachments should not include attachment lines."""
    mock_resolve.return_value = "Test User"

    msg = _make_message(text="Plain text message")

    chat_service = Mock()
    chat_service.spaces().get().execute.return_value = {"displayName": "Test Space"}
    chat_service.spaces().messages().list().execute.return_value = {"messages": [msg]}

    people_service = Mock()

    from gchat.chat_tools import get_messages

    result = await _unwrap(get_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        space_id="spaces/S",
    )

    assert "Plain text message" in result
    assert "[attachment" not in result


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_get_messages_multiple_attachments(mock_resolve):
    """Multiple attachments should each appear with their index."""
    mock_resolve.return_value = "Test User"

    attachments = [
        _make_attachment(content_name="photo.jpg", content_type="image/jpeg"),
        _make_attachment(
            name="spaces/S/messages/M/attachments/B",
            content_name="doc.pdf",
            content_type="application/pdf",
        ),
    ]
    msg = _make_message(attachments=attachments)

    chat_service = Mock()
    chat_service.spaces().get().execute.return_value = {"displayName": "Test Space"}
    chat_service.spaces().messages().list().execute.return_value = {"messages": [msg]}

    people_service = Mock()

    from gchat.chat_tools import get_messages

    result = await _unwrap(get_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        space_id="spaces/S",
    )

    assert "[attachment 0: photo.jpg (image/jpeg)]" in result
    assert "[attachment 1: doc.pdf (application/pdf)]" in result


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_get_messages_exposes_message_filter_and_forwards_it(mock_resolve):
    """get_messages should expose message_filter publicly and pass it to the Chat API."""
    mock_resolve.return_value = "Test User"

    msg = _make_message(text="Filtered message")
    chat_service = Mock()
    chat_service.spaces().get().execute.return_value = {"displayName": "Test Space"}
    chat_service.spaces().messages().list().execute.return_value = {"messages": [msg]}
    people_service = Mock()

    from gchat.chat_tools import get_messages

    public_fn = getattr(get_messages, "fn", get_messages)
    params = inspect.signature(public_fn).parameters

    assert "message_filter" in params
    assert "filter" not in params

    result = await _unwrap(get_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        space_id="spaces/S",
        message_filter="thread.name = spaces/S/threads/T",
    )

    assert "Filtered message" in result
    list_kwargs = chat_service.spaces().messages().list.call_args.kwargs
    assert list_kwargs["filter"] == "thread.name = spaces/S/threads/T"


@pytest.mark.asyncio
async def test_get_messages_resolves_senders_sequentially(monkeypatch):
    """get_messages should avoid concurrent People API sender resolution."""
    state = {"current": 0, "max": 0}

    async def fake_resolve(_people_service, sender_obj):
        state["current"] += 1
        state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.01)
        try:
            return f"Resolved {sender_obj['name']}"
        finally:
            state["current"] -= 1

    msg_one = _make_message(text="First message", msg_name="spaces/S/messages/M1")
    msg_one["sender"] = {"name": "users/1"}
    msg_two = _make_message(text="Second message", msg_name="spaces/S/messages/M2")
    msg_two["sender"] = {"name": "users/2"}

    chat_service = Mock()
    chat_service.spaces().get().execute.return_value = {"displayName": "Test Space"}
    chat_service.spaces().messages().list().execute.return_value = {
        "messages": [msg_one, msg_two]
    }
    people_service = Mock()

    monkeypatch.setattr("gchat.chat_tools._resolve_sender", fake_resolve)

    from gchat.chat_tools import get_messages

    result = await _unwrap(get_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        space_id="spaces/S",
    )

    assert "Resolved users/1" in result
    assert "Resolved users/2" in result
    assert state["max"] == 1


# ---------------------------------------------------------------------------
# search_messages: attachment indicator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_search_messages_shows_attachment_indicator(mock_resolve):
    """search_messages should show [attachment: filename] for messages with attachments."""
    mock_resolve.return_value = "Test User"

    att = _make_attachment(content_name="report.pdf", content_type="application/pdf")
    msg = _make_message(text="Here is the report", attachments=[att])
    msg["_space_name"] = "General"

    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [{"name": "spaces/S", "displayName": "General"}]
    }
    chat_service.spaces().messages().list().execute.return_value = {"messages": [msg]}

    people_service = Mock()

    from gchat.chat_tools import search_messages

    result = await _unwrap(search_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        query="report",
    )

    assert "[attachment: report.pdf (application/pdf)]" in result


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_search_messages_combines_filters_and_uses_page_size(mock_resolve):
    """Cross-space search should honor page_size and only send supported API filters."""
    mock_resolve.return_value = "Test User"

    msg = _make_message(text="Deploy finished")
    msg["_space_name"] = "General"

    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [{"name": "spaces/S", "displayName": "General"}]
    }
    chat_service.spaces().messages().list().execute.return_value = {"messages": [msg]}
    people_service = Mock()

    from gchat.chat_tools import search_messages

    result = await _unwrap(search_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        query="deploy",
        time_filter='createTime > "2026-03-18T00:00:00-03:00"',
        page_size=7,
    )

    assert 'text "deploy" and createTime > "2026-03-18T00:00:00-03:00"' in result
    list_kwargs = chat_service.spaces().messages().list.call_args.kwargs
    assert list_kwargs["pageSize"] == 7
    assert list_kwargs["filter"] == 'createTime > "2026-03-18T00:00:00-03:00"'


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_search_messages_query_only_filters_client_side_without_api_filter(
    mock_resolve,
):
    """Query-only search should avoid unsupported Chat API text filters."""
    mock_resolve.return_value = "Test User"

    matching = _make_message(text="Deploy finished")
    matching["_space_name"] = "General"
    non_matching = _make_message(text="Lunch plans")
    non_matching["_space_name"] = "General"

    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [{"name": "spaces/S", "displayName": "General"}]
    }
    chat_service.spaces().messages().list().execute.return_value = {
        "messages": [matching, non_matching]
    }
    people_service = Mock()

    from gchat.chat_tools import search_messages

    result = await _unwrap(search_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        query="deploy",
    )

    assert "Deploy finished" in result
    assert "Lunch plans" not in result
    list_kwargs = chat_service.spaces().messages().list.call_args.kwargs
    assert list_kwargs["pageSize"] == 25
    assert "filter" not in list_kwargs


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_search_messages_limits_parallel_space_fetches(mock_resolve, monkeypatch):
    """Cross-space search should cap concurrent threaded Chat API calls."""
    mock_resolve.return_value = "Test User"

    from gchat.chat_tools import _SEARCH_MESSAGES_MAX_CONCURRENT_SPACE_FETCHES
    from gchat.chat_tools import search_messages

    state = {"current": 0, "max": 0}

    async def fake_to_thread(fn, *args, **kwargs):
        state["current"] += 1
        state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.01)
        try:
            return fn(*args, **kwargs)
        finally:
            state["current"] -= 1

    monkeypatch.setattr("gchat.chat_tools.asyncio.to_thread", fake_to_thread)

    spaces = [{"name": f"spaces/S{i}", "displayName": f"Space {i}"} for i in range(5)]

    def list_messages(**kwargs):
        parent = kwargs["parent"]
        request = Mock()
        request.execute.return_value = {
            "messages": [_make_message(text=f"message from {parent}")]
        }
        return request

    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {"spaces": spaces}
    chat_service.spaces().messages().list.side_effect = list_messages
    people_service = Mock()

    result = await _unwrap(search_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        query="message",
        max_spaces=len(spaces),
    )

    assert "Found 5 messages matching 'text \"message\"'" in result
    assert state["max"] <= _SEARCH_MESSAGES_MAX_CONCURRENT_SPACE_FETCHES


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_search_messages_retries_ssl_per_space_without_restarting_search(
    mock_resolve, monkeypatch
):
    """A transient SSL failure in one space should retry only that space."""
    mock_resolve.return_value = "Test User"

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("gchat.chat_tools.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("gchat.chat_tools.asyncio.sleep", AsyncMock())

    from gchat.chat_tools import search_messages

    attempt_counts = {"spaces/S1": 0, "spaces/S2": 0}

    def list_messages(**kwargs):
        parent = kwargs["parent"]
        request = Mock()

        def execute():
            attempt_counts[parent] += 1
            if parent == "spaces/S1" and attempt_counts[parent] < 3:
                raise ssl.SSLError("read operation timed out")
            return {"messages": [_make_message(text=f"message from {parent}")]}

        request.execute.side_effect = execute
        return request

    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [
            {"name": "spaces/S1", "displayName": "Space 1"},
            {"name": "spaces/S2", "displayName": "Space 2"},
        ]
    }
    chat_service.spaces().messages().list.side_effect = list_messages
    people_service = Mock()

    result = await _unwrap(search_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="test@example.com",
        query="message",
        max_spaces=2,
    )

    assert "Found 2 messages matching 'text \"message\"'" in result
    assert attempt_counts == {"spaces/S1": 3, "spaces/S2": 1}
    assert chat_service.spaces().list().execute.call_count == 1


@pytest.mark.asyncio
@patch("gchat.chat_tools._resolve_sender", new_callable=AsyncMock)
async def test_search_messages_raises_transient_error_when_all_spaces_ssl_fail(
    mock_resolve, monkeypatch
):
    """If every searched space fails transiently, surface a transient network error."""
    mock_resolve.return_value = "Test User"

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("gchat.chat_tools.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("gchat.chat_tools.asyncio.sleep", AsyncMock())

    from core.utils import TransientNetworkError
    from gchat.chat_tools import search_messages

    def list_messages(**kwargs):  # noqa: ARG001
        request = Mock()
        request.execute.side_effect = ssl.SSLError("connection reset")
        return request

    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [{"name": "spaces/S1", "displayName": "Space 1"}]
    }
    chat_service.spaces().messages().list.side_effect = list_messages
    people_service = Mock()

    with pytest.raises(TransientNetworkError, match="transient SSL error"):
        await _unwrap(search_messages)(
            chat_service=chat_service,
            people_service=people_service,
            user_google_email="test@example.com",
            query="message",
        )


# ---------------------------------------------------------------------------
# download_chat_attachment: edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_no_attachments():
    """Should return a clear message when the message has no attachments."""
    service = Mock()
    service.spaces().messages().get().execute.return_value = _make_message()

    from gchat.chat_tools import download_chat_attachment

    result = await _unwrap(download_chat_attachment)(
        service=service,
        user_google_email="test@example.com",
        message_id="spaces/S/messages/M",
    )

    assert "No attachments found" in result


@pytest.mark.asyncio
async def test_download_invalid_index():
    """Should return an error for out-of-range attachment_index."""
    msg = _make_message(attachments=[_make_attachment()])
    service = Mock()
    service.spaces().messages().get().execute.return_value = msg

    from gchat.chat_tools import download_chat_attachment

    result = await _unwrap(download_chat_attachment)(
        service=service,
        user_google_email="test@example.com",
        message_id="spaces/S/messages/M",
        attachment_index=5,
    )

    assert "Invalid attachment_index" in result
    assert "1 attachment(s)" in result


@pytest.mark.asyncio
async def test_download_uses_api_media_endpoint():
    """Should always use chat.googleapis.com media endpoint, not downloadUri."""
    fake_bytes = b"fake image content"
    att = _make_attachment()
    # Even with a downloadUri present, we should use the API endpoint
    att["downloadUri"] = "https://chat.google.com/api/get_attachment_url?bad=url"
    msg = _make_message(attachments=[att])

    service = Mock()
    service.spaces().messages().get().execute.return_value = msg
    service._http.credentials.token = "fake-access-token"

    from gchat.chat_tools import download_chat_attachment

    saved = Mock()
    saved.path = "/tmp/image_abc.png"
    saved.file_id = "abc"

    mock_response = Mock()
    mock_response.content = fake_bytes
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.request.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("core.file_limits.httpx.AsyncClient", return_value=mock_client),
        patch("auth.oauth_config.is_stateless_mode", return_value=False),
        patch("core.config.get_transport_mode", return_value="stdio"),
        patch("core.attachment_storage.get_attachment_storage") as mock_get_storage,
    ):
        mock_get_storage.return_value.save_attachment_bytes.return_value = saved

        result = await _unwrap(download_chat_attachment)(
            service=service,
            user_google_email="test@example.com",
            message_id="spaces/S/messages/M",
            attachment_index=0,
        )

    assert "image.png" in result
    assert "/tmp/image_abc.png" in result
    assert "Saved to:" in result

    # Verify we used the API endpoint with attachmentDataRef.resourceName
    call_args = mock_client.request.call_args
    url_used = call_args.args[1]
    parsed = urlparse(url_used)
    assert call_args.args[0] == "GET"
    assert parsed.scheme == "https"
    assert parsed.hostname == "chat.googleapis.com"
    assert "alt=media" in url_used
    assert "spaces/S/attachments/A" in parsed.path
    assert "/messages/" not in parsed.path

    # Verify Bearer token
    assert call_args.kwargs["headers"]["Authorization"] == "Bearer fake-access-token"

    # Verify decoded bytes are written directly without a base64 round trip.
    save_args = mock_get_storage.return_value.save_attachment_bytes.call_args
    assert save_args.kwargs["filename"] == "image.png"
    assert save_args.kwargs["mime_type"] == "image/png"
    assert save_args.kwargs["file_bytes"] == fake_bytes


@pytest.mark.asyncio
async def test_download_falls_back_to_att_name():
    """When attachmentDataRef is missing, should fall back to attachment name."""
    fake_bytes = b"fetched content"
    att = _make_attachment(name="spaces/S/messages/M/attachments/A", resource_name=None)
    msg = _make_message(attachments=[att])

    service = Mock()
    service.spaces().messages().get().execute.return_value = msg
    service._http.credentials.token = "fake-access-token"

    saved = Mock()
    saved.path = "/tmp/image_fetched.png"
    saved.file_id = "f1"

    mock_response = Mock()
    mock_response.content = fake_bytes
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.request.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    from gchat.chat_tools import download_chat_attachment

    with (
        patch("core.file_limits.httpx.AsyncClient", return_value=mock_client),
        patch("auth.oauth_config.is_stateless_mode", return_value=False),
        patch("core.config.get_transport_mode", return_value="stdio"),
        patch("core.attachment_storage.get_attachment_storage") as mock_get_storage,
    ):
        mock_get_storage.return_value.save_attachment_bytes.return_value = saved

        result = await _unwrap(download_chat_attachment)(
            service=service,
            user_google_email="test@example.com",
            message_id="spaces/S/messages/M",
            attachment_index=0,
        )

    assert "image.png" in result
    assert "/tmp/image_fetched.png" in result

    # Falls back to attachment name when no attachmentDataRef
    call_args = mock_client.request.call_args
    assert "spaces/S/messages/M/attachments/A" in call_args.args[1]


@pytest.mark.asyncio
async def test_download_http_mode_returns_url():
    """In HTTP mode, should return a download URL instead of file path."""
    fake_bytes = b"image data"
    att = _make_attachment()
    msg = _make_message(attachments=[att])

    service = Mock()
    service.spaces().messages().get().execute.return_value = msg
    service._http.credentials.token = "fake-token"

    mock_response = Mock()
    mock_response.content = fake_bytes
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.request.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    saved = Mock()
    saved.path = "/tmp/image_alt.png"
    saved.file_id = "alt1"

    from gchat.chat_tools import download_chat_attachment

    with (
        patch("core.file_limits.httpx.AsyncClient", return_value=mock_client),
        patch("auth.oauth_config.is_stateless_mode", return_value=False),
        patch("core.config.get_transport_mode", return_value="http"),
        patch("core.attachment_storage.get_attachment_storage") as mock_get_storage,
        patch(
            "core.attachment_storage.get_attachment_url",
            return_value="http://localhost:8005/attachments/alt1",
        ),
    ):
        mock_get_storage.return_value.save_attachment_bytes.return_value = saved

        result = await _unwrap(download_chat_attachment)(
            service=service,
            user_google_email="test@example.com",
            message_id="spaces/S/messages/M",
            attachment_index=0,
        )

    assert "Download URL:" in result
    assert "expire after 1 hour" in result


@pytest.mark.asyncio
async def test_download_returns_error_on_failure():
    """When download fails, should return a clear error message."""
    att = _make_attachment()
    att["downloadUri"] = "https://storage.googleapis.com/fake?alt=media"
    msg = _make_message(attachments=[att])

    service = Mock()
    service.spaces().messages().get().execute.return_value = msg
    service._http.credentials.token = "fake-token"

    mock_client = AsyncMock()
    mock_client.request.side_effect = Exception("connection refused")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    from gchat.chat_tools import download_chat_attachment

    with patch("core.file_limits.httpx.AsyncClient", return_value=mock_client):
        result = await _unwrap(download_chat_attachment)(
            service=service,
            user_google_email="test@example.com",
            message_id="spaces/S/messages/M",
            attachment_index=0,
        )

    assert "Failed to download" in result
    assert "connection refused" in result


@pytest.mark.asyncio
async def test_download_rejects_oversized_when_capped(monkeypatch):
    """With WORKSPACE_MCP_MAX_FILE_BYTES set, oversized Chat downloads abort."""
    monkeypatch.setenv("WORKSPACE_MCP_MAX_FILE_BYTES", "10")
    att = _make_attachment()
    msg = _make_message(attachments=[att])

    service = Mock()
    service.spaces().messages().get().execute.return_value = msg
    service._http.credentials.token = "fake-token"

    from core.file_limits import FileTooLargeError
    from gchat.chat_tools import download_chat_attachment

    with patch(
        "gchat.chat_tools.download_http_url_bytes",
        side_effect=FileTooLargeError(
            'Error: "image.png" is too large to load into this MCP server '
            "(50 bytes; limit is 10 bytes via WORKSPACE_MCP_MAX_FILE_BYTES)."
        ),
    ):
        result = await _unwrap(download_chat_attachment)(
            service=service,
            user_google_email="test@example.com",
            message_id="spaces/S/messages/M",
            attachment_index=0,
        )

    assert result.startswith("Error:")
    assert "WORKSPACE_MCP_MAX_FILE_BYTES" in result
    assert "image.png" in result


# ---------------------------------------------------------------------------
# send_message: editing an existing message in place
# ---------------------------------------------------------------------------


def _mock_chat_service():
    """Chat service mock exposing a stable spaces().messages() resource."""
    service = Mock()
    messages = Mock()
    service.spaces.return_value.messages.return_value = messages
    return service, messages


@pytest.mark.asyncio
async def test_send_message_exposes_message_name_param():
    """send_message should expose message_name for in-place edits."""
    from gchat.chat_tools import send_message

    public_fn = getattr(send_message, "fn", send_message)
    params = inspect.signature(public_fn).parameters

    assert "message_name" in params
    assert params["message_name"].default is None


@pytest.mark.asyncio
async def test_send_message_edits_existing_message_instead_of_creating():
    """With message_name set, send_message should patch the message, not create one."""
    service, messages = _mock_chat_service()
    messages.patch.return_value.execute.return_value = {
        "name": "spaces/S/messages/M",
        "lastUpdateTime": "2025-01-01T00:00:00Z",
    }

    from gchat.chat_tools import send_message

    result = await _unwrap(send_message)(
        service=service,
        user_google_email="test@example.com",
        space_id="spaces/S",
        message_text="corrected text",
        message_name="spaces/S/messages/M",
    )

    assert messages.create.call_count == 0
    patch_kwargs = messages.patch.call_args.kwargs
    assert patch_kwargs["name"] == "spaces/S/messages/M"
    assert patch_kwargs["updateMask"] == "text"
    assert patch_kwargs["body"] == {"text": "corrected text"}
    assert "updated" in result.lower()
    assert "spaces/S/messages/M" in result


@pytest.mark.asyncio
async def test_send_message_without_message_name_still_creates():
    """The default path must stay a create, untouched by the edit support."""
    service, messages = _mock_chat_service()
    messages.create.return_value.execute.return_value = {
        "name": "spaces/S/messages/NEW",
        "createTime": "2025-01-01T00:00:00Z",
    }

    from gchat.chat_tools import send_message

    result = await _unwrap(send_message)(
        service=service,
        user_google_email="test@example.com",
        space_id="spaces/S",
        message_text="hello",
    )

    assert messages.patch.call_count == 0
    assert messages.create.call_args.kwargs["parent"] == "spaces/S"
    assert "sent" in result.lower()


@pytest.mark.asyncio
async def test_send_message_rejects_message_name_with_thread_reply():
    """An edit cannot also be a thread reply — the thread of a message is fixed."""
    service, messages = _mock_chat_service()

    from gchat.chat_tools import send_message
    from core.utils import UserInputError

    with pytest.raises(UserInputError):
        await _unwrap(send_message)(
            service=service,
            user_google_email="test@example.com",
            space_id="spaces/S",
            message_text="corrected text",
            message_name="spaces/S/messages/M",
            thread_name="spaces/S/threads/T",
        )

    assert messages.patch.call_count == 0
    assert messages.create.call_count == 0


@pytest.mark.asyncio
async def test_send_message_rejects_message_name_outside_the_space():
    """A message_name from another space would silently report the wrong space."""
    service, messages = _mock_chat_service()

    from gchat.chat_tools import send_message
    from core.utils import UserInputError

    with pytest.raises(UserInputError):
        await _unwrap(send_message)(
            service=service,
            user_google_email="test@example.com",
            space_id="spaces/S",
            message_text="corrected text",
            message_name="spaces/OTHER/messages/M",
        )

    assert messages.patch.call_count == 0


@pytest.mark.asyncio
async def test_send_message_rejects_malformed_message_name():
    """A bare message id cannot be patched — fail before the API does."""
    service, messages = _mock_chat_service()

    from gchat.chat_tools import send_message
    from core.utils import UserInputError

    with pytest.raises(UserInputError):
        await _unwrap(send_message)(
            service=service,
            user_google_email="test@example.com",
            space_id="spaces/S",
            message_text="corrected text",
            message_name="M",
        )

    assert messages.patch.call_count == 0


@pytest.mark.asyncio
async def test_send_message_edit_falls_back_to_create_time():
    """Older API responses omit lastUpdateTime; the confirmation must stay readable."""
    service, messages = _mock_chat_service()
    messages.patch.return_value.execute.return_value = {
        "name": "spaces/S/messages/M",
        "createTime": "2025-01-01T00:00:00Z",
    }

    from gchat.chat_tools import send_message

    result = await _unwrap(send_message)(
        service=service,
        user_google_email="test@example.com",
        space_id="spaces/S",
        message_text="corrected text",
        message_name="spaces/S/messages/M",
    )

    assert "2025-01-01T00:00:00Z" in result


@pytest.mark.asyncio
async def test_send_message_rejects_empty_message_name():
    """An empty message_name is a failed edit, not a request to post a new message."""
    service, messages = _mock_chat_service()

    from gchat.chat_tools import send_message
    from core.utils import UserInputError

    with pytest.raises(UserInputError):
        await _unwrap(send_message)(
            service=service,
            user_google_email="test@example.com",
            space_id="spaces/S",
            message_text="corrected text",
            message_name="",
        )

    assert messages.create.call_count == 0
    assert messages.patch.call_count == 0


# ---------------------------------------------------------------------------
# Naming direct messages and unnamed group chats
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_chat_name_caches():
    """Space/user name caches are module level; keep tests independent."""
    from gchat import chat_tools

    chat_tools._user_identity_cache.clear()
    chat_tools._space_name_cache.clear()
    yield
    chat_tools._user_identity_cache.clear()
    chat_tools._space_name_cache.clear()


def _mock_people_service(directory):
    """People service whose directory maps 'people/ID' -> (name, email)."""
    people_service = Mock()

    def get(resourceName=None, personFields=None):  # noqa: ARG001
        request = Mock()
        entry = directory.get(resourceName)
        if entry is None:
            request.execute.return_value = {}
        else:
            name, email = entry
            request.execute.return_value = {
                "names": [{"displayName": name}],
                "emailAddresses": [{"value": email}],
            }
        return request

    people_service.people().get.side_effect = get
    return people_service


def _membership(user_id):
    return {"member": {"name": user_id, "type": "HUMAN"}}


def _http_error(status, message):
    from googleapiclient.errors import HttpError

    resp = Mock(status=status, reason=message)
    return HttpError(resp=resp, content=f'{{"error": "{message}"}}'.encode())


@pytest.mark.asyncio
async def test_list_spaces_names_direct_message_after_other_member():
    """A DM has no displayName; it should be named after the other participant."""
    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [{"name": "spaces/DM1", "spaceType": "DIRECT_MESSAGE"}]
    }
    chat_service.spaces().members().list().execute.return_value = {
        "memberships": [_membership("users/1"), _membership("users/2")]
    }
    people_service = _mock_people_service(
        {
            "people/1": ("Me Myself", "me@example.com"),
            "people/2": ("Alice Smith", "alice@example.com"),
        }
    )

    from gchat.chat_tools import list_spaces

    result = await _unwrap(list_spaces)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        space_type="dm",
    )

    assert "DM: Alice Smith" in result
    assert "Unnamed Space" not in result
    assert "Me Myself" not in result


@pytest.mark.asyncio
async def test_list_spaces_names_unnamed_group_chat_after_members():
    """An unnamed group chat should list its other members."""
    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [{"name": "spaces/G1", "spaceType": "GROUP_CHAT"}]
    }
    chat_service.spaces().members().list().execute.return_value = {
        "memberships": [
            _membership("users/1"),
            _membership("users/2"),
            _membership("users/3"),
        ]
    }
    people_service = _mock_people_service(
        {
            "people/1": ("Me Myself", "me@example.com"),
            "people/2": ("Alice Smith", "alice@example.com"),
            "people/3": ("Bob Jones", "bob@example.com"),
        }
    )

    from gchat.chat_tools import list_spaces

    result = await _unwrap(list_spaces)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
    )

    assert "Group: Alice Smith, Bob Jones" in result


@pytest.mark.asyncio
async def test_list_spaces_keeps_display_name_and_skips_membership_lookup():
    """Named rooms should not trigger a memberships call."""
    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [
            {"name": "spaces/S1", "displayName": "Engineering", "spaceType": "SPACE"}
        ]
    }
    people_service = Mock()

    from gchat.chat_tools import list_spaces

    result = await _unwrap(list_spaces)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
    )

    assert "Engineering" in result
    assert chat_service.spaces().members().list.call_count == 0


@pytest.mark.asyncio
async def test_get_messages_names_direct_message_space():
    """get_messages should title a DM after the other participant, not 'Unknown Space'."""
    msg = _make_message(text="Hi there", msg_name="spaces/DM1/messages/M1")
    msg["sender"] = {"name": "users/2"}

    chat_service = Mock()
    chat_service.spaces().get().execute.return_value = {
        "name": "spaces/DM1",
        "spaceType": "DIRECT_MESSAGE",
    }
    chat_service.spaces().messages().list().execute.return_value = {"messages": [msg]}
    chat_service.spaces().members().list().execute.return_value = {
        "memberships": [_membership("users/1"), _membership("users/2")]
    }
    people_service = _mock_people_service(
        {
            "people/1": ("Me Myself", "me@example.com"),
            "people/2": ("Alice Smith", "alice@example.com"),
        }
    )

    from gchat.chat_tools import get_messages

    result = await _unwrap(get_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        space_id="spaces/DM1",
    )

    assert "Messages from 'DM: Alice Smith'" in result
    assert "Unknown Space" not in result


@pytest.mark.asyncio
async def test_get_messages_falls_back_to_senders_when_memberships_forbidden():
    """Without the memberships scope, name the DM from who has posted in it."""
    msg = _make_message(text="Hi there", msg_name="spaces/DM1/messages/M1")
    msg["sender"] = {"name": "users/2"}

    chat_service = Mock()
    chat_service.spaces().get().execute.return_value = {
        "name": "spaces/DM1",
        "spaceType": "DIRECT_MESSAGE",
    }
    chat_service.spaces().messages().list().execute.return_value = {"messages": [msg]}
    chat_service.spaces().members().list().execute.side_effect = _http_error(
        403, "insufficient scope"
    )
    people_service = _mock_people_service(
        {"people/2": ("Alice Smith", "alice@example.com")}
    )

    from gchat.chat_tools import get_messages

    result = await _unwrap(get_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        space_id="spaces/DM1",
    )

    assert "Messages from 'DM: Alice Smith'" in result


@pytest.mark.asyncio
async def test_get_messages_reports_space_kind_when_nothing_resolves():
    """An unresolvable DM should say what it is instead of 'Unknown Space'."""
    chat_service = Mock()
    chat_service.spaces().get().execute.return_value = {
        "name": "spaces/DM1",
        "spaceType": "DIRECT_MESSAGE",
    }
    chat_service.spaces().messages().list().execute.return_value = {"messages": []}
    chat_service.spaces().members().list().execute.side_effect = _http_error(
        403, "insufficient scope"
    )
    people_service = _mock_people_service({})

    from gchat.chat_tools import get_messages

    result = await _unwrap(get_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        space_id="spaces/DM1",
    )

    assert "Direct Message (participant unavailable)" in result


@pytest.mark.asyncio
async def test_search_messages_names_direct_message_spaces():
    """Search results should attribute DM hits to the other participant."""
    dm_msg = _make_message(text="message in dm", msg_name="spaces/DM1/messages/M1")
    dm_msg["sender"] = {"name": "users/2"}

    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [{"name": "spaces/DM1", "spaceType": "DIRECT_MESSAGE"}]
    }
    chat_service.spaces().messages().list().execute.return_value = {
        "messages": [dm_msg]
    }
    chat_service.spaces().members().list().execute.return_value = {
        "memberships": [_membership("users/1"), _membership("users/2")]
    }
    people_service = _mock_people_service(
        {
            "people/1": ("Me Myself", "me@example.com"),
            "people/2": ("Alice Smith", "alice@example.com"),
        }
    )

    from gchat.chat_tools import search_messages

    result = await _unwrap(search_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        query="message",
    )

    assert "in 'DM: Alice Smith'" in result
    assert "Unknown" not in result


@pytest.mark.asyncio
async def test_search_messages_names_single_space_target():
    """Searching one DM by ID should label it by participant, keeping the raw ID."""
    dm_msg = _make_message(text="message in dm", msg_name="spaces/DM1/messages/M1")
    dm_msg["sender"] = {"name": "users/2"}

    chat_service = Mock()
    chat_service.spaces().get().execute.return_value = {
        "name": "spaces/DM1",
        "spaceType": "DIRECT_MESSAGE",
    }
    chat_service.spaces().messages().list().execute.return_value = {
        "messages": [dm_msg]
    }
    chat_service.spaces().members().list().execute.return_value = {
        "memberships": [_membership("users/1"), _membership("users/2")]
    }
    people_service = _mock_people_service(
        {
            "people/1": ("Me Myself", "me@example.com"),
            "people/2": ("Alice Smith", "alice@example.com"),
        }
    )

    from gchat.chat_tools import search_messages

    result = await _unwrap(search_messages)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        query="message",
        space_id="spaces/DM1",
    )

    assert "space 'DM: Alice Smith' (ID: spaces/DM1)" in result


@pytest.mark.asyncio
async def test_space_name_resolution_is_cached_across_calls():
    """A resolved DM name should be reused without re-listing memberships."""
    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {
        "spaces": [{"name": "spaces/DM1", "spaceType": "DIRECT_MESSAGE"}]
    }
    chat_service.spaces().members().list().execute.return_value = {
        "memberships": [_membership("users/2")]
    }
    people_service = _mock_people_service(
        {"people/2": ("Alice Smith", "alice@example.com")}
    )

    from gchat.chat_tools import list_spaces

    call = lambda: _unwrap(list_spaces)(  # noqa: E731
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        space_type="dm",
    )

    assert "DM: Alice Smith" in await call()
    calls_after_first = chat_service.spaces().members().list.call_count
    assert "DM: Alice Smith" in await call()
    assert chat_service.spaces().members().list.call_count == calls_after_first


# ---------------------------------------------------------------------------
# list_spaces: spaceType filter syntax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("space_type", "expected_filter"),
    [
        ("dm", 'spaceType = "GROUP_CHAT" OR spaceType = "DIRECT_MESSAGE"'),
        ("room", 'spaceType = "SPACE"'),
    ],
)
@pytest.mark.asyncio
async def test_list_spaces_quotes_space_type_filter(space_type, expected_filter):
    """Chat rejects an unquoted enum value with INVALID_ARGUMENT."""
    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {"spaces": []}
    people_service = Mock()

    from gchat.chat_tools import list_spaces

    await _unwrap(list_spaces)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
        space_type=space_type,
    )

    assert chat_service.spaces().list.call_args.kwargs["filter"] == expected_filter


@pytest.mark.asyncio
async def test_list_spaces_sends_no_filter_for_all_space_types():
    """'all' must not constrain the listing."""
    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {"spaces": []}
    people_service = Mock()

    from gchat.chat_tools import list_spaces

    await _unwrap(list_spaces)(
        chat_service=chat_service,
        people_service=people_service,
        user_google_email="me@example.com",
    )

    assert "filter" not in chat_service.spaces().list.call_args.kwargs
