from typing import Any

import pytest

from mascan.tools.common.source_fetch import (
    InaccessibleSourceError,
    OperationalSourceFetchError,
    SourceFetchTool,
    UnsafeSourceUrlError,
)


class _Document:
    markdown = "Page evidence"
    metadata = {"title": "Evidence page", "url": "https://example.com/final"}


def test_source_fetch_uses_firecrawl_scrape(mocker: Any) -> None:
    tool = SourceFetchTool(api_key="test")
    tool._validate_public_url = mocker.Mock()  # type: ignore[method-assign]
    tool.client = mocker.Mock()
    tool.client.scrape.return_value = _Document()

    result = tool.run(url="https://example.com/source")

    assert result.success is True
    assert result.data is not None
    assert result.data["title"] == "Evidence page"
    assert result.data["final_url"] == "https://example.com/final"
    tool.client.scrape.assert_called_once_with(
        "https://example.com/source",
        formats=["markdown"],
        only_main_content=True,
        timeout=30_000,
    )


def test_source_fetch_returns_structured_failure_for_empty_content(mocker: Any) -> None:
    tool = SourceFetchTool(api_key="test")
    tool._validate_public_url = mocker.Mock()  # type: ignore[method-assign]
    tool._fetch_direct = mocker.Mock(  # type: ignore[method-assign]
        side_effect=OperationalSourceFetchError("Direct page had no readable content")
    )
    tool.client = mocker.Mock()
    empty = mocker.Mock(markdown="", metadata={})
    tool.client.scrape.return_value = empty

    result = tool.run(url="https://example.com/empty")

    assert result.success is False
    assert "no readable page content" in (result.error or "")
    assert result.metadata["failure_kind"] == "operational"


def test_source_fetch_uses_direct_fallback_after_firecrawl_failure(mocker: Any) -> None:
    tool = SourceFetchTool(api_key="test")
    tool._validate_public_url = mocker.Mock()  # type: ignore[method-assign]
    tool._fetch_with_retries = mocker.Mock(  # type: ignore[method-assign]
        side_effect=RuntimeError("document_antibot")
    )
    tool._fetch_direct = mocker.Mock(  # type: ignore[method-assign]
        return_value={
            "requested_url": "https://example.com/source",
            "final_url": "https://example.com/source",
            "title": "Direct source",
            "markdown": "Readable fallback evidence",
            "checked_at": "2026-07-19T00:00:00+00:00",
            "truncated": False,
            "retrieval_method": "direct",
        }
    )

    result = tool.run(url="https://example.com/source")

    assert result.success is True
    assert result.data is not None
    assert result.data["markdown"] == "Readable fallback evidence"
    assert result.metadata["retrieval_method"] == "direct"


def test_source_fetch_marks_confirmed_missing_fallback_inaccessible(mocker: Any) -> None:
    tool = SourceFetchTool(api_key="test")
    tool._validate_public_url = mocker.Mock()  # type: ignore[method-assign]
    tool._fetch_with_retries = mocker.Mock(  # type: ignore[method-assign]
        side_effect=RuntimeError("document_antibot")
    )
    tool._fetch_direct = mocker.Mock(  # type: ignore[method-assign]
        side_effect=InaccessibleSourceError("HTTP 404")
    )

    result = tool.run(url="https://example.com/missing")

    assert result.success is False
    assert result.metadata["failure_kind"] == "inaccessible"


def test_source_fetch_marks_antibot_failure_operational(mocker: Any) -> None:
    tool = SourceFetchTool(api_key="test")
    tool._validate_public_url = mocker.Mock()  # type: ignore[method-assign]
    tool._fetch_with_retries = mocker.Mock(  # type: ignore[method-assign]
        side_effect=RuntimeError("document_antibot")
    )
    tool._fetch_direct = mocker.Mock(  # type: ignore[method-assign]
        side_effect=OperationalSourceFetchError("anti-bot challenge page")
    )

    result = tool.run(url="https://example.com/protected")

    assert result.success is False
    assert result.metadata["failure_kind"] == "operational"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://127.0.0.1/",
        "http://user:password@example.com/",
        "http://postgres/",
    ],
)
def test_source_fetch_rejects_non_public_urls(url: str) -> None:
    with pytest.raises(UnsafeSourceUrlError):
        SourceFetchTool._validate_public_url(url)
