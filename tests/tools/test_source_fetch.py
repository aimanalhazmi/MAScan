from typing import Any

import pytest

from mascan.tools.common.source_fetch import SourceFetchTool, UnsafeSourceUrlError


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
    tool.client = mocker.Mock()
    empty = mocker.Mock(markdown="", metadata={})
    tool.client.scrape.return_value = empty

    result = tool.run(url="https://example.com/empty")

    assert result.success is False
    assert "no readable page content" in (result.error or "")


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
