from typing import Any

from mascan.tools.common.web_search import WebSearchTool


class _Doc:
    def __init__(self, title: str, url: str, markdown: str) -> None:
        self.title = title
        self.url = url
        self.markdown = markdown
        self.metadata = {"title": title, "url": url}


def test_truncate_markdown_caps_long_pages() -> None:
    long_body = "x" * (WebSearchTool.MAX_MARKDOWN_CHARS + 500)

    truncated = WebSearchTool._truncate_markdown(long_body)

    assert len(truncated) < len(long_body)
    assert truncated.endswith("[...truncated...]")


def test_truncate_markdown_leaves_short_pages_untouched() -> None:
    body = "short body"

    assert WebSearchTool._truncate_markdown(body) == body


def test_search_impl_truncates_each_result(mocker: Any) -> None:
    tool = WebSearchTool(api_key="test")
    long_body = "y" * (WebSearchTool.MAX_MARKDOWN_CHARS + 1000)
    response = mocker.Mock()
    response.web = [_Doc("Title", "https://example.com", long_body)]

    fake_client = mocker.Mock()
    fake_client.search.return_value = response
    tool.client = fake_client

    results = tool.search_impl(query="anything", max_results=1)

    assert len(results) == 1
    assert results[0]["markdown"].endswith("[...truncated...]")
    assert len(results[0]["markdown"]) <= WebSearchTool.MAX_MARKDOWN_CHARS + len(
        "\n\n[...truncated...]"
    )
