from mascan.orchestrator.attribution import parse_attribution_document


def test_parser_uses_summary_ast_and_backward_attribution() -> None:
    report = """\
# Final Report

## Summary

The EU has a renewable target. Hydrogen production should increase [1](https://example.com/eu).

- A second claim [2](https://example.org/market).

```md
Fake citation [9](https://fake.example/code)
```

## Sources

1. [EU](https://example.com/eu)
3. [Unused Agent Source](https://agent.example/unused)
"""

    document = parse_attribution_document(report)

    assert [citation.url for citation in document.citations] == [
        "https://example.com/eu",
        "https://example.org/market",
    ]
    assert [attribution.claim for attribution in document.attributions] == [
        "The EU has a renewable target.",
        "Hydrogen production should increase.",
        "A second claim.",
    ]
    assert document.attributions[0].citations[0].number == 1
    assert all("fake.example" not in citation.url for citation in document.citations)
    assert all("agent.example" not in citation.url for citation in document.citations)


def test_parser_keeps_all_unique_body_urls_without_a_ten_source_cap() -> None:
    paragraphs = [
        f"Claim {number} [${number}](https://example.com/{number}).".replace("[$", "[")
        for number in range(1, 13)
    ]
    report = "# Final Report\n\n## Summary\n\n" + "\n\n".join(paragraphs)

    document = parse_attribution_document(report)

    assert len(document.citations) == 12
    assert document.citations[-1].url == "https://example.com/12"


def test_parser_ignores_uncited_claims() -> None:
    report = "## Summary\n\nAn important uncited number is 42."

    document = parse_attribution_document(report)

    assert document.citations == ()
    assert document.attributions == ()


def test_uploaded_document_bare_citation_is_outside_public_validation() -> None:
    document = parse_attribution_document(
        "## Summary\n\nEVONIK reported higher earnings [3].\n\n## Sources\n\n3. Factsheet"
    )

    assert document.citations == ()
    assert document.attributions == ()


def test_parser_accepts_safe_uploaded_file_links() -> None:
    document = parse_attribution_document(
        "## Summary\n\nEVONIK reported higher earnings [3](/rag/files/EVONIK%20Q1%202026.pdf)."
    )

    assert document.citations[0].url == "/rag/files/EVONIK%20Q1%202026.pdf"
    assert document.attributions[0].claim == "EVONIK reported higher earnings."


def test_parser_rejects_arbitrary_or_unsafe_local_links() -> None:
    document = parse_attribution_document(
        "## Summary\n\n"
        "Local [1](/etc/passwd). "
        "Encoded traversal [2](/rag/files/..%2Fsecret.pdf). "
        "Null byte [3](/rag/files/report%00.pdf)."
    )

    assert document.citations == ()
    assert document.attributions == ()
