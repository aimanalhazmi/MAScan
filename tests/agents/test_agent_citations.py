from mascan.agents.economics.prompts import build_user_prompt as economics_prompt
from mascan.agents.environmental.prompts import build_user_prompt as environmental_prompt
from mascan.agents.legal.prompts import build_user_prompt as legal_prompt
from mascan.agents.political.prompts import build_user_prompt as political_prompt
from mascan.agents.social.prompts import build_user_prompt as social_prompt
from mascan.agents.sources import (
    cited_provided_sources,
    normalize_agent_citations,
    provided_sources_from_context,
)
from mascan.agents.technological.prompts import build_user_prompt as technological_prompt
from mascan.contracts.reports import Source


def test_all_agent_prompts_share_the_citation_contract() -> None:
    attachment = Source(
        name="factsheet.pdf",
        url="/rag/files/factsheet.pdf",
        metadata={
            "kind": "uploaded_document",
            "content": "[Page 2]\nAdjusted EBITDA was EUR 522 million.",
        },
    )
    context = {
        "user_input": "Use the attached factsheet.",
        "objective_context": "Assess company performance.",
        "provided_sources": [attachment],
    }
    prompts = [
        political_prompt(["policy"], "", context=context),
        economics_prompt(["costs"], "", context=context),
        social_prompt(["workforce"], "", context=context),
        environmental_prompt(["emissions"], context=context),
        legal_prompt(["regulation"], "", context=context),
        technological_prompt(["AI adoption"], context=context),
    ]

    for prompt in prompts:
        assert "Citation requirements:" in prompt
        assert "exact uploaded-file" in prompt
        assert "Do not assign citation numbers" in prompt
        assert "Original user request:" in prompt
        assert "[factsheet.pdf](/rag/files/factsheet.pdf)" in prompt
        assert "Adjusted EBITDA" in prompt


def test_technological_prompt_uses_technology_not_environmental_instructions() -> None:
    prompt = technological_prompt(["AI adoption and automation"])

    assert "AI and automation" in prompt
    assert "scholar_search" in prompt
    assert "water stress" not in prompt
    assert "world_bank_environmental_indicators" not in prompt
    assert "biodiversity" not in prompt


def test_agent_citations_follow_first_use_and_keep_first_duplicate_source() -> None:
    source_a = Source(name="Source A", url="https://example.com/a/")
    source_b = Source(name="Source B", url="https://example.com/b")
    duplicate_b = Source(name="Duplicate B", url="https://EXAMPLE.com/b/#section")
    unused = Source(name="Unused", url="https://example.com/unused")
    findings = (
        "B changed [Publisher B](https://example.com/b#article). "
        "A changed [Publisher A](https://example.com/a). "
        "B again [Publisher B](https://example.com/b/)."
    )

    normalized, sources = normalize_agent_citations(
        findings,
        [source_a, source_b, duplicate_b, unused],
    )

    assert normalized == (
        "B changed [1](https://example.com/b). "
        "A changed [2](https://example.com/a/). "
        "B again [1](https://example.com/b)."
    )
    assert sources == [source_b, source_a]


def test_unknown_links_are_plain_text_and_not_added_to_sources() -> None:
    source = Source(name="Known", url="https://example.com/known")

    normalized, sources = normalize_agent_citations(
        "Known [Known](https://example.com/known). "
        "Unknown [Invented](https://invented.example/story).",
        [source],
    )

    assert normalized == "Known [1](https://example.com/known). Unknown Invented."
    assert sources == [source]


def test_agent_can_cite_a_provided_uploaded_file_but_not_other_local_paths() -> None:
    uploaded = Source(name="Factsheet", url="/rag/files/EVONIK%20Q1.pdf")

    normalized, sources = normalize_agent_citations(
        "Company fact [Factsheet](/rag/files/EVONIK%20Q1.pdf). "
        "Unknown [Local file](/tmp/private.pdf).",
        [uploaded],
    )

    assert normalized == (
        "Company fact [1](/rag/files/EVONIK%20Q1.pdf). "
        "Unknown Local file."
    )
    assert sources == [uploaded]


def test_provided_sources_are_read_from_internal_agent_context() -> None:
    uploaded = Source(name="Factsheet", url="/rag/files/factsheet.pdf")

    assert provided_sources_from_context({"provided_sources": [uploaded]}) == [uploaded]
    assert provided_sources_from_context({"provided_sources": [uploaded.model_dump()]}) == [
        uploaded
    ]
    assert provided_sources_from_context({"provided_sources": ["invalid"]}) == []


def test_only_inline_cited_provided_sources_enter_agent_report_registry() -> None:
    used = Source(name="Used", url="/rag/files/used.pdf")
    unused = Source(name="Unused", url="/rag/files/unused.pdf")
    context = {"provided_sources": [used, unused]}

    assert cited_provided_sources(
        "Supported fact [Used](/rag/files/used.pdf).",
        context,
    ) == [used]


def test_no_body_citations_fall_back_to_all_url_sources() -> None:
    first = Source(name="First", url="https://example.com/first")
    duplicate = Source(name="First duplicate", url="https://example.com/first/")
    offline = Source(name="Offline")
    second = Source(name="Second", url="https://example.com/second")

    normalized, sources = normalize_agent_citations(
        "Analysis without a Markdown link.",
        [first, duplicate, offline, second],
    )

    assert normalized == "Analysis without a Markdown link."
    assert sources == [first, second]
