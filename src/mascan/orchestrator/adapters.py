from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from mascan.agents.base import BaseAgent
from mascan.contracts.reports import Source
from mascan.core.logging import get_logger
from mascan.orchestrator.state import GraphState

logger = get_logger("orchestrator.adapters")


def evidence_for_documents(
    rag_evidence: list[dict[str, Any]],
    documents: list[str],
) -> list[dict[str, Any]]:
    """Select passages whose exact document names were assigned by the planner."""
    allowed = {document.strip() for document in documents if document.strip()}
    if not allowed:
        return []
    return [
        evidence
        for evidence in rag_evidence
        if str((evidence.get("citation") or {}).get("document") or "") in allowed
    ]


def uploaded_document_sources(rag_evidence: list[dict[str, Any]]) -> list[Source]:
    """Group retrieved upload passages into exact, citable file sources."""
    grouped: dict[str, list[tuple[int | None, str]]] = {}
    for evidence in rag_evidence:
        citation = evidence.get("citation") or {}
        document = str(citation.get("document") or "uploaded document")
        page = citation.get("page") if isinstance(citation.get("page"), int) else None
        content = str(evidence.get("content") or "").strip()
        if not content:
            continue
        passage = (page, content)
        if passage not in grouped.setdefault(document, []):
            grouped[document].append(passage)

    sources: list[Source] = []
    for document, passages in grouped.items():
        pages = list(dict.fromkeys(page for page, _ in passages if page is not None))
        evidence_text = "\n\n".join(
            f"[Page {page}]\n{content}" if page is not None else content
            for page, content in passages
        )
        sources.append(
            Source(
                name=document,
                url=f"/rag/files/{quote(document, safe='')}",
                metadata={
                    "kind": "uploaded_document",
                    "citation": {"document": document, "pages": pages},
                    "content": evidence_text,
                },
            )
        )
    return sources


def make_agent_node(agent: BaseAgent) -> Callable[[GraphState], dict[str, Any]]:
    def node(state: GraphState) -> dict[str, Any]:
        assignment = state.plan.get(agent.name)
        if assignment is None or not assignment.tasks:
            logger.info("Agent %r has no tasks; skipping.", agent.name)
            return {}

        try:
            assigned_evidence = evidence_for_documents(
                state.rag_evidence,
                assignment.evidence_documents,
            )
            matched_documents = {
                str((evidence.get("citation") or {}).get("document") or "")
                for evidence in assigned_evidence
            }
            missing_documents = [
                document
                for document in assignment.evidence_documents
                if document not in matched_documents
            ]
            if missing_documents:
                logger.warning(
                    "Agent %r was assigned unavailable upload evidence: %s",
                    agent.name,
                    missing_documents,
                )
            provided_sources = uploaded_document_sources(assigned_evidence)
            report = agent.run(
                tasks=assignment.tasks,
                context={
                    "objective_context": assignment.objective_context,
                    "user_input": state.user_input,
                    "rag_evidence": assigned_evidence,
                    "provided_sources": provided_sources,
                    "salient_factors": assignment.salient_factors,
                    # Agents do not receive GraphState directly, so pass runtime metadata here.
                    "runtime": state.runtime_context.model_dump(),
                },
            )
            return {
                "reports": {agent.name: report},
                "component_metrics": report.component_metrics,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent %r failed", agent.name)
            return {"failures": {agent.name: f"{type(exc).__name__}: {exc}"}}

    node.__name__ = f"agent_{agent.name}"
    return node
