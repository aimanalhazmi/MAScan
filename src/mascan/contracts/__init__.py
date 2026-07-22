"""Shared data shapes used across all MAScan components."""

from mascan.contracts.planning import AgentAssignment
from mascan.contracts.reports import AgentReport, FinalReport, Source
from mascan.contracts.retrieval import RetrievalQuery, RetrievedChunk
from mascan.contracts.tools import ToolResult
from mascan.contracts.validation import ValidationReport

__all__ = [
    "FinalReport",
    "AgentAssignment",
    "AgentReport",
    "Source",
    "ToolResult",
    "RetrievedChunk",
    "RetrievalQuery",
    "ValidationReport",
]
