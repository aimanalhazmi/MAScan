"""Shared data shapes used across all MAScan components."""

from mascan.contracts.reports import AgentReport, Source, FinalReport
from mascan.contracts.retrieval import RetrievalQuery, RetrievedChunk
from mascan.contracts.tools import ToolResult

__all__ = [
    "FinalReport",
    "AgentReport",
    "Source",
    "ToolResult",
    "RetrievedChunk",
    "RetrievalQuery",
]