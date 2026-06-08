"""Shared data shapes used across all MAScan components."""

from mascan.contracts.reports import (
    AgentQualityReview,
    AgentReport,
    AgentRetryFeedback,
    FinalReport,
    Source,
)
from mascan.contracts.retrieval import RetrievalQuery, RetrievedChunk
from mascan.contracts.tools import ToolResult

__all__ = [
    "FinalReport",
    "AgentReport",
    "AgentQualityReview",
    "AgentRetryFeedback",
    "Source",
    "ToolResult",
    "RetrievedChunk",
    "RetrievalQuery",
]
