"""Scholar Search Tool"""

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool

class ScholarSearchInput(BaseModel):
    query: str = Field(..., description="Search query for academic papers and scholarly articles.")
    mode: str = Field("default", description="Search mode, e.g., 'default', 'advanced'.")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of results to return.")

class ScholarSearchTool(BaseTool):
    name = "scholar_search"
    description = (
        "Search for academic papers and scholarly articles."
    )