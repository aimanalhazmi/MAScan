from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool
from mascan.tools.http_client import http_get

# This tool is planned but not yet implemented.
# Turns out that the EPO OPS API requires a verified account, which takes time to set up.
# And USPTO API requires ID verification.

class PatentSearchTool(BaseTool):
    name = "patent_search"
    description = (
        "Fetch relevant patents"
    )