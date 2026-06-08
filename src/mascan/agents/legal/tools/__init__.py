"""Legal-agent tools."""

from mascan.agents.legal.tools.eur_lex import EurLexTool
from mascan.agents.legal.tools.federal_register import FederalRegisterTool
from mascan.tools.registry import tool_registry

tool_registry.register(FederalRegisterTool())
tool_registry.register(EurLexTool())

__all__ = ["EurLexTool", "FederalRegisterTool"]
