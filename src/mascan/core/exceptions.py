"""Custom exception types used across MAScan."""


class MAScanError(Exception):
    """Base exception for all MAScan errors."""


class ConfigError(MAScanError):
    """Raised when configuration is invalid or missing."""


class AgentError(MAScanError):
    """Raised when an agent fails during execution."""


class ToolError(MAScanError):
    """Raised when a tool fails during execution."""


class RegistryError(MAScanError):
    """Raised when registry lookup or registration fails."""
