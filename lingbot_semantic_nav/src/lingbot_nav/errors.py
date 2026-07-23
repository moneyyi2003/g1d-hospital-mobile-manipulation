"""Domain-specific errors with messages suitable for robot status output."""


class SemanticNavError(Exception):
    """Base class for expected navigation-pipeline failures."""


class ConfigurationError(SemanticNavError):
    """Configuration is missing or invalid."""


class IntentParseError(SemanticNavError):
    """Natural-language intent could not be parsed safely."""


class UnknownPlaceError(SemanticNavError):
    """The requested place does not exist in the semantic place database."""


class AmbiguousPlaceError(SemanticNavError):
    """The request matches multiple places too closely."""


class NoPathError(SemanticNavError):
    """No collision-free path exists between start and goal."""


class ProviderError(SemanticNavError):
    """An external LLM provider call failed or returned invalid data."""

