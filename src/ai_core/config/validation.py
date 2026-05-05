"""Runtime configuration validation helpers.

These primitives back :meth:`ai_core.config.settings.AppSettings.validate_for_runtime`.
They accumulate every problem before raising so a developer can fix the entire
configuration in one pass instead of one error at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """A single runtime-configuration problem.

    Attributes:
        path: Dotted path inside :class:`AppSettings`, e.g. ``"llm.default_model"``.
        message: Short human-readable description of what is wrong.
        hint: Optional, actionable instruction for the developer (env var name,
            override pattern, doc link, …). ``None`` when there is no useful hint.
    """

    path: str
    message: str
    hint: str | None = None


@dataclass(slots=True)
class ValidationContext:
    """Accumulator passed through individual validators.

    Validators call :py:meth:`fail` to record problems. The context is consumed
    by :meth:`AppSettings.validate_for_runtime`, which raises
    :class:`ai_core.exceptions.ConfigurationError` when :py:attr:`has_issues`
    is ``True``.
    """

    issues: list[ConfigIssue] = field(default_factory=list)

    def fail(self, path: str, message: str, hint: str | None = None) -> None:
        """Record a single problem; never raises."""
        self.issues.append(ConfigIssue(path=path, message=message, hint=hint))

    @property
    def has_issues(self) -> bool:
        """True when at least one validator has called :py:meth:`fail`."""
        return bool(self.issues)


__all__ = ["ConfigIssue", "ValidationContext"]
