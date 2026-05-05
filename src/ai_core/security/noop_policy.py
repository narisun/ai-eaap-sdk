"""Always-allow policy evaluator for development without OPA running.

Production deployments MUST override the default DI binding with a real
evaluator (e.g. :class:`OPAPolicyEvaluator`) via
:class:`ai_core.di.module.ProductionSecurityModule`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from ai_core.di.interfaces import IPolicyEvaluator, PolicyDecision


class NoOpPolicyEvaluator(IPolicyEvaluator):
    """Policy evaluator that returns ``PolicyDecision(allowed=True)`` for every call.

    Use this in local development environments where standing up OPA is overkill.
    The reason field is set to a recognisable string so audit trails show that no
    real policy evaluation occurred.
    """

    async def evaluate(
        self, *, decision_path: str, input: Mapping[str, Any]
    ) -> PolicyDecision:
        """Always allow."""
        return PolicyDecision(
            allowed=True,
            obligations={},
            reason="no-op evaluator (development only)",
        )


__all__ = ["NoOpPolicyEvaluator"]
