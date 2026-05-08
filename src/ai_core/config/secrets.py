"""Secret resolution abstraction.

The SDK never embeds raw secret material in code or logs. Instead it
treats secrets as opaque references resolved through an
:class:`ISecretManager` implementation bound by the DI container.

Two resolution paths are supported out of the box:

* **Inline** values — already present in :class:`AppSettings` as
  :class:`pydantic.SecretStr`. These are returned as-is.
* **Reference** values — :class:`SecretRef` markers that name an external
  backend (env, AWS Secrets Manager, GCP Secret Manager, Vault, …).

Concrete cloud-backed providers live in :mod:`ai_core.persistence` /
extras and are bound to :class:`ISecretManager` by the application's
DI module.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ai_core.exceptions import SecretResolutionError


@dataclass(frozen=True, slots=True)
class SecretRef:
    """A pointer to a secret stored in an external backend.

    Attributes:
        backend: Identifier of the backend (e.g. ``"env"``, ``"aws-sm"``).
        name: Backend-specific identifier of the secret (env var, ARN, path).
        version: Optional version tag (e.g. AWS Secrets Manager VersionId).
    """

    backend: str
    name: str
    version: str | None = None

    def __str__(self) -> str:
        return f"secret://{self.backend}/{self.name}{f'@{self.version}' if self.version else ''}"


class ISecretManager(ABC):
    """Resolve :class:`SecretRef` instances to concrete secret values.

    Implementations MUST:

    * be safe to call concurrently from multiple coroutines;
    * raise :class:`ai_core.exceptions.SecretResolutionError` (not the
      backend's native exception) on any failure;
    * never log the resolved value.
    """

    @abstractmethod
    async def resolve(self, ref: SecretRef) -> str:
        """Return the plaintext secret value referenced by ``ref``.

        Args:
            ref: The secret reference to dereference.

        Returns:
            The plaintext secret value.

        Raises:
            SecretResolutionError: If the backend fails or the secret is missing.
        """

    async def resolve_optional(self, ref: SecretRef | None) -> str | None:
        """Convenience wrapper that returns ``None`` for a ``None`` reference.

        Args:
            ref: A reference or ``None``.

        Returns:
            The resolved value, or ``None`` if ``ref`` is ``None``.
        """
        if ref is None:
            return None
        return await self.resolve(ref)


class EnvSecretManager(ISecretManager):
    """Default :class:`ISecretManager` that reads from process environment.

    Recognises only the ``"env"`` backend. Suitable for local development
    and small deployments where secrets are sourced from a sealed
    environment (e.g. Kubernetes secrets injected as env vars).
    """

    BACKEND: str = "env"

    async def resolve(self, ref: SecretRef) -> str:
        """Look ``ref.name`` up in :data:`os.environ`.

        Args:
            ref: The secret reference; ``backend`` must be ``"env"``.

        Returns:
            The environment variable's value.

        Raises:
            SecretResolutionError: If ``ref.backend`` is not ``"env"`` or the
                env var is unset.
        """
        if ref.backend != self.BACKEND:
            raise SecretResolutionError(
                f"EnvSecretManager cannot resolve backend {ref.backend!r}",
                details={"backend": ref.backend, "name": ref.name},
            )
        try:
            return os.environ[ref.name]
        except KeyError as exc:
            raise SecretResolutionError(
                f"Environment variable {ref.name!r} is not set",
                details={"backend": ref.backend, "name": ref.name},
                cause=exc,
            ) from exc


__all__ = ["EnvSecretManager", "ISecretManager", "SecretRef"]
