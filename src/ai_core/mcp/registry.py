"""In-memory registry for active Agents and MCP servers.

The registry is the rendezvous point for *dynamic discovery*: when an
Agent needs to call a tool that lives on an MCP server, it looks the
server up here, opens (or reuses) a connection, and invokes the tool.

Concurrency:
    All mutating operations are serialised by an :class:`asyncio.Lock`.
    Reads use a snapshot of the dict to avoid lock contention.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ai_core.di.interfaces import IComponent
from ai_core.exceptions import RegistryError

ComponentType = Literal["agent", "mcp_server"]


@dataclass(slots=True)
class RegisteredComponent:
    """Registry record describing one live component.

    Attributes:
        component: The actual object (Agent instance or MCP wrapper).
        component_type: Either ``"agent"`` or ``"mcp_server"``.
        registered_at: UTC timestamp at which the component was registered.
        metadata: Free-form metadata attached at registration time.
    """

    component: IComponent
    component_type: ComponentType
    registered_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    """Async-safe registry of agents and MCP servers."""

    def __init__(self) -> None:
        self._items: dict[str, RegisteredComponent] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        component: IComponent,
        *,
        component_type: ComponentType,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> RegisteredComponent:
        """Register a component under its ``component_id``.

        Args:
            component: Object exposing :class:`IComponent` attributes.
            component_type: One of ``"agent"`` or ``"mcp_server"``.
            metadata: Optional metadata to record alongside the component.
            replace: If ``True``, replace any existing component with the
                same id; otherwise raise :class:`RegistryError`.

        Returns:
            The :class:`RegisteredComponent` record.

        Raises:
            RegistryError: If ``replace=False`` and the id is already taken.
        """
        component_id = component.component_id
        async with self._lock:
            if component_id in self._items and not replace:
                raise RegistryError(
                    f"Component {component_id!r} is already registered",
                    details={"component_id": component_id, "component_type": component_type},
                )
            record = RegisteredComponent(
                component=component,
                component_type=component_type,
                registered_at=datetime.now(UTC),
                metadata=dict(metadata or {}),
            )
            self._items[component_id] = record
            return record

    async def unregister(self, component_id: str) -> bool:
        """Remove a component from the registry.

        Args:
            component_id: Identifier passed to :meth:`register`.

        Returns:
            ``True`` if a record was removed, ``False`` if absent.
        """
        async with self._lock:
            return self._items.pop(component_id, None) is not None

    def get(self, component_id: str) -> RegisteredComponent:
        """Return the record for ``component_id``.

        Args:
            component_id: Identifier passed to :meth:`register`.

        Returns:
            The :class:`RegisteredComponent` record.

        Raises:
            RegistryError: If no component is registered under that id.
        """
        record = self._items.get(component_id)
        if record is None:
            raise RegistryError(
                f"No component registered under {component_id!r}",
                details={"component_id": component_id},
            )
        return record

    def list(
        self,
        *,
        component_type: ComponentType | None = None,
    ) -> Sequence[RegisteredComponent]:
        """Return a snapshot of registered components, optionally filtered."""
        snapshot = list(self._items.values())
        if component_type is None:
            return snapshot
        return [r for r in snapshot if r.component_type == component_type]

    def __contains__(self, component_id: object) -> bool:
        return isinstance(component_id, str) and component_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    async def health_check_all(self) -> dict[str, bool]:
        """Concurrently call ``health_check()`` on every registered component.

        Returns:
            Mapping of ``component_id`` to ``True``/``False``. A ``False``
            value means either the component reported unhealthy or its
            ``health_check()`` raised — exceptions are swallowed.
        """
        snapshot = list(self._items.items())

        async def _check(cid: str, rec: RegisteredComponent) -> tuple[str, bool]:
            try:
                healthy = bool(await rec.component.health_check())
            except Exception:  # noqa: BLE001 — registry must not crash on a bad component
                healthy = False
            return cid, healthy

        results = await asyncio.gather(*(_check(cid, rec) for cid, rec in snapshot))
        return dict(results)


__all__ = ["ComponentRegistry", "RegisteredComponent", "ComponentType"]
