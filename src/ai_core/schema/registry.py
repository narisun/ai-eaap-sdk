"""Versioned Pydantic schema registry.

Use case
--------
Agent-to-Agent and Agent-to-MCP communication is JSON-over-something.
Without a versioned contract, a tool author can rename a field and
silently break every caller. The registry provides:

* a single source of truth keyed by ``(name, version)`` for both the
  *input* and *output* schemas of a tool;
* a :py:meth:`SchemaRegistry.validate_tool` decorator that wraps a
  sync or async callable so its arguments are parsed against the
  registered input schema and its return value validated against the
  registered output schema;
* a :py:meth:`SchemaRegistry.latest_version` lookup so callers can
  resolve to the newest contract on registration order.

Versions are positive integers. Hosts that prefer semver should encode
``MAJOR`` as the registered version and treat MINOR/PATCH as
backwards-compatible refinements that don't require a new entry.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from ai_core.exceptions import SchemaValidationError

SchemaVersion = int

_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(slots=True, frozen=True)
class SchemaRecord:
    """Immutable record describing one registered (name, version) entry.

    Attributes:
        name: Tool / message identifier.
        version: Positive integer schema version.
        input_schema: Pydantic model parsed from incoming payloads.
        output_schema: Pydantic model returned values are validated against.
        description: Optional human-readable description.
    """

    name: str
    version: SchemaVersion
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    description: str = ""


class SchemaRegistry:
    """In-memory registry of versioned tool/message schemas.

    The registry is safe to share across coroutines: registration is
    serialised by a lock; lookups and decoration take a snapshot of the
    underlying mapping and don't acquire the lock.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, SchemaVersion], SchemaRecord] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        version: SchemaVersion,
        *,
        input_schema: type[BaseModel],
        output_schema: type[BaseModel],
        description: str = "",
        replace: bool = False,
    ) -> SchemaRecord:
        """Register an input/output schema pair under ``(name, version)``.

        Args:
            name: Tool / message identifier.
            version: Positive integer schema version (must be ``>= 1``).
            input_schema: Pydantic model used for input validation.
            output_schema: Pydantic model used for output validation.
            description: Optional human-readable description.
            replace: If ``True`` overwrites an existing registration.

        Returns:
            The created :class:`SchemaRecord`.

        Raises:
            SchemaValidationError: On bad arguments or duplicate registration.
        """
        if not name:
            raise SchemaValidationError("Schema name must be non-empty")
        if version < 1:
            raise SchemaValidationError(
                "Schema version must be >= 1",
                details={"name": name, "version": version},
            )
        if not (isinstance(input_schema, type) and issubclass(input_schema, BaseModel)):
            raise SchemaValidationError(
                "input_schema must be a Pydantic BaseModel subclass",
                details={"name": name, "version": version},
            )
        if not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
            raise SchemaValidationError(
                "output_schema must be a Pydantic BaseModel subclass",
                details={"name": name, "version": version},
            )

        key = (name, version)
        if key in self._records and not replace:
            raise SchemaValidationError(
                f"Schema {name!r} v{version} is already registered",
                details={"name": name, "version": version},
            )
        record = SchemaRecord(
            name=name,
            version=version,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
        )
        self._records[key] = record
        return record

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get(self, name: str, version: SchemaVersion | None = None) -> SchemaRecord:
        """Return the record for ``(name, version)``, or the latest if ``version`` is None.

        Args:
            name: Tool / message identifier.
            version: Specific version, or ``None`` for the latest.

        Returns:
            The matching :class:`SchemaRecord`.

        Raises:
            SchemaValidationError: If no matching record exists.
        """
        if version is None:
            latest = self.latest_version(name)
            if latest is None:
                raise SchemaValidationError(
                    f"No schema registered under {name!r}",
                    details={"name": name},
                )
            version = latest
        try:
            return self._records[(name, version)]
        except KeyError as exc:
            raise SchemaValidationError(
                f"No schema {name!r} v{version} registered",
                details={"name": name, "version": version},
            ) from exc

    def latest_version(self, name: str) -> SchemaVersion | None:
        """Return the highest version registered for ``name``, or ``None``."""
        versions = [v for (n, v) in self._records if n == name]
        return max(versions) if versions else None

    def versions(self, name: str) -> list[SchemaVersion]:
        """Return every registered version for ``name`` in ascending order."""
        return sorted(v for (n, v) in self._records if n == name)

    def names(self) -> list[str]:
        """Return every registered tool/message name (sorted, unique)."""
        return sorted({n for (n, _) in self._records})

    def iter_records(self) -> list[SchemaRecord]:
        """Return every registered :class:`SchemaRecord` sorted by ``(name, version)``.

        A list (not an iterator) is returned so callers can iterate
        multiple times without re-acquiring a snapshot. Ordering is
        deterministic across runs — useful for documentation and
        diff-friendly schema exports.
        """
        return [self._records[k] for k in sorted(self._records.keys())]

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            return cast(tuple, key) in self._records
        return False

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Validation decorator
    # ------------------------------------------------------------------
    def validate_tool(
        self,
        name: str,
        *,
        version: SchemaVersion | None = None,
    ) -> Callable[[_F], _F]:
        """Decorator that enforces the registered schemas around a tool function.

        The wrapped function is invoked with **a single positional argument** —
        the parsed input model instance. The function's return value is
        validated against the output schema and returned to the caller.

        Both sync and async callables are supported. The decorator preserves
        the wrapped function's signature for documentation purposes.

        Args:
            name: Registered tool name.
            version: Specific version; defaults to the latest registered.

        Returns:
            A decorator that wraps the target callable.

        Raises:
            SchemaValidationError: At registration time if the schema is
                missing; at call time if input parsing or output validation
                fails.

        Example::

            class CreateTicketIn(BaseModel):
                title: str
                priority: Literal["low", "high"] = "low"

            class CreateTicketOut(BaseModel):
                ticket_id: str

            registry.register("create_ticket", 1,
                              input_schema=CreateTicketIn,
                              output_schema=CreateTicketOut)

            @registry.validate_tool("create_ticket", version=1)
            async def create_ticket(payload: CreateTicketIn) -> CreateTicketOut:
                ...
        """
        record = self.get(name, version)

        def _decorator(func: _F) -> _F:
            if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
                @functools.wraps(func)
                async def _async_wrapper(payload: Any, /, *args: Any, **kwargs: Any) -> Any:
                    parsed = _parse_input(record, payload)
                    raw_result = await cast(
                        Callable[..., Awaitable[Any]], func
                    )(parsed, *args, **kwargs)
                    return _validate_output(record, raw_result)

                return cast(_F, _async_wrapper)

            @functools.wraps(func)
            def _sync_wrapper(payload: Any, /, *args: Any, **kwargs: Any) -> Any:
                parsed = _parse_input(record, payload)
                raw_result = func(parsed, *args, **kwargs)
                return _validate_output(record, raw_result)

            return cast(_F, _sync_wrapper)

        return _decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_input(record: SchemaRecord, payload: Any) -> BaseModel:
    if isinstance(payload, record.input_schema):
        return payload
    try:
        if isinstance(payload, BaseModel):
            return record.input_schema.model_validate(payload.model_dump())
        return record.input_schema.model_validate(payload)
    except ValidationError as exc:
        raise SchemaValidationError(
            f"Invalid input for {record.name!r} v{record.version}",
            details={
                "name": record.name,
                "version": record.version,
                "errors": exc.errors(),
            },
            cause=exc,
        ) from exc


def _validate_output(record: SchemaRecord, value: Any) -> BaseModel:
    if isinstance(value, record.output_schema):
        return value
    try:
        if isinstance(value, BaseModel):
            return record.output_schema.model_validate(value.model_dump())
        return record.output_schema.model_validate(value)
    except ValidationError as exc:
        raise SchemaValidationError(
            f"Tool {record.name!r} v{record.version} returned invalid output",
            details={
                "name": record.name,
                "version": record.version,
                "errors": exc.errors(),
            },
            cause=exc,
        ) from exc


__all__ = ["SchemaRegistry", "SchemaRecord", "SchemaVersion"]
