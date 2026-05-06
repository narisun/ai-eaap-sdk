"""Tests for ai_core.testing.ScriptedLLM."""
from __future__ import annotations

import pytest

from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_scripted_llm_returns_responses_in_order() -> None:
    r1 = make_llm_response("first")
    r2 = make_llm_response("second")
    llm = ScriptedLLM([r1, r2])
    out1 = await llm.complete(model=None, messages=[])
    out2 = await llm.complete(model=None, messages=[])
    assert out1.content == "first"
    assert out2.content == "second"


@pytest.mark.asyncio
async def test_scripted_llm_records_calls() -> None:
    llm = ScriptedLLM([make_llm_response("a"), make_llm_response("b")])
    await llm.complete(model="x", messages=[{"role": "user", "content": "1"}])
    await llm.complete(
        model="y",
        messages=[{"role": "user", "content": "2"}],
        tenant_id="t1",
    )
    assert len(llm.calls) == 2
    assert llm.calls[0]["model"] == "x"
    assert llm.calls[1]["tenant_id"] == "t1"


@pytest.mark.asyncio
async def test_scripted_llm_raises_index_error_on_exhaustion() -> None:
    llm = ScriptedLLM([make_llm_response("only")])
    await llm.complete(model=None, messages=[])
    with pytest.raises(IndexError, match="ScriptedLLM exhausted"):
        await llm.complete(model=None, messages=[])


@pytest.mark.asyncio
async def test_scripted_llm_repeat_last_keeps_returning_final_response() -> None:
    r1 = make_llm_response("first")
    r2 = make_llm_response("last")
    llm = ScriptedLLM([r1, r2], repeat_last=True)
    out1 = await llm.complete(model=None, messages=[])
    out2 = await llm.complete(model=None, messages=[])
    out3 = await llm.complete(model=None, messages=[])
    out4 = await llm.complete(model=None, messages=[])
    assert out1.content == "first"
    assert out2.content == "last"
    assert out3.content == "last"
    assert out4.content == "last"


def test_scripted_llm_init_rejects_empty_responses() -> None:
    with pytest.raises(ValueError, match="at least one response"):
        ScriptedLLM([])


def test_scripted_llm_satisfies_illmclient_protocol() -> None:
    llm = ScriptedLLM([make_llm_response()])
    assert isinstance(llm, ILLMClient)
