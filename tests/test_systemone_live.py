"""Opt-in synthetic protocol probes; no personal or application data."""

import os

import pytest

import litelm

pytestmark = pytest.mark.live

QUESTIONS = {
    "correct": {"type": "noul", "instructions": "Ist zwei plus zwei gleich vier?"},
    "category": {
        "type": "choice",
        "instructions": "Welche Kategorie beschreibt den Text?",
        "criteria": {"arithmetic": "Rechenaufgabe", "weather": "Wetterbericht"},
    },
    "clarity": {
        "type": "score",
        "instructions": "Wie verständlich ist der Text?",
        "criteria": ["Unverständlich", "Teilweise verständlich", "Verständlich"],
    },
}


@pytest.mark.parametrize("provider", ["typesafe", "azure_ai"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_live_synthetic_systemone(provider, async_call):
    if not os.getenv(f"{provider.upper()}_API_KEY"):
        pytest.skip("Native decision provider credential unavailable")
    if provider == "azure_ai":
        deployment = os.getenv("AZURE_AI_DECISION_DEPLOYMENT")
        if not deployment or not os.getenv("AZURE_AI_API_BASE"):
            pytest.skip("Foundry decision deployment configuration unavailable")
    else:
        deployment = os.getenv("TYPESAFE_DECISION_MODEL", "jev-latest")
    params = dict(model=f"{provider}/{deployment}", state="Zwei plus zwei ist vier.", questions=QUESTIONS)
    response = await litelm.asystemone(**params) if async_call else litelm.systemone(**params)
    assert response.answers["correct"].noul > 0.5
    assert response.answers["category"].choice == "arithmetic"
    assert 0 <= response.answers["clarity"].score <= 2
    assert litelm.SystemOneResponse.model_validate_json(response.model_dump_json()) == response
