"""
Unit tests for Procedural Toolkit module.
"""

import pytest
from config import ProceduralToolkitConfig
from procedural_toolkit import ProceduralToolkit


@pytest.fixture
def toolkit():
    config = ProceduralToolkitConfig()
    return ProceduralToolkit(config)


def test_backward_chaining(toolkit):
    goal = "Launch Rocket"
    facts = []

    steps = toolkit.backward_chaining(goal, facts)

    assert len(steps) > 0
    assert "Verify prerequisites" in steps[0]
    assert "Validate Launch Rocket" in steps[-1]


def test_backtracking(toolkit):
    history = [{"action": "Step 1", "state": "ok"}, {"action": "Step 2", "state": "ok"}, {"action": "Step 3", "state": "failed"}]

    result = toolkit.backtracking(history, failure_point=2)

    assert result["status"] == "success"
    assert result["recovery_index"] == 1
    assert result["recovered_state"]["action"] == "Step 2"
    assert "Step 3" in result["recommendation"]


def test_backtracking_invalid(toolkit):
    result = toolkit.backtracking([], 0)
    assert result["status"] == "failed"


def test_counterfactual_reasoning(toolkit):
    scenario = "Engine failed due to overheating"
    condition = "Cooling system was active"

    outcome = toolkit.counterfactual_reasoning(scenario, condition)

    assert "If 'Cooling system was active' were true" in outcome
    assert "outcome would likely shift" in outcome


def test_analogical_mapping(toolkit):
    mapping = toolkit.analogical_mapping("Software", "Construction")

    assert "structure" in mapping
    assert mapping["structure"] == "architecture"
