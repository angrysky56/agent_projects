"""
Unit tests for SLAP Pipeline representation flexibility.
"""

import pytest
from config import SLAPConfig
from slap_pipeline import SLAPPipeline


@pytest.fixture
def pipeline():
    config = SLAPConfig()
    return SLAPPipeline(config)


def test_hierarchical_plan_generation(pipeline):
    task = "Design a system architecture"
    objectives = []

    plan = pipeline.create_reasoning_plan(task, objectives, representation_type="hierarchical")

    assert plan["type"] == "hierarchical"
    assert "decomposition" in plan
    assert plan["model"]["structure"] == "hierarchical"
    assert "subtasks" in plan["decomposition"]


def test_network_plan_generation(pipeline):
    task = "Map relationships"
    objectives = []

    plan = pipeline.create_reasoning_plan(task, objectives, representation_type="network")

    assert plan["type"] == "network"
    assert "connectivity" in plan
    assert plan["model"]["structure"] == "network"
    assert "hubs" in plan["connectivity"]


def test_causal_plan_generation(pipeline):
    task = "Analyze root cause"
    objectives = []

    plan = pipeline.create_reasoning_plan(task, objectives, representation_type="causal")

    assert plan["type"] == "causal"
    assert "causal_chain" in plan
    assert plan["model"]["structure"] == "causal"
    assert "root_cause" in plan["causal_chain"]


def test_sequential_plan_generation_default(pipeline):
    task = "Simple task"
    objectives = []

    plan = pipeline.create_reasoning_plan(task, objectives)

    assert plan.get("type") == "sequential"
    assert "derivation" in plan
    assert plan["model"]["structure"] == "sequential"
