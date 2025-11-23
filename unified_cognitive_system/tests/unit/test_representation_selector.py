"""
Unit tests for Representation Selector module.
"""

import pytest
from config import DynamicWorkspaceConfig
from representation_selector import RepresentationSelector
from utils import RepresentationState


@pytest.fixture
def selector():
    config = DynamicWorkspaceConfig(default_representation="sequential", allowed_representations=["sequential", "hierarchical", "network", "causal", "spatial", "temporal"], enable_adaptive_switching=True)
    return RepresentationSelector(config)


def test_initialization(selector):
    assert selector.config.default_representation == "sequential"


def test_hierarchical_detection(selector):
    task = "Design a system architecture for a new app"
    state = selector.select_representation(task)

    assert state.current_type == "hierarchical"
    assert "decomposition" in state.reason_for_selection.lower() or "structural" in state.reason_for_selection.lower()


def test_network_detection(selector):
    task = "Map the relationships between different social groups"
    state = selector.select_representation(task)

    assert state.current_type == "network"
    assert "relationships" in state.reason_for_selection.lower()


def test_causal_detection(selector):
    task = "Analyze the root cause of the system failure"
    state = selector.select_representation(task)

    assert state.current_type == "causal"
    assert "causal" in state.reason_for_selection.lower()


def test_default_fallback(selector):
    task = "Just do something simple"
    state = selector.select_representation(task)

    assert state.current_type == "sequential"
    assert "Default" in state.reason_for_selection


def test_adaptive_switching(selector):
    # Test switching from sequential to hierarchical if stuck
    current_state = RepresentationState(current_type="sequential", confidence=0.5, history=["sequential"] * 6, reason_for_selection="Default")

    new_type = selector.should_switch_representation(current_state, progress=0.1, step_data={})
    assert new_type == "hierarchical"

    # Test switching to network if content suggests relationships
    current_state = RepresentationState(current_type="sequential", confidence=0.5, history=["sequential"], reason_for_selection="Default")

    new_type = selector.should_switch_representation(current_state, progress=0.5, step_data={"reasoning": "This connects to that via a strong relationship"})
    assert new_type == "network"
