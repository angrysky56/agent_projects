"""
Integration tests for CGRA-enhanced COMPASS framework.
"""

import pytest
from unittest.mock import MagicMock, patch
from compass_framework import COMPASS
from config import COMPASSConfig


@pytest.fixture
def compass_framework():
    config = COMPASSConfig()
    # Use minimal resources for testing
    config.omcd.max_resources = 100.0
    config.self_discover.max_trials = 2
    return COMPASS(config)


def test_full_cgra_flow(compass_framework):
    """Test the complete CGRA flow with mocked sub-components."""

    # Mock components to avoid heavy computation and external calls
    compass_framework._shape_processor = MagicMock()
    compass_framework._shape_processor.process_user_input.return_value = {"input": "test task"}
    compass_framework._shape_processor.expand_shorthand.return_value = "expanded test task"
    compass_framework._shape_processor.map_semantics.return_value = {"enriched_prompt": "enriched test task"}

    compass_framework._smart_planner = MagicMock()
    compass_framework._smart_planner.create_objectives_from_task.return_value = []

    compass_framework._slap_pipeline = MagicMock()
    compass_framework._slap_pipeline.create_reasoning_plan.return_value = {"facts": []}

    compass_framework._intelligence_core = MagicMock()
    compass_framework._intelligence_core.make_decision.return_value = "test decision"

    # Mock CGRA components
    compass_framework._constraint_governor = MagicMock()
    compass_framework._constraint_governor.validate_reasoning_step.return_value = True

    compass_framework._executive_controller = MagicMock()
    compass_framework._executive_controller.coordinate_iteration.return_value = {"goal": None, "strategy": [1, 2], "resources": {"amount": 10.0}, "should_stop": False}
    compass_framework._executive_controller.evaluate_reasoning_quality.return_value = 0.8

    compass_framework._representation_selector = MagicMock()
    compass_framework._representation_selector.select_representation.return_value = MagicMock(current_type="sequential")

    compass_framework._procedural_toolkit = MagicMock()

    # Run process_task
    result = compass_framework.process_task("Test Task")

    # Verify flow
    assert result["success"] is True
    assert result["score"] == 0.8

    # Verify calls
    compass_framework.constraint_governor.validate_reasoning_step.assert_called()
    compass_framework.executive_controller.coordinate_iteration.assert_called()
    compass_framework.representation_selector.select_representation.assert_called()
    compass_framework.slap_pipeline.create_reasoning_plan.assert_called()
    compass_framework.executive_controller.evaluate_reasoning_quality.assert_called()


def test_cgra_causal_branch(compass_framework):
    """Test that causal representation triggers procedural toolkit."""

    # Mock setup similar to above
    compass_framework._shape_processor = MagicMock()
    compass_framework._shape_processor.map_semantics.return_value = {"enriched_prompt": "test task"}
    compass_framework._smart_planner = MagicMock()
    compass_framework._smart_planner.create_objectives_from_task.return_value = []
    compass_framework._slap_pipeline = MagicMock()
    compass_framework._slap_pipeline.create_reasoning_plan.return_value = {"facts": []}
    compass_framework._intelligence_core = MagicMock()
    compass_framework._intelligence_core.make_decision.return_value = "decision"

    compass_framework._constraint_governor = MagicMock()
    compass_framework._constraint_governor.validate_reasoning_step.return_value = True

    compass_framework._executive_controller = MagicMock()
    compass_framework._executive_controller.coordinate_iteration.return_value = {"goal": None, "strategy": [], "resources": {"amount": 10}, "should_stop": True}
    compass_framework._executive_controller.evaluate_reasoning_quality.return_value = 0.5

    # Mock Representation Selector to return 'causal'
    compass_framework._representation_selector = MagicMock()
    compass_framework._representation_selector.select_representation.return_value = MagicMock(current_type="causal")

    compass_framework._procedural_toolkit = MagicMock()

    # Run
    compass_framework.process_task("Causal Task")

    # Verify procedural toolkit was called
    compass_framework.procedural_toolkit.backward_chaining.assert_called()
