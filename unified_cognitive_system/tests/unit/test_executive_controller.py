"""
Unit tests for Executive Controller module.
"""

import pytest
from config import ExecutiveControllerConfig, oMCDConfig, SelfDiscoverConfig
from executive_controller import ExecutiveController
from utils import Goal


@pytest.fixture
def controller():
    exec_config = ExecutiveControllerConfig()
    omcd_config = oMCDConfig()
    sd_config = SelfDiscoverConfig()

    return ExecutiveController(exec_config, omcd_config, sd_config)


def test_initialization(controller):
    assert controller.omcd is not None
    assert controller.self_discover is not None
    assert controller.iteration_count == 0


def test_coordinate_iteration_initial(controller):
    # Test first iteration behavior
    task = "Solve a complex problem"
    state = {"score": 0.0}

    decision = controller.coordinate_iteration(task, state)

    # Check goal establishment
    assert decision["goal"] is not None
    assert "Solve task" in decision["goal"].description

    # Check strategy selection
    assert len(decision["strategy"]) > 0

    # Check resource allocation
    assert decision["resources"] is not None

    # Check solvability assessment
    assert "solvability_score" in decision["solvability"]


def test_goal_management_integration(controller):
    # Test manual goal establishment via controller's omcd instance
    goal = controller.omcd.establish_goal("Manual Goal", priority=0.8)
    assert goal.id in [g.id for g in controller.omcd.goal_stack]

    # Run iteration
    decision = controller.coordinate_iteration("Task", {})

    # Should use the manual goal as current
    assert decision["goal"].id == goal.id

    # Update status
    controller.update_goal_status(goal.id, 50.0)
    assert controller.omcd.get_current_goal().progress == 50.0


def test_context_awareness_integration(controller):
    # Test context passing
    context = {"complexity": "high", "domain": "creative"}
    task = "Design a new system"

    decision = controller.coordinate_iteration(task, {}, context=context)

    # Strategy should be influenced (hard to test exact modules without mocking random,
    # but we can check that context was stored)
    assert controller.context == context


def test_reset(controller):
    controller.coordinate_iteration("Task", {})
    assert controller.iteration_count == 1

    controller.reset()
    assert controller.iteration_count == 0
    assert len(controller.omcd.goal_stack) == 0
