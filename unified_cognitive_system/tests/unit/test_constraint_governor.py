"""
Unit tests for Constraint Governor module.
"""

import pytest
from config import ConstraintGovernorConfig
from constraint_governor import ConstraintGovernor, ConstraintViolation


@pytest.fixture
def governor():
    config = ConstraintGovernorConfig(enable_validation=True, validate_logical_coherence=True, validate_compositionality=True, validate_productivity=True, validate_conceptual_processing=True, max_contradictions_allowed=1)
    return ConstraintGovernor(config)


def test_initialization(governor):
    assert governor.config.enable_validation is True
    assert len(governor.violations) == 0


def test_logical_coherence_check(governor):
    # Test contradiction detection
    step_data = {"reasoning": "The sky is blue. However, this contradicts my earlier statement that the sky is green.", "action": ""}
    violations = governor.validate_step(step_data, {})

    assert len(violations) == 1
    assert violations[0].invariant_type == "logical_coherence"
    assert "contradiction detected" in violations[0].description.lower()


def test_compositionality_check(governor):
    # Test valid decomposition
    valid_step = {"action_type": "decompose", "subtasks": ["task1", "task2"]}
    violations = governor.validate_step(valid_step, {})
    assert len(violations) == 0

    # Test invalid decomposition (missing subtasks)
    invalid_step = {"action_type": "decompose", "subtasks": []}
    violations = governor.validate_step(invalid_step, {})

    assert len(violations) == 1
    assert violations[0].invariant_type == "compositionality"
    assert "missing subtasks" in violations[0].description


def test_productivity_check(governor):
    # Test loop detection
    step1 = {"reasoning": "Thinking about X"}
    governor.validate_step(step1, {})

    step2 = {"reasoning": "Thinking about Y"}
    governor.validate_step(step2, {})

    # Identical to step 2 - should trigger productivity warning
    step3 = {"reasoning": "Thinking about Y"}
    violations = governor.validate_step(step3, {})

    assert len(violations) == 1
    assert violations[0].invariant_type == "productivity"
    assert "repetitive reasoning" in violations[0].description.lower()


def test_conceptual_processing_check(governor):
    # Test shallow processing detection
    # We need 3 steps to trigger the heuristic

    step1 = {"reasoning": "word word word", "conceptual_depth": 0.1}
    governor.validate_step(step1, {})

    step2 = {"reasoning": "text text text", "conceptual_depth": 0.1}
    governor.validate_step(step2, {})

    step3 = {"reasoning": "phrase phrase phrase", "conceptual_depth": 0.1}
    violations = governor.validate_step(step3, {})

    assert len(violations) == 1
    assert violations[0].invariant_type == "conceptual_processing"
    assert "shallow processing" in violations[0].description.lower()


def test_violation_report(governor):
    # Generate some violations
    governor.validate_step({"reasoning": "However, this contradicts..."}, {})
    governor.validate_step({"action_type": "decompose", "subtasks": []}, {})

    report = governor.get_violation_report()

    assert report["total_violations"] == 2
    assert report["by_type"]["logical_coherence"] == 1
    assert report["by_type"]["compositionality"] == 1
