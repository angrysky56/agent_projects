"""
End-to-End Test for CGRA-Enhanced COMPASS
"""

import pytest
from compass_framework import COMPASS, quick_solve
from config import COMPASSConfig


def test_simple_task_execution():
    """
    Test a simple task execution end-to-end.
    """
    task = "Design a simple calculator application with basic arithmetic operations."

    # Run quick solve
    result = quick_solve(
        task,
        # Override config to speed up test
        self_discover={"max_trials": 3, "pass_threshold": 0.6},
        omcd={"max_resources": 50.0, "R": 1.0},
    )

    # Verify success
    assert result["success"] is True
    assert result["iterations"] > 0
    assert result["score"] >= 0.6

    # Verify artifacts
    assert "trajectory" in result
    assert len(result["trajectory"]["steps"]) > 0
    assert "objectives" in result

    # Verify CGRA components influence (indirectly via logs or structure)
    # We can check if the solution structure implies hierarchical planning
    # since "Design" usually triggers hierarchical representation

    print(f"Task completed with score: {result['score']}")
    print(f"Iterations: {result['iterations']}")


if __name__ == "__main__":
    test_simple_task_execution()
