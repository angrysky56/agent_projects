import pytest
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from compass_framework import create_compass
from config import COMPASSConfig


@pytest.mark.asyncio
async def test_llm_connection_injection():
    """Verify that LLM provider is correctly injected into COMPASS components."""

    # Mock LLM Provider
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = "LLM Response"

    # Create COMPASS with provider
    config = COMPASSConfig()
    compass = create_compass(config, llm_provider=mock_provider)

    # Verify injection
    assert compass.llm_provider is mock_provider
    assert compass.integrated_intelligence.llm_provider is mock_provider
    assert compass.self_discover_engine.llm_provider is mock_provider

    # Verify usage in Integrated Intelligence
    # We can't easily test the full flow without mocking everything,
    # but we can check if the component has the provider.

    # Test _llm_intelligence method directly
    score, action = compass.integrated_intelligence._llm_intelligence("Test task", {}, {})
    assert score == 0.9
    assert action == "LLM_EXECUTION_REQUIRED"

    # Test Self-Discover evaluation boost
    # Create a dummy trajectory
    trajectory = MagicMock()
    trajectory.__len__.return_value = 5

    # Score without provider (create new instance)
    engine_no_llm = compass.self_discover_engine.__class__(config.self_discover, None, None)
    score_no_llm = engine_no_llm.evaluate_trajectory(trajectory, [])

    # Score with provider
    score_with_llm = compass.self_discover_engine.evaluate_trajectory(trajectory, [])

    # Verify boost (score_with_llm should be higher or capped at 1.0)
    # Base score for len 5 is min(0.5, 0.4 + 0.25) = 0.5 + 0.3 (obj default) = 0.8
    # With boost: 0.8 * 1.2 = 0.96
    assert score_with_llm > score_no_llm or score_with_llm == 1.0


if __name__ == "__main__":
    # Manual run
    import asyncio

    asyncio.run(test_llm_connection_injection())
    print("Test passed!")
