"""
COMPASS Framework API wrapper with intelligent auto-configuration.

Integrates the COMPASS cognitive framework with the web UI backend.
"""

from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass, asdict
import asyncio
import logging
import sys
import os

# Add parent directory to path for COMPASS imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from compass_framework import create_compass, COMPASS
from config import COMPASSConfig
from auto_config import auto_configure, FrameworkActivation, TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class ProcessingUpdate:
    """Real-time processing update."""

    stage: str
    message: str
    progress: float  # 0.0 to 1.0
    data: Optional[Dict[str, Any]] = None


@dataclass
class COMPASSResult:
    """Result from COMPASS processing."""

    success: bool
    solution: str
    score: float
    iterations: int
    resources_used: float
    reflections: List[Dict[str, Any]]
    trajectory: List[Dict[str, Any]]
    config_used: Dict[str, Any]
    activated_frameworks: Dict[str, bool]


class COMPASSAPIWrapper:
    """
    API wrapper for COMPASS framework with auto-configuration.

    Provides streaming updates and intelligent parameter selection.
    """

    def __init__(self):
        self.compass: Optional[COMPASS] = None
        self.current_config: Optional[COMPASSConfig] = None
        self.current_activation: Optional[FrameworkActivation] = None

    async def process_task(self, task_description: str, context: Optional[Dict[str, Any]] = None, user_config_overrides: Optional[Dict[str, Any]] = None, max_iterations: int = 10) -> AsyncIterator[ProcessingUpdate | COMPASSResult]:
        """
        Process a task through the COMPASS framework with streaming updates.

        Args:
            task_description: The task to process
            context: Optional context information
            user_config_overrides: Manual configuration overrides
            max_iterations: Maximum reasoning iterations

        Yields:
            ProcessingUpdate objects during processing
            Final COMPASSResult at the end
        """
        try:
            # Yield initial update
            yield ProcessingUpdate(stage="initialization", message="Analyzing task and auto-configuring COMPASS...", progress=0.0)

            # Auto-configure COMPASS
            config, activation = auto_configure(task_description, context, user_config_overrides)

            self.current_config = config
            self.current_activation = activation

            yield ProcessingUpdate(stage="configuration", message=f"Configuration optimized (Complexity: {activation.get_active_count()}/6 frameworks)", progress=0.1, data={"activated_frameworks": asdict(activation), "config_summary": self._get_config_summary(config)})

            # Initialize COMPASS
            self.compass = create_compass(config)

            yield ProcessingUpdate(stage="compass_init", message="COMPASS framework initialized", progress=0.2)

            # Process through COMPASS
            # Note: We'll need to modify compass_framework.py to support streaming
            # For now, we'll simulate by running and yielding periodic updates

            yield ProcessingUpdate(stage="processing", message="Processing task through cognitive pipeline...", progress=0.3)

            # Run COMPASS processing (this is synchronous in the current implementation)
            # We'll wrap it in an executor to make it async
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.compass.process_task, task_description, context or {}, max_iterations)

            yield ProcessingUpdate(stage="complete", message="Task processing complete", progress=1.0)

            # Build final result
            compass_result = COMPASSResult(
                success=result.get("success", False),
                solution=result.get("solution", ""),
                score=result.get("score", 0.0),
                iterations=result.get("iterations", 0),
                resources_used=result.get("resources_used", 0.0),
                reflections=result.get("reflections", []),
                trajectory=result.get("trajectory", []),
                config_used=self._get_config_summary(config),
                activated_frameworks=asdict(activation),
            )

            yield compass_result

        except Exception as e:
            logger.error(f"Error processing task: {e}", exc_info=True)
            yield ProcessingUpdate(stage="error", message=f"Error: {str(e)}", progress=0.0)

    def _get_config_summary(self, config: COMPASSConfig) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            "omcd": {
                "R": config.omcd.R,
                "max_resources": config.omcd.max_resources,
            },
            "slap": {
                "alpha": config.slap.alpha,
                "beta": config.slap.beta,
            },
            "self_discover": {
                "max_trials": config.self_discover.max_trials,
            },
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get current COMPASS status."""
        if not self.compass:
            return {"initialized": False}

        status = self.compass.get_status()
        status["initialized"] = True
        status["current_activation"] = asdict(self.current_activation) if self.current_activation else None

        return status

    async def export_reasoning_trace(self) -> Dict[str, Any]:
        """Export full reasoning trace from last execution."""
        if not self.compass:
            return {}

        # Get status which contains history
        status = self.compass.get_status()

        return {"task_history": status.get("task_history", []), "config_used": self._get_config_summary(self.current_config) if self.current_config else {}, "activated_frameworks": asdict(self.current_activation) if self.current_activation else {}}


# Global COMPASS API instance
_compass_api: Optional[COMPASSAPIWrapper] = None


def get_compass_api() -> COMPASSAPIWrapper:
    """Get or create the global COMPASS API instance."""
    global _compass_api
    if _compass_api is None:
        _compass_api = COMPASSAPIWrapper()
    return _compass_api
