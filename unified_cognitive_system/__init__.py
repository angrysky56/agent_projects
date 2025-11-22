"""
COMPASS Unified Cognitive System

A sophisticated AI reasoning framework integrating:
- SLAP: Semantic Logic Auto Progression
- SHAPE: Shorthand Assisted Prompt Engineering
- SMART: Strategic Management & Resource Tracking
- oMCD: Online Metacognitive Control of Decisions
- Self-Discover: Reinforcement via Self-Reflection
- Integrated Intelligence: Multi-Modal Intelligence

Author: Ty
License: MIT
"""

__version__ = "0.1.0"

from .compass_framework import COMPASS, create_compass, quick_solve
from .config import COMPASSConfig, oMCDConfig, SLAPConfig, SMARTConfig, SelfDiscoverConfig, SHAPEConfig, IntegratedIntelligenceConfig, get_config, create_custom_config
from .utils import COMPASSLogger, Trajectory, SelfReflection, ObjectiveState, Timer

__all__ = [
    # Main Framework
    "COMPASS",
    "create_compass",
    "quick_solve",
    # Configuration
    "COMPASSConfig",
    "oMCDConfig",
    "SLAPConfig",
    "SMARTConfig",
    "SelfDiscoverConfig",
    "SHAPEConfig",
    "IntegratedIntelligenceConfig",
    "get_config",
    "create_custom_config",
    # Utilities
    "COMPASSLogger",
    "Trajectory",
    "SelfReflection",
    "ObjectiveState",
    "Timer",
]
