"""
AI-driven auto-configuration system for COMPASS framework.

Intelligently selects optimal framework parameters based on task analysis.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
import logging

# Import COMPASS config
# import sys
# import os

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .core.config import COMPASSConfig, create_custom_config

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels."""

    SIMPLE = "simple"  # Quick questions, simple queries
    MODERATE = "moderate"  # Standard tasks requiring some reasoning
    COMPLEX = "complex"  # Multi-step problems, deep reasoning needed
    CRITICAL = "critical"  # High-stakes, requires maximum resources


class TaskDomain(Enum):
    """Task domain categories."""

    CODING = "coding"
    MATH = "math"
    PLANNING = "planning"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    GENERAL = "general"


@dataclass
class TaskFeatures:
    """Extracted features from task analysis."""

    complexity: TaskComplexity
    domain: TaskDomain
    word_count: int
    has_code: bool
    has_math: bool
    has_constraints: bool
    estimated_steps: int
    requires_creativity: bool
    technical_density: float  # 0.0 to 1.0


@dataclass
class FrameworkActivation:
    """Which COMPASS frameworks to activate."""

    shape: bool = True  # Always active for input processing
    slap: bool = False  # Semantic logic progression
    smart: bool = False  # Strategic planning
    omcd: bool = False  # Resource allocation
    self_discover: bool = False  # Self-reflection
    intelligence: bool = True  # Always active for decision synthesis

    def get_active_count(self) -> int:
        """Count active frameworks."""
        return sum([self.shape, self.slap, self.smart, self.omcd, self.self_discover, self.intelligence])


class TaskAnalyzer:
    """Analyzes user tasks to extract features for configuration."""

    # Technical keywords for different domains
    CODING_KEYWORDS = {"function", "class", "variable", "algorithm", "code", "debug", "implement", "refactor", "api", "library", "framework", "syntax"}

    MATH_KEYWORDS = {"calculate", "equation", "solve", "proof", "theorem", "formula", "probability", "statistics", "optimize", "derivative", "integral"}

    PLANNING_KEYWORDS = {"plan", "schedule", "organize", "coordinate", "timeline", "roadmap", "strategy", "milestone", "objective", "goal"}

    ANALYSIS_KEYWORDS = {"analyze", "evaluate", "compare", "assess", "investigate", "examine", "review", "critique", "interpret", "synthesize"}

    CREATIVE_KEYWORDS = {"create", "design", "generate", "invent", "brainstorm", "imagine", "innovative", "original", "novel", "artistic"}

    def analyze(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> TaskFeatures:
        """Analyze a task and extract features."""
        task_lower = task_description.lower()
        words = task_lower.split()
        word_count = len(words)

        # Detect code presence
        has_code = bool(re.search(r"[{}()[\];]|`.*`|```", task_description))

        # Detect math presence
        has_math = bool(re.search(r"[\d+\-*/=<>∫∑√π]|math|equation", task_lower))

        # Detect constraints
        has_constraints = bool(re.search(r"must|should|require|need|constraint|limit|within", task_lower))

        # Estimate steps (questions, conjunctions, enumerations)
        question_count = task_description.count("?")
        conjunction_count = len(re.findall(r"\b(and|then|after|next|also)\b", task_lower))
        list_items = len(re.findall(r"(?:^|\n)\s*[\d•\-*]", task_description))
        estimated_steps = max(1, question_count + conjunction_count + list_items)

        # Domain detection
        domain = self._detect_domain(task_lower)

        # Creativity detection
        requires_creativity = any(kw in task_lower for kw in self.CREATIVE_KEYWORDS)

        # Technical density
        technical_terms = sum(len([w for w in words if w in keywords]) for keywords in [self.CODING_KEYWORDS, self.MATH_KEYWORDS, self.PLANNING_KEYWORDS, self.ANALYSIS_KEYWORDS])
        technical_density = min(1.0, technical_terms / max(1, word_count) * 10)

        # Determine complexity
        complexity = self._determine_complexity(word_count, estimated_steps, has_constraints, technical_density, context)

        return TaskFeatures(complexity=complexity, domain=domain, word_count=word_count, has_code=has_code, has_math=has_math, has_constraints=has_constraints, estimated_steps=estimated_steps, requires_creativity=requires_creativity, technical_density=technical_density)

    def _detect_domain(self, task_lower: str) -> TaskDomain:
        """Detect the primary domain of the task."""
        scores = {
            TaskDomain.CODING: sum(1 for kw in self.CODING_KEYWORDS if kw in task_lower),
            TaskDomain.MATH: sum(1 for kw in self.MATH_KEYWORDS if kw in task_lower),
            TaskDomain.PLANNING: sum(1 for kw in self.PLANNING_KEYWORDS if kw in task_lower),
            TaskDomain.ANALYSIS: sum(1 for kw in self.ANALYSIS_KEYWORDS if kw in task_lower),
            TaskDomain.CREATIVE: sum(1 for kw in self.CREATIVE_KEYWORDS if kw in task_lower),
        }

        max_score = max(scores.values())
        if max_score == 0:
            return TaskDomain.GENERAL

        # Return domain with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def _determine_complexity(self, word_count: int, estimated_steps: int, has_constraints: bool, technical_density: float, context: Optional[Dict[str, Any]]) -> TaskComplexity:
        """Determine task complexity level."""
        # Calculate complexity score
        score = 0

        # Word count contribution
        if word_count < 20:
            score += 0
        elif word_count < 50:
            score += 1
        elif word_count < 100:
            score += 2
        else:
            score += 3

        # Steps contribution
        if estimated_steps <= 1:
            score += 0
        elif estimated_steps <= 3:
            score += 1
        elif estimated_steps <= 7:
            score += 2
        else:
            score += 3

        # Technical density
        score += int(technical_density * 2)

        # Constraints add complexity
        if has_constraints:
            score += 1

        # Context considerations
        if context:
            if context.get("high_stakes"):
                score += 2
            if context.get("time_critical"):
                score += 1

        # Map score to complexity
        if score <= 2:
            return TaskComplexity.SIMPLE
        elif score <= 5:
            return TaskComplexity.MODERATE
        elif score <= 8:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.CRITICAL


class ConfigOptimizer:
    """Optimizes COMPASS configuration based on task features."""

    def __init__(self):
        self.analyzer = TaskAnalyzer()

    def optimize(self, task_description: str, context: Optional[Dict[str, Any]] = None, user_overrides: Optional[Dict[str, Any]] = None) -> tuple[COMPASSConfig, FrameworkActivation]:
        """
        Generate optimal COMPASS configuration.

        Returns:
            Tuple of (COMPASSConfig, FrameworkActivation)
        """
        # Analyze task
        features = self.analyzer.analyze(task_description, context)

        logger.info(f"Task analysis: {features.complexity.value} {features.domain.value} ({features.word_count} words, {features.estimated_steps} steps)")

        # Determine framework activation
        activation = self._select_frameworks(features)

        # Create base config
        config_params = self._create_config_params(features, activation)

        # Apply user overrides
        if user_overrides:
            config_params.update(user_overrides)

        config = create_custom_config(**config_params)

        logger.info(f"Activated {activation.get_active_count()}/6 frameworks")

        return config, activation

    def _select_frameworks(self, features: TaskFeatures) -> FrameworkActivation:
        """Select which frameworks to activate based on task features."""
        activation = FrameworkActivation()

        # SHAPE and Intelligence are always active
        activation.shape = True
        activation.intelligence = True

        # Activate based on complexity
        if features.complexity == TaskComplexity.SIMPLE:
            # Simple tasks: just SHAPE + Intelligence (fast)
            pass

        elif features.complexity == TaskComplexity.MODERATE:
            # Moderate: add SLAP for reasoning
            activation.slap = True

            # Add SMART if planning domain
            if features.domain == TaskDomain.PLANNING:
                activation.smart = True

        elif features.complexity == TaskComplexity.COMPLEX:
            # Complex: activate most frameworks
            activation.slap = True
            activation.smart = True
            activation.omcd = True  # Resource optimization

            # Add self-discover for multi-step tasks
            if features.estimated_steps >= 5:
                activation.self_discover = True

        else:  # CRITICAL
            # Critical: full COMPASS (all 6 frameworks)
            activation.slap = True
            activation.smart = True
            activation.omcd = True
            activation.self_discover = True

        return activation

    def _create_config_params(self, features: TaskFeatures, activation: FrameworkActivation) -> Dict[str, Any]:
        """Create configuration parameters based on features."""
        params = {}

        # oMCD configuration
        if activation.omcd:
            omcd_params = {}

            # Adjust resource allocation based on complexity
            if features.complexity == TaskComplexity.CRITICAL:
                omcd_params["R"] = 20.0  # High importance
                omcd_params["max_resources"] = 150.0
            elif features.complexity == TaskComplexity.COMPLEX:
                omcd_params["R"] = 15.0
                omcd_params["max_resources"] = 120.0
            else:
                omcd_params["R"] = 10.0  # Default

            params["omcd"] = omcd_params

        # SLAP configuration
        if activation.slap:
            slap_params = {}

            # Adjust scrutiny vs improvement based on domain
            if features.domain == TaskDomain.MATH:
                # High scrutiny for math
                slap_params["alpha"] = 0.6  # Scrutiny
                slap_params["beta"] = 0.4  # Improvement
            elif features.domain == TaskDomain.CREATIVE:
                # High improvement for creative
                slap_params["alpha"] = 0.3
                slap_params["beta"] = 0.7
            else:
                # Balanced
                slap_params["alpha"] = 0.4
                slap_params["beta"] = 0.6

            params["slap"] = slap_params

        # Self-Discover configuration
        if activation.self_discover:
            sd_params = {}

            # More trials for complex tasks
            if features.complexity == TaskComplexity.CRITICAL:
                sd_params["max_trials"] = 15
            elif features.complexity == TaskComplexity.COMPLEX:
                sd_params["max_trials"] = 10
            else:
                sd_params["max_trials"] = 7

            params["self_discover"] = sd_params

        return params


# Global optimizer instance
_optimizer: Optional[ConfigOptimizer] = None


def get_optimizer() -> ConfigOptimizer:
    """Get or create the global configuration optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ConfigOptimizer()
    return _optimizer


def auto_configure(task_description: str, context: Optional[Dict[str, Any]] = None, user_overrides: Optional[Dict[str, Any]] = None) -> tuple[COMPASSConfig, FrameworkActivation]:
    """
    Main entry point for auto-configuration.

    Args:
        task_description: The user's task description
        context: Optional context (e.g., {'high_stakes': True})
        user_overrides: Optional manual parameter overrides

    Returns:
        Tuple of (COMPASSConfig, FrameworkActivation)
    """
    optimizer = get_optimizer()
    return optimizer.optimize(task_description, context, user_overrides)
