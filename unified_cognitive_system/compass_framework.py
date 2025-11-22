"""
COMPASS Framework - Core Orchestrator

COMPASS: Cognitive Orchestration & Metacognitive Planning for Adaptive Semantic Systems

This is the main orchestrator that coordinates all subsystems:
- SHAPE (User Interface Layer)
- oMCD + Self-Discover (Metacognitive Layer)
- SLAP + SMART (Reasoning & Planning Layer)
- Integrated Intelligence (Execution Layer)
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from config import COMPASSConfig, get_config
from utils import COMPASSLogger, Trajectory, SelfReflection, ObjectiveState, Timer, ensure_directory, save_json


class COMPASS:
    """
    Main orchestrator for the COMPASS cognitive architecture.

    Integrates all six frameworks into a unified decision-making and reasoning system.
    """

    def __init__(self, config: Optional[COMPASSConfig] = None):
        """
        Initialize COMPASS framework.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or get_config()
        self.logger = COMPASSLogger("COMPASS", level=self.config.log_level)

        # Initialize output directory
        if self.config.save_trajectories or self.config.save_reflections:
            self.output_dir = ensure_directory(self.config.output_directory)

        # Component placeholders (will be lazily initialized)
        self._shape_processor = None
        self._omcd_controller = None
        self._self_discover_engine = None
        self._slap_pipeline = None
        self._smart_planner = None
        self._intelligence_core = None

        # State tracking
        self.trajectories: List[Trajectory] = []
        self.reflections: List[SelfReflection] = []
        self.objectives: List[ObjectiveState] = []
        self.current_resources: float = self.config.omcd.max_resources

        self.logger.info("COMPASS framework initialized")

    @property
    def shape_processor(self):
        """Lazy initialization of SHAPE processor."""
        if self._shape_processor is None:
            from shape_processor import SHAPEProcessor

            self._shape_processor = SHAPEProcessor(self.config.shape, self.logger)
        return self._shape_processor

    @property
    def omcd_controller(self):
        """Lazy initialization of oMCD controller."""
        if self._omcd_controller is None:
            from omcd_controller import oMCDController

            self._omcd_controller = oMCDController(self.config.omcd, self.logger)
        return self._omcd_controller

    @property
    def self_discover_engine(self):
        """Lazy initialization of Self-Discover engine."""
        if self._self_discover_engine is None:
            from self_discover_engine import SelfDiscoverEngine

            self._self_discover_engine = SelfDiscoverEngine(self.config.self_discover, self.logger)
        return self._self_discover_engine

    @property
    def slap_pipeline(self):
        """Lazy initialization of SLAP pipeline."""
        if self._slap_pipeline is None:
            from slap_pipeline import SLAPPipeline

            self._slap_pipeline = SLAPPipeline(self.config.slap, self.logger)
        return self._slap_pipeline

    @property
    def smart_planner(self):
        """Lazy initialization of SMART planner."""
        if self._smart_planner is None:
            from smart_planner import SMARTPlanner

            self._smart_planner = SMARTPlanner(self.config.smart, self.logger)
        return self._smart_planner

    @property
    def intelligence_core(self):
        """Lazy initialization of Intelligence core."""
        if self._intelligence_core is None:
            from integrated_intelligence import IntegratedIntelligence

            self._intelligence_core = IntegratedIntelligence(self.config.intelligence, self.logger)
        return self._intelligence_core

    def process_task(self, task_description: str, context: Optional[Dict[str, Any]] = None, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a complete task through the COMPASS framework.

        This is the main entry point that orchestrates all layers:
        1. SHAPE processes user input
        2. SMART creates objectives
        3. SLAP builds reasoning plan
        4. Self-Discover manages execution with reflection
        5. oMCD optimizes resource allocation
        6. Integrated Intelligence executes decisions

        Args:
            task_description: Natural language task description
            context: Optional context information
            max_iterations: Maximum iterations (uses Self-Discover config if None)

        Returns:
            Result dictionary with solution, trajectory, reflections, and metrics
        """
        with Timer("Task Processing", self.logger):
            self.logger.info(f"Processing task: {task_description}")

            context = context or {}
            max_iterations = max_iterations or self.config.self_discover.max_trials

            # ================================================================
            # Layer 1: User Interface - SHAPE Processing
            # ================================================================
            self.logger.info("Layer 1: SHAPE processing user input")
            processed_input = self.shape_processor.process_user_input(task_description)
            expanded_prompt = self.shape_processor.expand_shorthand(processed_input)
            semantic_prompt = self.shape_processor.map_semantics(expanded_prompt, context)

            # ================================================================
            # Layer 2: Planning - SMART Objectives
            # ================================================================
            self.logger.info("Layer 2: Creating SMART objectives")
            # Extract enriched prompt string for components that expect text
            task_text = semantic_prompt["enriched_prompt"]

            objectives = self.smart_planner.create_objectives_from_task(task_text, context)
            self.objectives.extend(objectives)

            # ================================================================
            # Layer 3: Reasoning - SLAP Pipeline
            # ================================================================
            self.logger.info("Layer 3: Building SLAP reasoning structure")
            reasoning_plan = self.slap_pipeline.create_reasoning_plan(task_text, objectives)

            # ================================================================
            # Layer 4: Metacognitive Control - Self-Discover with oMCD
            # ================================================================
            self.logger.info("Layer 4: Executing with metacognitive control")

            # Initialize trajectory
            trajectory = Trajectory(steps=[])
            best_solution = None
            best_score = float("-inf")

            iteration = 0
            while iteration < max_iterations:
                self.logger.debug(f"Iteration {iteration + 1}/{max_iterations}")

                # Select reasoning modules (Self-Discover)
                selected_modules = self.self_discover_engine.select_reasoning_modules(task_text, self.reflections)

                # Determine resource allocation (oMCD)
                resource_allocation = self.omcd_controller.determine_resource_allocation(current_state=context, importance=self._calculate_importance(objectives), available_resources=self.current_resources)

                # Execute reasoning with allocated resources
                action, observation = self._execute_reasoning_step(task_text, reasoning_plan, selected_modules, resource_allocation, context)

                trajectory.add_step(action, observation)

                # Evaluate trajectory
                score = self.self_discover_engine.evaluate_trajectory(trajectory, objectives)

                # Update best solution
                if score > best_score:
                    best_score = score
                    best_solution = observation

                # Check if passing threshold reached
                if score >= self.config.self_discover.pass_threshold:
                    self.logger.info(f"Passing threshold reached at iteration {iteration + 1}")
                    break

                # Generate self-reflection
                reflection = self.self_discover_engine.generate_reflection(trajectory, score, objectives)
                self.reflections.append(reflection)

                # Update context with reflection insights
                context["reflections"] = [r.to_dict() for r in self.reflections[-5:]]

                # Check stopping criterion (oMCD optimal stopping)
                if self.omcd_controller.should_stop(score, iteration, resource_allocation):
                    self.logger.info(f"Optimal stopping criterion met at iteration {iteration + 1}")
                    break

                iteration += 1

            # ================================================================
            # Finalization
            # ================================================================
            trajectory.score = best_score
            self.trajectories.append(trajectory)

            # Save artifacts if configured
            if self.config.save_trajectories:
                self._save_trajectory(trajectory)

            if self.config.save_reflections:
                self._save_reflections()

            # Update SMART objectives progress
            for obj in objectives:
                obj.current_value = best_score * obj.target_value

            # Collect feedback for SHAPE adaptation
            self.shape_processor.collect_feedback(original_input=task_description, processed_output=best_solution, score=best_score)

            # Prepare result
            result = {
                "success": best_score >= self.config.self_discover.pass_threshold,
                "solution": best_solution,
                "score": best_score,
                "iterations": iteration + 1,
                "trajectory": trajectory.to_dict(),
                "reflections": [r.to_dict() for r in self.reflections],
                "objectives": [obj.to_dict() for obj in objectives],
                "resources_used": self.config.omcd.max_resources - self.current_resources,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(f"Task completed: success={result['success']}, score={best_score:.3f}, iterations={iteration + 1}")

            return result

    def _execute_reasoning_step(self, task: str, reasoning_plan: Dict, selected_modules: List[int], resource_allocation: float, context: Dict) -> Tuple[str, Any]:
        """
        Execute a single reasoning step using all available intelligence.

        Args:
            task: Task description
            reasoning_plan: SLAP reasoning plan
            selected_modules: Selected reasoning module indices
            resource_allocation: Amount of resources to use
            context: Current context

        Returns:
            (action, observation) tuple
        """
        # Use Integrated Intelligence to synthesize decision
        decision = self.intelligence_core.make_decision(task=task, reasoning_plan=reasoning_plan, modules=selected_modules, resources=resource_allocation, context=context)

        # Update resource tracking
        self.current_resources -= resource_allocation["amount"]

        # Action is the decision plan, observation is the result
        action = {"decision": decision, "modules": selected_modules, "resources": resource_allocation}

        observation = decision  # In a real system, this would be actual execution results

        return action, observation

    def _calculate_importance(self, objectives: List[ObjectiveState]) -> float:
        """
        Calculate importance weight for resource allocation.

        Args:
            objectives: List of objectives

        Returns:
            Importance weight (R parameter for oMCD)
        """
        if not objectives:
            return self.config.omcd.R

        # Weight by number of high-priority objectives
        return self.config.omcd.R * (1.0 + 0.1 * len(objectives))

    def _save_trajectory(self, trajectory: Trajectory):
        """Save trajectory to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"trajectory_{timestamp}.json"
        save_json(trajectory.to_dict(), str(filename))
        self.logger.debug(f"Trajectory saved to {filename}")

    def _save_reflections(self):
        """Save all reflections to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"reflections_{timestamp}.json"
        data = [r.to_dict() for r in self.reflections]
        save_json(data, str(filename))
        self.logger.debug(f"Reflections saved to {filename}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current framework status.

        Returns:
            Status dictionary with metrics and state
        """
        return {
            "trajectories_count": len(self.trajectories),
            "reflections_count": len(self.reflections),
            "objectives_count": len(self.objectives),
            "current_resources": self.current_resources,
            "resource_utilization": 1.0 - (self.current_resources / self.config.omcd.max_resources),
            "average_score": np.mean([t.score for t in self.trajectories if t.score is not None]) if self.trajectories else 0.0,
        }

    def reset(self):
        """Reset framework state."""
        self.trajectories.clear()
        self.reflections.clear()
        self.objectives.clear()
        self.current_resources = self.config.omcd.max_resources
        self.logger.info("COMPASS framework reset")


# ============================================================================
# Convenience Functions
# ============================================================================


def create_compass(config: Optional[COMPASSConfig] = None) -> COMPASS:
    """
    Create a COMPASS framework instance.

    Args:
        config: Optional configuration

    Returns:
        Initialized COMPASS instance
    """
    return COMPASS(config)


def quick_solve(task: str, **config_overrides) -> Dict[str, Any]:
    """
    Quick one-shot task solving with COMPASS.

    Args:
        task: Task description
        **config_overrides: Configuration overrides

    Returns:
        Solution result
    """
    from config import create_custom_config

    config = create_custom_config(**config_overrides) if config_overrides else None
    compass = create_compass(config)

    return compass.process_task(task)
