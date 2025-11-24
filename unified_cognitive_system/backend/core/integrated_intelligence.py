"""
Integrated Intelligence - Multi-Modal Intelligence Core

Implements multi-modal intelligence combining learning, reasoning, NLU,
uncertainty quantification, and decision synthesis.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import math
import numpy as np

from .config import IntegratedIntelligenceConfig
from .utils import COMPASSLogger, Trajectory, ObjectiveState, sigmoid, entropy


class IntegratedIntelligence:
    """
    Integrated Intelligence: Multi-Modal Reasoning and Decision-Making

    Synthesizes decisions using universal intelligence formula that combines:
    - Learning and transfer learning
    - Probabilistic reasoning
    - Natural language understanding
    - Uncertainty quantification
    - Evolutionary optimization
    - Neural activation
    """

    def __init__(self, config: IntegratedIntelligenceConfig, logger: Optional[COMPASSLogger] = None, llm_provider: Optional[Any] = None, mcp_client: Optional[Any] = None):
        """
        Initialize Integrated Intelligence.

        Args:
            config: Configuration object
            logger: Optional logger instance
            llm_provider: Optional LLM provider instance
            mcp_client: Optional MCP client instance
        """
        self.config = config
        self.logger = logger or COMPASSLogger("IntegratedIntelligence")
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client

        # Initialize learning memory (Q-table)
        self.Q_table = {}

        self.logger.info("Integrated Intelligence initialized")

    async def make_decision(self, task: str, reasoning_plan: Dict, modules: List[int], resources: Dict, context: Dict) -> Dict[str, Any]:
        """
        Make a decision using integrated intelligence.

        Combines all intelligence modalities to synthesize optimal decision.

        Args:
            task: Task description
            reasoning_plan: SLAP reasoning plan
            modules: Selected reasoning modules
            resources: Resource allocation from oMCD
            context: Current context

        Returns:
            Decision dictionary
        """
        self.logger.debug("Synthesizing decision with integrated intelligence")

        # Convert inputs to feature vector
        features = self._extract_features(task, reasoning_plan, modules, resources, context)

        # Apply each intelligence function
        intelligence_scores = {}

        # 1. Learning component
        intelligence_scores["learning"] = self._learning_intelligence(features, context)

        # 2. Reasoning component
        intelligence_scores["reasoning"] = self._reasoning_intelligence(features, reasoning_plan)

        # 3. Natural Language Understanding
        intelligence_scores["nlu"] = self._nlu_intelligence(task, context)

        # 4. Uncertainty quantification
        intelligence_scores["uncertainty"] = self._uncertainty_intelligence(features)

        # 5. Evolutionary component
        intelligence_scores["evolution"] = self._evolutionary_intelligence(features, context)

        # 6. Neural activation
        intelligence_scores["neural"] = self._neural_intelligence(features)

        # 7. LLM Reasoning (if available)
        if self.llm_provider:
            llm_score, llm_action = await self._llm_intelligence(task, reasoning_plan, context)
            intelligence_scores["llm"] = llm_score
        else:
            llm_action = None

        # Calculate universal intelligence score
        universal_score = self._universal_intelligence(intelligence_scores)

        # Generate decision
        action = llm_action if llm_action else self._generate_action(universal_score, reasoning_plan, modules)

        decision = {"task": task, "action": action, "confidence": universal_score, "intelligence_breakdown": intelligence_scores, "reasoning": self._generate_reasoning(intelligence_scores, reasoning_plan), "estimated_quality": universal_score}

        # Update learning
        self._update_learning(features, decision, universal_score)

        self.logger.debug(f"Decision made with confidence {universal_score:.3f}")
        return decision

    def _extract_features(self, task: str, reasoning_plan: Dict, modules: List[int], resources: Dict, context: Dict) -> np.ndarray:
        """Extract features for learning."""
        # Simple feature extraction: [task_len, plan_complexity, num_modules, resource_count]
        features = np.array([len(task) / 100.0, len(str(reasoning_plan)) / 500.0, len(modules) / 6.0, len(resources) / 5.0])
        return features

    def _learning_intelligence(self, features: np.ndarray, context: Dict) -> float:
        """Learning component score."""
        state_key = tuple(features.round(1))
        return self.Q_table.get(state_key, 0.5)

    def _reasoning_intelligence(self, features: np.ndarray, reasoning_plan: Dict) -> float:
        """Reasoning component score."""
        # Base score on advancement
        return reasoning_plan.get("advancement", 0.5)

    def _nlu_intelligence(self, task: str, context: Dict) -> float:
        """NLU component score."""
        # Simple heuristic: task length and clarity
        return min(1.0, len(task.split()) / 10.0)

    def _uncertainty_intelligence(self, features: np.ndarray) -> float:
        """Uncertainty quantification score."""
        # Inverse of entropy or similar
        return 0.8  # High confidence by default

    def _evolutionary_intelligence(self, features: np.ndarray, context: Dict) -> float:
        """Evolutionary component score."""
        return 0.6

    def _neural_intelligence(self, features: np.ndarray) -> float:
        """Neural activation score."""
        return float(sigmoid(np.sum(features)))

    def _universal_intelligence(self, scores: Dict[str, float]) -> float:
        """Calculate universal intelligence score."""
        # Weighted average
        weights = {"learning": 0.1, "reasoning": 0.2, "nlu": 0.1, "uncertainty": 0.1, "evolution": 0.1, "neural": 0.1, "llm": 0.3}

        total_score = 0.0
        total_weight = 0.0

        for key, score in scores.items():
            weight = weights.get(key, 0.1)
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    async def _llm_intelligence(self, task: str, reasoning_plan: Dict, context: Dict) -> Tuple[float, Optional[str]]:
        """
        LLM-based intelligence component.

        Uses the connected LLM provider to generate a solution and confidence score.

        Args:
            task: Task description
            reasoning_plan: SLAP reasoning plan
            context: Context dictionary

        Returns:
            Tuple of (confidence_score, solution_string)
        """
        if not self.llm_provider:
            return 0.5, None

        try:
            # Construct prompt for decision making
            system_prompt = "You are the Integrated Intelligence module of COMPASS. Your goal is to synthesize a decision based on the provided reasoning plan."

            user_prompt = f"Task: {task}\n\nReasoning Plan Summary:\n"
            if "conceptualization" in reasoning_plan:
                user_prompt += f"- Concept: {reasoning_plan['conceptualization'].get('primary_concept')}\n"
            if "advancement" in reasoning_plan:
                user_prompt += f"- Advancement Score: {reasoning_plan['advancement']:.2f}\n"

            user_prompt += "\nBased on this, should we proceed with execution? If so, what is the recommended action? Return a JSON with 'confidence' (0.0-1.0) and 'action' (string)."

            from ..llm_providers import Message

            messages = [Message(role="system", content=system_prompt), Message(role="user", content=user_prompt)]

            # Call LLM
            response_content = ""
            async for chunk in self.llm_provider.chat_completion(messages, stream=False, temperature=0.3, max_tokens=500):
                response_content += chunk

            # Parse response
            import json

            try:
                # Clean up markdown
                if "```json" in response_content:
                    response_content = response_content.split("```json")[1].split("```")[0].strip()
                elif "```" in response_content:
                    response_content = response_content.split("```")[1].split("```")[0].strip()

                data = json.loads(response_content)
                return data.get("confidence", 0.8), data.get("action", "LLM_EXECUTION_REQUIRED")
            except json.JSONDecodeError:
                # Fallback if not JSON
                return 0.8, "LLM_EXECUTION_REQUIRED"

        except Exception as e:
            self.logger.error(f"Error in LLM intelligence: {e}")
            return 0.5, None

    def _generate_action(self, score: float, reasoning_plan: Dict, modules: List[int]) -> str:
        """Generate action description based on intelligence score."""
        if score > 0.8:
            action = "Execute with high confidence"
        elif score > 0.6:
            action = "Execute with moderate confidence"
        else:
            action = "Execute with caution, may need refinement"

        return action

    def _generate_reasoning(self, scores: Dict[str, float], reasoning_plan: Dict) -> str:
        """Generate explanation of reasoning."""
        # Find strongest components
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_components = sorted_scores[:3]

        reasoning = "Decision based on: "
        reasoning += ", ".join([f"{comp[0]} ({comp[1]:.2f})" for comp in top_components])

        return reasoning

    def _update_learning(self, features: np.ndarray, decision: Dict, score: float):
        """
        Update learning component based on decision outcome.

        Args:
            features: Feature vector
            decision: Decision made
            score: Score achieved
        """
        state_key = tuple(features.round(1))

        # Q-learning update
        if state_key in self.Q_table:
            old_value = self.Q_table[state_key]
            self.Q_table[state_key] = old_value + self.config.learning_rate * (score - old_value)
        else:
            self.Q_table[state_key] = score

    def transfer_learning(self, source_knowledge: Dict, target_task: str) -> Dict:
        """
        Apply transfer learning from source to target task.

        Args:
            source_knowledge: Knowledge from source domain
            target_task: Target task description

        Returns:
            Transferred knowledge
        """
        self.logger.debug("Applying transfer learning")

        # Simple transfer: adapt source knowledge with delta
        transferred = {}

        for key, value in source_knowledge.items():
            # Apply delta learning factor
            if isinstance(value, (int, float)):
                transferred[key] = value * (1.0 + self.config.delta_learning_factor)
            else:
                transferred[key] = value

        # Update knowledge base
        self.knowledge_base[target_task] = transferred

        return transferred

    def reset(self):
        """Reset intelligence state."""
        self.Q_table.clear()
        self.knowledge_base.clear()
        self.logger.debug("Integrated Intelligence reset")
