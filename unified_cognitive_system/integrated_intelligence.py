"""
Integrated Intelligence - Multi-Modal Intelligence Core

Implements multi-modal intelligence combining learning, reasoning, NLU,
uncertainty quantification, and decision synthesis.
"""

from typing import Any, Dict, List, Optional, Callable
import math
import numpy as np

from utils import COMPASSLogger, sigmoid, entropy


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

    def __init__(self, config, logger: Optional[COMPASSLogger] = None):
        """
        Initialize Integrated Intelligence core.

        Args:
            config: IntegratedIntelligenceConfig instance
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or COMPASSLogger("IntegratedIntelligence")

        # Learning state
        self.Q_table = {}  # Q-learning table
        self.knowledge_base = {}

        self.logger.info("Integrated Intelligence initialized")

    def make_decision(self, task: str, reasoning_plan: Dict, modules: List[int], resources: Dict, context: Dict) -> Dict[str, Any]:
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

        # Calculate universal intelligence score
        universal_score = self._universal_intelligence(intelligence_scores)

        # Generate decision
        decision = {"task": task, "action": self._generate_action(universal_score, reasoning_plan, modules), "confidence": universal_score, "intelligence_breakdown": intelligence_scores, "reasoning": self._generate_reasoning(intelligence_scores, reasoning_plan), "estimated_quality": universal_score}

        # Update learning
        self._update_learning(features, decision, universal_score)

        self.logger.debug(f"Decision made with confidence {universal_score:.3f}")
        return decision

    def _universal_intelligence(self, scores: Dict[str, float]) -> float:
        """
        Calculate universal intelligence using weighted combination.

        Formula: U(x) = Σ(ωᵢ * Fᵢ(x)) + Σ(ωⱼₖ * Fⱼ(x) * Fₖ(x))

        Args:
            scores: Intelligence component scores

        Returns:
            Universal intelligence score
        """
        # Linear combination
        linear_sum = sum(self.config.linear_weights.get(key, 0.0) * value for key, value in scores.items())

        # Interaction terms (pairwise)
        interaction_sum = 0.0
        keys = list(scores.keys())
        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1 :]:
                interaction_sum += scores[key1] * scores[key2]

        interaction_sum *= self.config.interaction_weight

        # Combine
        total = linear_sum + interaction_sum

        # Normalize to [0, 1]
        return min(1.0, max(0.0, total))

    def _extract_features(self, task: str, reasoning_plan: Dict, modules: List[int], resources: Dict, context: Dict) -> np.ndarray:
        """Extract feature vector from inputs."""
        features = []

        # Task complexity (based on length and technical terms)
        features.append(min(1.0, len(task) / 500.0))
        features.append(len(task.split()) / 100.0)

        # Reasoning plan advancement
        if "advancement" in reasoning_plan:
            features.append(reasoning_plan["advancement"])
        else:
            features.append(0.5)

        # Number of reasoning modules
        features.append(len(modules) / 10.0)

        # Resource allocation
        features.append(resources.get("confidence", 0.5))

        # Context richness
        features.append(min(1.0, len(context) / 10.0))

        return np.array(features)

    def _learning_intelligence(self, features: np.ndarray, context: Dict) -> float:
        """
        Learning component using Q-learning principles.

        Args:
            features: Feature vector
            context: Context

        Returns:
            Learning score
        """
        # Simple feature-based learning
        state_key = tuple(features.round(1))

        # Initialize if not seen
        if state_key not in self.Q_table:
            self.Q_table[state_key] = 0.5

        return self.Q_table[state_key]

    def _reasoning_intelligence(self, features: np.ndarray, reasoning_plan: Dict) -> float:
        """
        Reasoning component using Bayesian principles.

        Args:
            features: Feature vector
            reasoning_plan: SLAP plan

        Returns:
            Reasoning score
        """
        # Use advancement from SLAP if available
        if "advancement" in reasoning_plan:
            base_score = reasoning_plan["advancement"] / 3.0  # Normalize
        else:
            base_score = 0.5

        # Adjust based on semantic completeness
        if "semantic" in reasoning_plan:
            semantic = reasoning_plan["semantic"]
            if isinstance(semantic, dict) and semantic.get("ready_for_execution"):
                base_score *= 1.2

        return min(1.0, base_score)

    def _nlu_intelligence(self, task: str, context: Dict) -> float:
        """
        Natural Language Understanding component.

        Args:
            task: Task description
            context: Context

        Returns:
            NLU score
        """
        # Simple NLU: check for clarity and structure
        score = 0.5

        # Bonus for longer, more detailed tasks
        if len(task) > 50:
            score += 0.1

        # Bonus for presence of key action words
        action_words = ["create", "build", "analyze", "optimize", "implement"]
        if any(word in task.lower() for word in action_words):
            score += 0.2

        # Bonus for structured language
        if "." in task or "\n" in task:
            score += 0.1

        return min(1.0, score)

    def _uncertainty_intelligence(self, features: np.ndarray) -> float:
        """
        Uncertainty quantification using entropy.

        Args:
            features: Feature vector

        Returns:
            Uncertainty score (lower uncertainty = higher score)
        """
        # Normalize features to probabilities
        if features.sum() > 0:
            probs = features / features.sum()
        else:
            probs = np.ones_like(features) / len(features)

        # Calculate entropy
        ent = entropy(probs.tolist())

        # Convert to score: lower entropy = higher certainty = higher score
        max_entropy = math.log2(len(features))
        certainty = 1.0 - (ent / max_entropy if max_entropy > 0 else 0)

        return certainty

    def _evolutionary_intelligence(self, features: np.ndarray, context: Dict) -> float:
        """
        Evolutionary intelligence component.

        Args:
            features: Feature vector
            context: Context

        Returns:
            Evolution score
        """
        # Simple fitness function based on feature quality
        fitness = features.mean()

        # Bonus for diversity in features
        if features.std() > 0.1:
            fitness += 0.1

        return min(1.0, fitness)

    def _neural_intelligence(self, features: np.ndarray) -> float:
        """
        Neural activation component.

        Args:
            features: Feature vector

        Returns:
            Neural score
        """
        # Simple neural network: weighted sum + sigmoid
        weights = np.array([0.2, 0.15, 0.3, 0.15, 0.15, 0.05])[: len(features)]

        if len(weights) != len(features):
            weights = np.ones(len(features)) / len(features)

        activation = np.dot(features, weights)

        return sigmoid(activation, k=self.config.fuzzy_k, c=self.config.fuzzy_c)

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
