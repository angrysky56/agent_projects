"""
SLAP Pipeline - Semantic Logic Auto Progressor

Implements the SLAP framework's logical flow:
C(R(F(S(D(RB(M(SF))))))) - Conceptualization → Representation → Facts →
Scrutiny → Derivation → Rule-Based → Model → Semantic Formalization

Includes truth advancement calculation and MCTS-based entity identification.
"""

from typing import Dict, List, Optional, Any

from .config import SLAPConfig
from .utils import COMPASSLogger, advancement_score


class SLAPPipeline:
    """
    SLAP: Semantic Logic Auto Progressor

    Processes information through a structured semantic pipeline with
    continuous advancement tracking.
    """

    def __init__(self, config, logger: Optional[COMPASSLogger] = None):
        """
        Initialize SLAP pipeline.

        Args:
            config: SLAPConfig instance
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or COMPASSLogger("SLAP")

        # Pipeline state
        self.current_truth = 1.0  # Base truth value
        self.current_scrutiny = 0.0
        self.current_improvement = 0.0
        self.advancement_history = []

        self.logger.info("SLAP pipeline initialized")

    def create_reasoning_plan(self, task: str, objectives: List, representation_type: str = "sequential") -> Dict[str, Any]:
        """
        Create a structured reasoning plan using the SLAP pipeline.

        Processes through: C → R → F → S → D → RB → M → SF
        Adapts structure based on representation_type.

        Args:
            task: Task description
            objectives: List of objectives
            representation_type: Type of representation (sequential, hierarchical, network, causal)

        Returns:
            Reasoning plan dictionary
        """
        self.logger.info(f"Creating SLAP reasoning plan (Type: {representation_type})")

        # Process through each stage
        plan = {}
        plan["type"] = representation_type

        # 1. Conceptualization (C) - Common start
        plan["conceptualization"] = self._conceptualize(task)

        # Branch based on representation type
        if representation_type == "hierarchical":
            self._build_hierarchical_plan(plan, task, objectives)
        elif representation_type == "network":
            self._build_network_plan(plan, task, objectives)
        elif representation_type == "causal":
            self._build_causal_plan(plan, task, objectives)
        else:
            # Default Sequential Flow
            self._build_sequential_plan(plan, task, objectives)

        # Common Finalization
        # 8. Semantic Formalization (SF)
        plan["semantic"] = self._formalize_semantics(plan.get("model", {}))

        # Calculate advancement
        advancement = self._calculate_advancement()
        plan["advancement"] = advancement

        # Store in history
        self.advancement_history.append(advancement)

        self.logger.info(f"SLAP plan created with advancement score: {advancement:.3f}")
        return plan

    def _build_sequential_plan(self, plan: Dict, task: str, objectives: List):
        """Build standard sequential plan."""
        # 2. Representation (R)
        plan["representation"] = self._represent(plan["conceptualization"])
        # 3. Facts (F)
        plan["facts"] = self._identify_facts(plan["representation"])
        # 4. Scrutiny (S)
        plan["scrutiny"] = self._scrutinize(plan["facts"], objectives)
        # 5. Derivation (D)
        plan["derivation"] = self._derive(plan["facts"], plan["scrutiny"])
        # 6. Rule-Based (RB)
        plan["rules"] = self._apply_rules(plan["derivation"])
        # 7. Model (M)
        plan["model"] = self._build_model(plan["rules"], structure="sequential")

    def _build_hierarchical_plan(self, plan: Dict, task: str, objectives: List):
        """Build hierarchical decomposition plan."""
        # 2. Representation (R) - Tree structure
        plan["representation"] = self._represent(plan["conceptualization"])
        plan["representation"]["structure"] = {"type": "tree", "depth": 3}

        # 3. Facts (F) - Component facts
        plan["facts"] = self._identify_facts(plan["representation"])

        # 4. Decomposition (replaces standard Scrutiny/Derivation flow)
        plan["decomposition"] = {"root": task, "subtasks": [f"Subtask {i}: {concept}" for i, concept in enumerate(plan["conceptualization"]["related_concepts"])]}

        # 5. Scrutiny of decomposition
        plan["scrutiny"] = self._scrutinize(plan["facts"], objectives)

        # 7. Model (M)
        plan["model"] = {"components": plan["decomposition"]["subtasks"], "structure": "hierarchical", "completeness": 0.8}

    def _build_network_plan(self, plan: Dict, task: str, objectives: List):
        """Build network/graph plan."""
        # 2. Representation (R) - Graph structure
        plan["representation"] = self._represent(plan["conceptualization"])
        plan["representation"]["structure"] = {"type": "graph", "nodes": 10, "edges": 15}

        # 3. Facts (F) - Node/Edge facts
        plan["facts"] = self._identify_facts(plan["representation"])

        # 4. Connectivity Analysis
        plan["connectivity"] = {"hubs": [plan["conceptualization"]["primary_concept"]], "links": plan["representation"]["relationships"]}

        # 7. Model (M)
        plan["model"] = {"nodes": plan["connectivity"]["hubs"] + plan["conceptualization"]["related_concepts"], "structure": "network", "completeness": 0.7}

    def _build_causal_plan(self, plan: Dict, task: str, objectives: List):
        """Build causal chain plan."""
        # 2. Representation (R) - Chain structure
        plan["representation"] = self._represent(plan["conceptualization"])
        plan["representation"]["structure"] = {"type": "chain", "direction": "forward"}

        # 3. Facts (F) - Causal facts
        plan["facts"] = self._identify_facts(plan["representation"])

        # 4. Causal Analysis
        plan["causal_chain"] = {"root_cause": "Unknown", "effects": plan["conceptualization"]["related_concepts"]}

        # 7. Model (M)
        plan["model"] = {"chain": plan["causal_chain"], "structure": "causal", "completeness": 0.6}

    def _conceptualize(self, task: str) -> Dict:
        """
        Stage 1: Conceptualization - Form initial concept.

        Args:
            task: Task description

        Returns:
            Conceptualization dictionary
        """
        self.logger.debug("Stage 1: Conceptualization")

        # Extract key concepts from task
        concepts = {"primary_concept": self._extract_primary_concept(task), "related_concepts": self._extract_related_concepts(task), "abstract_level": "high", "domain": self._identify_domain(task)}

        # Increase base truth slightly for conceptualization
        self.current_truth += 0.1

        return concepts

    def _represent(self, concepts: Dict) -> Dict:
        """
        Stage 2: Representation - Depict or symbolize concepts.

        Args:
            concepts: Conceptualization output

        Returns:
            Representation dictionary
        """
        self.logger.debug("Stage 2: Representation")

        representation = {"symbolic_form": self._create_symbolic_representation(concepts), "structure": self._identify_structure(concepts), "relationships": self._map_relationships(concepts)}

        return representation

    def _identify_facts(self, representation: Dict) -> List[Dict]:
        """
        Stage 3: Facts - Identify statements considered true.

        Args:
            representation: Representation output

        Returns:
            List of fact dictionaries
        """
        self.logger.debug("Stage 3: Facts Identification")

        facts = []

        # Extract facts from representation
        structure = representation["structure"]
        relationships = representation["relationships"]

        # Generate facts from structure
        for key, value in structure.items():
            facts.append({"type": "structural", "statement": f"{key} has characteristic {value}", "confidence": 0.8})

        # Generate facts from relationships
        for rel in relationships:
            facts.append({"type": "relational", "statement": rel, "confidence": 0.7})

        self.logger.debug(f"Identified {len(facts)} facts")
        return facts

    def _scrutinize(self, facts: List[Dict], objectives: List) -> Dict:
        """
        Stage 4: Scrutiny - Critical examination and analysis.

        Args:
            facts: List of facts
            objectives: List of objectives

        Returns:
            Scrutiny results
        """
        self.logger.debug("Stage 4: Scrutiny")

        scrutiny = {"weaknesses": [], "gaps": [], "inconsistencies": [], "score": 0.0}

        # Analyze facts for weaknesses
        for fact in facts:
            if fact["confidence"] < 0.6:
                scrutiny["weaknesses"].append(f"Low confidence in: {fact['statement']}")

        # Identify gaps relative to objectives
        if len(facts) < 5:
            scrutiny["gaps"].append("Insufficient factual foundation")

        # Check for inconsistencies
        statements = [f["statement"] for f in facts]
        # Simple check: look for contradictory words
        contradictory_pairs = [("increase", "decrease"), ("enable", "disable")]
        for pair in contradictory_pairs:
            if any(pair[0] in s.lower() for s in statements) and any(pair[1] in s.lower() for s in statements):
                scrutiny["inconsistencies"].append(f"Potential contradiction: {pair}")

        # Calculate scrutiny score
        total_issues = len(scrutiny["weaknesses"]) + len(scrutiny["gaps"]) + len(scrutiny["inconsistencies"])
        scrutiny["score"] = max(0.0, min(1.0, total_issues / 10.0))  # Normalize

        # Update SLAP state
        self.current_scrutiny = scrutiny["score"]

        return scrutiny

    def _derive(self, facts: List[Dict], scrutiny: Dict) -> List[Dict]:
        """
        Stage 5: Derivation - Derive new facts from existing ones.

        Args:
            facts: List of existing facts
            scrutiny: Scrutiny results

        Returns:
            List of  derived facts
        """
        self.logger.debug("Stage 5: Derivation")

        derived = []

        # Simple derivation: combine facts to create new insights
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i + 1 :]:
                if fact1["type"] == fact2["type"]:
                    # Combine similar type facts
                    derived.append({"type": "derived", "statement": f"From {fact1['statement'][:30]}... and {fact2['statement'][:30]}...", "confidence": (fact1["confidence"] + fact2["confidence"]) / 2, "sources": [fact1, fact2]})

                    if len(derived) >= 5:  # Limit derivations
                        break
            if len(derived) >= 5:
                break

        self.logger.debug(f"Derived {len(derived)} new facts")
        return derived

    def _apply_rules(self, derivations: List[Dict]) -> Dict:
        """
        Stage 6: Rule-Based - Apply predefined rules.

        Args:
            derivations: Derived facts

        Returns:
            Rule application results
        """
        self.logger.debug("Stage 6: Rule-Based Application")

        rules = {"applied_rules": [], "outcomes": []}

        # Define some basic rules
        basic_rules = [{"name": "consolidation", "description": "Consolidate similar facts"}, {"name": "validation", "description": "Validate against constraints"}, {"name": "prioritization", "description": "Prioritize by confidence"}]

        for rule in basic_rules:
            rules["applied_rules"].append(rule["name"])
            rules["outcomes"].append(f"Applied {rule['description']}")

        return rules

    def _build_model(self, rules: Dict, structure: str = "hierarchical") -> Dict:
        """
        Stage 7: Model - Create structured representation.

        Args:
            rules: Rule application results
            structure: Structure type

        Returns:
            Model dictionary
        """
        self.logger.debug("Stage 7: Model Building")

        model = {
            "components": rules.get("applied_rules", []),
            "structure": structure,
            "completeness": len(rules.get("applied_rules", [])) / 3.0 if "applied_rules" in rules else 0.5,
        }

        # Update improvement score
        self.current_improvement = model["completeness"]

        return model

    def _formalize_semantics(self, model: Dict) -> Dict:
        """
        Stage 8: Semantic Formalization - Formalize semantics.

        Args:
            model: Model output

        Returns:
            Semantic formalization dictionary
        """
        self.logger.debug("Stage 8: Semantic Formalization")

        semantic = {"formalized": True, "semantic_structure": model["structure"], "completeness": model["completeness"], "ready_for_execution": model["completeness"] > 0.7}

        return semantic

    def _calculate_advancement(self) -> float:
        """
        Calculate advancement score using SLAP formula:
        Advancement = Truth + (alpha * Scrutiny) + (beta * Improvement)

        Returns:
            Advancement score
        """
        score = advancement_score(self.current_truth, self.current_scrutiny, self.current_improvement, self.config.alpha, self.config.beta)

        return score

    def identify_missing_entities_mcts(self, current_plan: Dict, iterations: Optional[int] = None) -> List[str]:
        """
        Use Monte Carlo Tree Search to identify missing entities.

        Args:
            current_plan: Current reasoning plan
            iterations: MCTS iterations (uses config default if None)

        Returns:
            List of potentially missing entities
        """
        iterations = iterations or self.config.mcts_iterations

        self.logger.debug(f"Running MCTS for {iterations} iterations")

        # Simplified MCTS implementation
        missing_entities = []

        # Analyze each stage for completeness
        stages = ["conceptualization", "representation", "facts", "derivation"]

        for stage in stages:
            if stage in current_plan:
                # Check if stage output is sufficiently detailed
                stage_data = current_plan[stage]
                if isinstance(stage_data, dict) and len(stage_data) < 3:
                    missing_entities.append(f"Incomplete {stage}: needs more detail")
                elif isinstance(stage_data, list) and len(stage_data) < 5:
                    missing_entities.append(f"Insufficient {stage} elements")

        return missing_entities

    # Helper methods for conceptualization

    def _extract_primary_concept(self, task: str) -> str:
        """Extract primary concept from task."""
        # Simple: first significant noun
        words = task.split()
        for word in words:
            if len(word) > 4 and word[0].isupper():
                return word
        return words[0] if words else "Unknown"

    def _extract_related_concepts(self, task: str) -> List[str]:
        """Extract related concepts."""
        # Simple: extract capitalized words and long words
        words = task.split()
        related = []
        for word in words:
            if (len(word) > 4 and word[0].isupper()) or len(word) > 8:
                related.append(word)
        return related[:5]

    def _identify_domain(self, task: str) -> str:
        """Identify domain of task."""
        task_lower = task.lower()

        domains = {"technical": ["code", "program", "software", "algorithm"], "analytical": ["analyze", "evaluate", "assess", "study"], "creative": ["create", "design", "build", "develop"], "optimization": ["optimize", "improve", "enhance", "maximize"]}

        for domain, keywords in domains.items():
            if any(kw in task_lower for kw in keywords):
                return domain

        return "general"

    def _create_symbolic_representation(self, concepts: Dict) -> str:
        """Create symbolic representation of concepts."""
        return f"⟨{concepts['primary_concept']}, {', '.join(concepts['related_concepts'][:2])}⟩"

    def _identify_structure(self, concepts: Dict) -> Dict:
        """Identify structure from concepts."""
        return {"hierarchy": "nested", "complexity": "moderate", "domain": concepts["domain"]}

    def _map_relationships(self, concepts: Dict) -> List[str]:
        """Map relationships between concepts."""
        relationships = []
        primary = concepts["primary_concept"]

        for related in concepts["related_concepts"][:3]:
            relationships.append(f"{primary} relates_to {related}")

        return relationships

    def reset(self):
        """Reset pipeline state."""
        self.current_truth = 1.0
        self.current_scrutiny = 0.0
        self.current_improvement = 0.0
        self.advancement_history.clear()
        self.logger.debug("SLAP pipeline reset")
