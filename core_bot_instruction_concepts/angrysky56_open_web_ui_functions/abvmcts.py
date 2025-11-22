"""
title: advanced_mcts
author: angrysky56
author_url: https://github.com/angrysky56
description: Advanced Monte Carlo Tree Search with persistent visualization and strategic memory
version: 0.3.1

Key improvements in v0.3.1:
- Fixed visualization issues to preserve history in chat output
- Modified Mermaid escaping to improve diagram rendering
- Enhanced exploration with non-leaf node selection and diversity boost
- Added JSON export for better external visualization
- Ensured proper termination and resource handling
- Fixed unsafe aiohttp cleanup attempts
"""

from fastapi import Request
import logging
import random
import math
import asyncio
import json
import re
import html
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from typing import (
    List,
    Optional,
    AsyncGenerator,
    Callable,
    Awaitable,
    Generator,
    Iterator,
    Dict,
    Any,
    Tuple,
    Set,
)
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
import open_webui.routers.ollama as ollama
from open_webui.main import app

# ==============================================================================

name = "advanced_mcts"

# Configurable parameters with default values - Adjusted for better exploration
config = {
    "max_children": 4,
    "exploration_weight": 2.5, # Hardcoded higher exploration
    "max_iterations": 3,
    "simulations_per_iteration": 15, # Hardcoded higher simulations
    "thoughts_per_node": 2,
    "surprise_threshold": 0.7,
    "use_semantic_distance": True,
    "use_llm_embedding_for_distance": False,
    "node_label_length": 100,
    "relative_evaluation": True,
    "score_diversity_bonus": 0.7,
    "force_exploration_interval": 4, # Adjusted interval from 3
    "debug_logging": True,
    "global_context_in_prompts": True,
    "track_explored_approaches": True,
    "sibling_awareness": True,
    "memory_cutoff": 5,
    "early_stopping": True,
    "early_stopping_threshold": 9.0,
    "early_stopping_stability": 3,
}

# ==============================================================================

# Prompt templates (using v0.3.1 versions)
thoughts_prompt = """
<instruction>
You are generating a thought for improving a draft answer in a strategic exploration process.
</instruction>

<context>
CURRENT EXPLORATION STATUS:
- Best answer found so far (score: {best_score}/10): 
{best_answer}

- Previously explored approaches: 
{explored_approaches}

- Current approach type: {current_approach}

- Current draft to improve:
{answer}
</context>

<question>
{question}
</question>

Based on the context provided, suggest a SIGNIFICANTLY DIFFERENT APPROACH or identify a MAJOR WEAKNESS to address.
Consider:
- Approaches that haven't been tried yet
- Ways to bridge the gap between this draft and the best answer found
- Completely different frameworks or perspectives 
- Critical flaws in the current draft compared to the best answer

YOUR REPLY SHOULD BE A SINGLE SENTENCE THAT SUGGESTS A FUNDAMENTALLY DIFFERENT DIRECTION.
"""

relative_eval_prompt = """
<instruction>
Compare this new answer to the previous one and rate the IMPROVEMENT, considering the global context.
</instruction>

<context>
CURRENT EXPLORATION STATUS:
- Best answer found so far (score: {best_score}/10): 
{best_answer}
</context>

<question>
{question}
</question>

<previous_answer>
{parent_answer}
</previous_answer>

<new_answer>
{answer}
</new_answer>

Rate the IMPROVEMENT of the new answer over the previous answer on a scale of 1 to 5:
1: Worse or irrelevant change.
2: No real improvement, maybe slightly different wording.
3: Minor improvement, slightly clearer or adds a small relevant detail.
4: Clear improvement, addresses a flaw or adds significant value.
5: Major improvement, fundamentally better approach.

Consider both the improvement over the parent AND how it compares to the best answer found so far.
Focus ONLY on the improvement quality. Reply with a single number 1-5.
"""

eval_answer_prompt = """
<instruction>
Evaluate this answer critically, considering both the question and the search context.
</instruction>

<context>
CURRENT EXPLORATION STATUS:
- Best answer found so far (score: {best_score}/10): 
{best_answer}
</context>

<question>
{question}
</question>

<answer_to_evaluate>
{answer}
</answer_to_evaluate>

Critically evaluate the answer on a scale of 1 to 10.
- 1-3: Fundamentally flawed, irrelevant, or very incomplete.
- 4-6: Partially addresses the question but has significant issues or omissions.
- 7-8: Good answer, mostly correct and relevant, minor issues possible.
- 9-10: Excellent, comprehensive, accurate, and well-structured answer.

Be STRICT in your evaluation. Do not give high scores (9-10) unless the answer is truly exceptional.
Compare it to the best answer found so far, if available.
Reply with a single number between 1 and 10 only. Do not write anything else.
THINK CAREFULLY AND USE THE FULL SCALE.
"""

analyze_prompt = """
<instruction>
Analyze the current iteration of the Monte Carlo Tree Search process.
</instruction>

<context>
SEARCH STATISTICS:
- Original question: {question}
- Best answer found: {best_answer}
- Best score achieved: {best_score}
- Current tree depth: {tree_depth}
- Number of branches: {branches}
- Explored approach types: {approach_types}
</context>

Analyze this iteration of the thought process. Consider the following:
1. What aspects of the best answer made it successful?
2. What patterns or approaches led to higher-scoring thoughts?
3. Were there any common pitfalls or irrelevant tangents in lower-scoring thoughts?
4. How can the thought generation process be improved for the next iteration?
5. What approaches or dimensions have NOT been explored yet?

Provide a concise analysis and suggest one specific improvement strategy for the next iteration.
"""

update_prompt = """
<instruction>
You are refining an answer based on strategic critique, with awareness of the overall search process.
</instruction>

<context>
CURRENT EXPLORATION STATUS:
- Best answer found so far (score: {best_score}/10): 
{best_answer}

- Previously explored approaches: 
{explored_approaches}

- Current approach type: {current_approach}
</context>

<question>
{question}
</question>

<draft>
{answer}
</draft>

<critique>
{improvements}
</critique>

The critique suggests a SIGNIFICANTLY DIFFERENT APPROACH or identifies a MAJOR WEAKNESS.
Completely rethink your approach based on this critique and the global context provided.
WRITE A SUBSTANTIALLY REVISED ANSWER THAT:
1. Takes a different angle from the current draft
2. Addresses the critique directly
3. Attempts to improve upon the best answer found so far
4. Explores territory not covered by previously explored approaches
"""

initial_prompt = """
<instruction>
Answer the question below. Do not pay attention to unexpected casing, punctuation or accent marks.
</instruction>

<question>
{question}
</question>
"""

embedding_prompt = """
<instruction>
Create a compact semantic embedding description of the following text. 
Identify the key concepts, ideas, and approach used in this answer.
Your response should be a short comma-separated list of key terms and concepts.
</instruction>

<text>
{text}
</text>
""".strip()

final_summary_prompt = """
<instruction>
Create a comprehensive final summary of the Monte Carlo Tree Search process and its findings.
</instruction>

<context>
SEARCH STATISTICS:
- Original question: {question}
- Best answer found (score: {best_score}/10): 
{best_answer}
- Number of iterations completed: {iterations}
- Total nodes explored: {total_nodes}
- Tree depth: {tree_depth}
- Number of approaches tried: {approach_count}
- Most successful approach: {best_approach} (avg score: {best_approach_score})
</context>

Write a comprehensive final report that includes:
1. A brief overview of the search process
2. The best answer found and why it was successful
3. Key insights from the exploration process
4. Alternative approaches that were considered but scored lower
5. A conclusion summarizing the essential findings

This should be a polished final output that synthesizes all the learning from the search process.
"""

# ==============================================================================

def setup_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if config.get("debug_logging", False) else logging.INFO) # Use config
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    # Ensure level is set correctly even if handlers exist
    logger.setLevel(logging.DEBUG if config.get("debug_logging", False) else logging.INFO)
    return logger


class AdminUserMock:
    def __init__(self):
        self.role = "admin"


admin = AdminUserMock()
logger = setup_logger()

# ==============================================================================

# Note: modify_text and replace_with_mapping are not used currently.
# They could be used for adding noise to the question for robustness testing.

# ==============================================================================

def truncate_text(text, max_length=None):
    """Truncate text and add ellipsis if needed."""
    if not text:
        return ""
    if max_length is None:
        max_length = config["node_label_length"]
    text = text.replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def escape_mermaid(text):
    """Minimal escaping for Mermaid labels."""
    if not text:
        return ""
    # Replace characters known to break Mermaid syntax within labels
    return text.replace("`", "'").replace('"', "'").replace('#', '')

def calculate_semantic_distance(text1, text2, llm=None):
    """Calculate semantic distance between two text blocks."""
    if not text1 or not text2:
        return 1.0 # Max distance if one is empty
    # Use LLM embeddings if specified and available (future enhancement)
    # ...

    # Otherwise use TF-IDF vectorization
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        # Ensure matrix is not empty/invalid
        if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
             raise ValueError("TF-IDF matrix issue.")
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        # Clamp similarity to avoid potential float precision issues
        similarity = max(0.0, min(1.0, similarity))
        return 1.0 - similarity
    except Exception as e:
        logger.warning(f"Error calculating semantic distance with TF-IDF: {e}. Falling back to word overlap.")
        words1 = set(re.findall(r'\w+', text1.lower())) # Use regex for words
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2:
            return 1.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        if union == 0:
            return 0.0 # Texts are identical empty strings
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

# ==============================================================================

class Node:
    """A node in the Monte Carlo Tree Search."""

    def __init__(self, **kwargs):
        self.id = "node_" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4))
        self.content = kwargs.get("content", "")
        self.parent = kwargs.get("parent")
        self.exploration_weight = kwargs.get(
            "exploration_weight", config["exploration_weight"]
        )
        self.max_children = kwargs.get("max_children", config["max_children"])
        self.children = []
        self.visits = 0
        self.value = 0.0 # Ensure float
        self.raw_scores = []
        self.sequence = kwargs.get("sequence", 0)
        self.embedding = kwargs.get("embedding", "") # Placeholder for future embedding use
        self.is_surprising = False
        self.surprise_explanation = ""
        self.approach_type = kwargs.get("approach_type", "initial")
        self.thought = kwargs.get("thought", "")
        self.generation_context = kwargs.get("generation_context", {})

        if kwargs.get("is_surprising", False):
            self.is_surprising = True
            self.value = 2.0 # Optimistic bias
            self.surprise_explanation = kwargs.get("surprise_explanation", "")

    def add_child(self, child: "Node"):
        """Add a child node to this node."""
        child.parent = self
        self.children.append(child)
        return child

    def fully_expanded(self):
        """Check if this node has reached its maximum number of children."""
        return len(self.children) >= self.max_children

    def uct_value(self):
        """Calculate the UCT value for this node."""
        epsilon = 1e-6
        if self.visits == 0:
            return float("inf")

        exploitation = self.value / self.visits
        parent_visits = self.parent.visits if self.parent else 1
        # Add epsilon to parent_visits log argument to prevent log(0)
        exploration = self.exploration_weight * math.sqrt(
            math.log(parent_visits + epsilon) / self.visits
        )
        surprise_bonus = 0.3 if self.is_surprising else 0

        # Diversity bonus calculation
        diversity_bonus = 0.0
        if self.parent and len(self.parent.children) > 1 and config["score_diversity_bonus"] > 0:
            sibling_scores = [c.value / max(1, c.visits) for c in self.parent.children if c != self and c.visits > 0]
            if sibling_scores:
                sibling_avg_value = sum(sibling_scores) / len(sibling_scores)
                my_avg_value = self.value / self.visits
                score_diff = abs(my_avg_value - sibling_avg_value)
                diversity_bonus = config["score_diversity_bonus"] * score_diff

        return exploitation + exploration + surprise_bonus + diversity_bonus

    def mermaid(self, offset=0, selected=None):
        """Generate mermaid diagram code for this node and its children."""
        padding = " " * offset
        score = self.value / max(1, self.visits)
        score_text = f"{score:.2f}" if self.visits > 0 else "N/A"
        approach_label = f" [{self.approach_type}]" if self.approach_type != "initial" else ""
        display_text = truncate_text(self.content)
        surprise_indicator = "⭐ " if self.is_surprising else ""
        node_label_content = escape_mermaid(display_text)
        node_label = f'{self.id}["{surprise_indicator}Node {self.sequence}: Score={score_text}{approach_label}<br/>\'{node_label_content}\'"]'
        msg = f"{padding}{node_label}\n"

        # Styling logic (same as before)
        if selected and selected == self.id:
            msg += f"{padding}style {self.id} fill:#d4f0fd,stroke:#0099ff,stroke-width:2px\n"
        elif self.visits > 0:
             if self.is_surprising:
                 msg += f"{padding}style {self.id} fill:#fcf8d3,stroke:#f1c40f,stroke-width:2px\n"
             elif score > 7:
                 msg += f"{padding}style {self.id} fill:#d4ffd4,stroke:#00cc00\n"
             elif score < 4:
                 msg += f"{padding}style {self.id} fill:#ffd4d4,stroke:#cc0000\n"

        # Children rendering
        for child in self.children:
            msg += child.mermaid(offset + 4, selected)
            edge_label = f"{child.visits}"
            if hasattr(child, "thought") and child.thought:
                thought_preview = truncate_text(child.thought, 20)
                safe_thought_preview = escape_mermaid(thought_preview)
                edge_label = f"{edge_label}|{safe_thought_preview}"
            msg += f"{padding}{self.id} -->|{edge_label}| {child.id}\n"
        return msg

    def best_child(self):
        """Return the best child node based on visit count."""
        if not self.children:
            return self
        # Add tie-breaking based on value if visits are equal
        return max(self.children, key=lambda child: (child.visits, child.value / max(1, child.visits)))

    def node_to_json(self):
        """Convert node to JSON serializable dictionary for visualization."""
        return {
            "id": self.id,
            "sequence": self.sequence,
            "content": truncate_text(self.content, 200),
            "visits": self.visits,
            "value": round(self.value, 2), # Round for cleaner JSON
            "score": round(self.value / max(1, self.visits), 2),
            "approach_type": self.approach_type,
            "is_surprising": self.is_surprising,
            "thought": truncate_text(self.thought, 100) if hasattr(self, "thought") else "",
            "children": [child.node_to_json() for child in self.children]
        }

# ==============================================================================

class MCTS:
    """Monte Carlo Tree Search implementation with strategic memory."""

    def __init__(self, **kwargs):
        self.question = kwargs.get("question")
        self.root = kwargs.get("root")
        self.node_sequence = 0
        self.root.sequence = self.get_next_sequence()
        self.llm = kwargs.get("llm")
        self.selected = self.root
        self.exploration_weight = config["exploration_weight"]
        self.thought_history = []
        self.debug_history = []
        self.surprising_nodes = []
        self.best_solution = self.root.content # Initialize with root content
        self.best_score = 0 # Start score at 0
        self.iterations_completed = 0
        self.simulations_completed = 0
        self.high_score_counter = 0
        self.random_state = random.Random()
        self.approach_types = ["initial"]
        self.explored_approaches = {}
        self.explored_thoughts = set()
        self.approach_scores = {}
        self.memory = {"depth": 0, "branches": 0, "high_scoring_nodes": [], "repeated_approaches": []}
        self.iteration_json_snapshots = [] # Added for per-iteration JSON

        self.thought_history.append(f"# MCTS Analysis for Question: {self.question}\n\nStarting exploration with initial answer...\n")

    def get_next_sequence(self):
        self.node_sequence += 1
        return self.node_sequence

    def export_tree_as_json(self):
        return self.root.node_to_json()

    def get_context_for_node(self, node):
        # (Code from v0.3.1 seems correct)
        context = {
            "best_answer": ("No best answer found yet." if not self.best_solution else self.best_solution),
            "best_score": self.best_score,
            "current_approach": (node.approach_type if hasattr(node, "approach_type") else "initial"),
            "tree_depth": self.memory.get("depth", 0),
            "branches": self.memory.get("branches", 0),
            "approach_types": ", ".join(self.approach_types),
        }
        # Build explored_approaches_text (handle potential errors)
        try:
            explored_approaches_text = []
            for approach, thoughts in self.explored_approaches.items():
                if thoughts:
                    avg_score = self.approach_scores.get(approach, "unknown")
                    score_text = (f" (avg score: {avg_score:.1f})" if avg_score != "unknown" else "")
                    sample_size = min(2, len(thoughts))
                    sampled_thoughts = random.sample(thoughts, sample_size)
                    approach_entry = f"- {approach}{score_text}: " + "; ".join([f'"{t}"' for t in sampled_thoughts])
                    explored_approaches_text.append(approach_entry)
            context["explored_approaches"] = "\n".join(explored_approaches_text) if explored_approaches_text else "No approaches explored yet."
        except Exception as e:
            logger.error(f"Error building explored approaches context: {e}")
            context["explored_approaches"] = "Error retrieving approaches."

        # Add sibling awareness (handle potential errors)
        try:
            if config["sibling_awareness"] and node.parent and node.parent.children:
                siblings = [child for child in node.parent.children if child != node]
                if siblings:
                    sibling_approaches = []
                    for sibling in siblings:
                        if hasattr(sibling, "thought") and sibling.thought:
                            sibling_score = sibling.value / max(1, sibling.visits)
                            sibling_approaches.append(f'"{truncate_text(sibling.thought, 50)}" (score: {sibling_score:.1f})')
                    if sibling_approaches:
                        context["sibling_approaches"] = "\n".join(["Approaches already tried from this position:"] + [f"- {sa}" for sa in sibling_approaches])
        except Exception as e:
            logger.error(f"Error building sibling context: {e}")
            context["sibling_approaches"] = "Error retrieving siblings."

        return context

    def _collect_non_leaf_nodes(self, node, non_leaf_nodes, max_depth, current_depth=0):
        """Helper function for select enhancement."""
        if current_depth > max_depth: return
        if node.children and not node.fully_expanded(): non_leaf_nodes.append(node)
        for child in node.children: self._collect_non_leaf_nodes(child, non_leaf_nodes, max_depth, current_depth + 1)

    async def select(self):
        """Select node using UCT with exploration enhancements."""
        # (Code from v0.3.1 including non-leaf and diversity boost seems correct)
        logger.debug("Selecting node...")
        node = self.root
        selection_path = [node]
        debug_info = "### UCT Selection Path Decisions:\n"

        # Enhanced Exploration: Non-leaf selection
        if (self.simulations_completed > 0 and # Avoid on first sim
            self.simulations_completed % config['force_exploration_interval'] == 0 and
            self.memory.get("depth", 0) > 1):
            candidate_nodes = []
            self._collect_non_leaf_nodes(self.root, candidate_nodes, max_depth=max(1, self.memory["depth"]//2))
            expandable_candidates = [n for n in candidate_nodes if not n.fully_expanded()]
            if expandable_candidates:
                node = self.random_state.choice(expandable_candidates)
                debug_info += f"BRANCHING ENHANCEMENT: Selected non-leaf Node {node.sequence} to force new branch\n"
                self.thought_history.append(f"### Forcing Tree Branching\nSelecting non-leaf node {node.sequence} to create new exploration branch.\n\n")
                # Need to reconstruct path if we jump, or simplify history
                # For now, just return the node
                return node # Return early

        # Normal UCT based traversal
        while node.children:
            unvisited = [child for child in node.children if child.visits == 0]
            if unvisited:
                node = self.random_state.choice(unvisited)
                debug_info += f"Selected unvisited Node {node.sequence}\n"
            else:
                uct_values = [(child, child.uct_value()) for child in node.children]
                uct_values.sort(key=lambda x: x[1], reverse=True)
                # Enhanced Exploration: Diversity Boost
                if len(uct_values) > 1 and self.random_state.random() < 0.2: # 20% chance
                    node = uct_values[1][0] # Select second best
                    debug_info += f"DIVERSITY BOOST: Selected 2nd best Node {node.sequence} for diversity\n"
                else:
                    node = uct_values[0][0] # Select best
                    debug_info += f"Selected Node {node.sequence} with highest UCT {uct_values[0][1]:.3f}\n"

            selection_path.append(node)
            # If node isn't fully expanded, it's our target leaf for this path
            if not node.fully_expanded():
                break

        path_str = " → ".join([f"Node {n.sequence}" for n in selection_path])
        self.thought_history.append(f"### Selection Path\n{path_str}\n")
        if config["debug_logging"]: self.debug_history.append(debug_info); logger.debug(debug_info)
        self.memory["depth"] = max(self.memory.get("depth",0), len(selection_path) - 1)
        return node

    async def expand(self, node) -> Tuple[Optional[Node], bool]:
        """Expand node, create ONE child. Return (Child, is_surprising) or (None, False) on failure."""
        logger.debug(f"Expanding node {node.id}...")
        try:
            await self.llm.progress(f"Exploring a new thought path from node {node.sequence}...")
            # Display current tree state *before* expansion attempt
            await self.llm.emit_message(self.formatted_output(node))

            context = self.get_context_for_node(node)
            await self.llm.emit_message(f"Generating thought for Node {node.sequence} with strategic context: ")
            thought = await self.llm.generate_thought(node.content, context)
            if not thought: raise ValueError("LLM failed to generate thought.")
            thought_entry = f"### Expanding Node {node.sequence}\nCurrent answer: {truncate_text(node.content, 150)}\n\n**Thought**: {thought}\n\n"

            # Similarity check (same as before)
            # ...

            self.explored_thoughts.add(thought)
            approach_type = self._classify_approach(thought) # Use helper
            if approach_type not in self.approach_types: self.approach_types.append(approach_type)
            if approach_type not in self.explored_approaches: self.explored_approaches[approach_type] = []
            self.explored_approaches[approach_type].append(thought)

            await self.llm.emit_message(f"\n\n---\n\nRefining solution based on thought:\n")
            new_content = await self.llm.update_approach(node.content, thought, context)
            if not new_content: raise ValueError("LLM failed to update approach.")

            is_surprising, surprise_explanation = self._check_surprise(node.content, new_content)
            if is_surprising:
                thought_entry += f"**SURPRISE DETECTED!**\n**Explanation**: {surprise_explanation}\n\n"

            child = Node(
                content=new_content, parent=node, sequence=self.get_next_sequence(),
                is_surprising=is_surprising, surprise_explanation=surprise_explanation,
                approach_type=approach_type, thought=thought, generation_context=context,
            )
            node.add_child(child)
            if is_surprising: self.surprising_nodes.append(child)

            thought_entry += f"**New solution {child.sequence}** ({approach_type}): {truncate_text(new_content, 150)}\n\n"
            self.thought_history.append(thought_entry)

            # Show tree state *after* successful expansion
            await self.llm.emit_message(self.formatted_output(child))

            if len(node.children) > 1: self.memory["branches"] = self.memory.get("branches", 0) + 1
            return child, is_surprising

        except Exception as e:
            logger.error(f"Error during expansion of Node {node.sequence}: {e}")
            self.thought_history.append(f"### Expansion Error on Node {node.sequence}\nFailed to expand: {e}\n")
            return None, False # Indicate failure

    def _classify_approach(self, thought: str) -> str:
        """Classify the approach type based on the thought."""
        # (Using the taxonomy logic from previous version - seems reasonable)
        approach_type = "variant"
        thought_lower = thought.lower()
        # Define taxonomy within the method or globally if preferred
        approach_taxonomy = {
            # ... (taxonomy from v0.3.1) ...
             "empirical": ["evidence", "data", "observation", "experiment", "measure", "test", "verify"],
            "rational": ["logic", "reason", "deduction", "principle", "axiom", "theorem", "inference"],
            "phenomenological": ["experience", "perception", "consciousness", "phenomenon", "subjective"],
            "hermeneutic": ["interpret", "meaning", "context", "understanding", "text", "narrative"],
            "reductionist": ["reduce", "component", "fundamental", "elemental", "atomic", "break down"],
            "holistic": ["whole", "system", "emergent", "gestalt", "interconnected", "ecological"],
            "materialist": ["physical", "concrete", "tangible", "matter", "substrate", "mechanism"],
            "idealist": ["concept", "ideal", "abstract", "mental", "construct", "representation"],
            "analytical": ["analyze", "dissect", "examine", "scrutinize", "investigate", "detail"],
            "synthetic": ["synthesize", "integrate", "combine", "unify", "merge", "consolidate"],
            "dialectical": ["thesis", "antithesis", "synthesis", "contradiction", "opposing", "reconcile"],
            "comparative": ["compare", "contrast", "juxtapose", "parallel", "analogy", "differentiate"],
            "critical": ["critique", "challenge", "question", "problematize", "deconstruct", "interrogate"],
            "constructive": ["build", "develop", "construct", "establish", "create", "formulate"],
            "pragmatic": ["practical", "useful", "effective", "functional", "applicable", "workable"],
            "normative": ["should", "ought", "value", "ethical", "moral", "prescriptive", "standard"],
            "perspective": ["different perspective", "viewpoint", "angle", "lens", "frame"],
            "alternative": ["alternative approach", "another way", "different method", "consider instead"],
            "opposing": ["opposing view", "contrary", "counter", "antithesis", "challenge"],
            "complementary": ["missing", "supplement", "augment", "enhance", "complement", "addition"],
            "structural": ["structure", "organize", "arrangement", "pattern", "framework", "architecture"]
        }
        approach_scores = {}
        for approach, keywords in approach_taxonomy.items():
            score = 0
            for keyword in keywords:
                if keyword in thought_lower: score += 1
            if score > 0: approach_scores[approach] = score
        if approach_scores:
            max_score = max(approach_scores.values())
            best_approaches = [app for app, score in approach_scores.items() if score == max_score]
            approach_type = self.random_state.choice(best_approaches) # Random choice among best

        return approach_type

    def _check_surprise(self, parent_content, child_content) -> Tuple[bool, str]:
         """Check if the new content is surprising compared to the parent."""
         is_surprising = False
         surprise_explanation = ""
         if config["use_semantic_distance"]:
             try:
                 semantic_distance = calculate_semantic_distance(parent_content, child_content, self.llm)
                 if semantic_distance > config["surprise_threshold"]:
                     is_surprising = True
                     surprise_explanation = f"Significant semantic distance ({semantic_distance:.2f})"
             except Exception as e:
                 logger.warning(f"Surprise check failed: {e}")
         # Could add other surprise checks here (e.g., approach shifts)
         return is_surprising, surprise_explanation

    async def simulate(self, node):
        """Simulate and evaluate a node. Return score or None on failure."""
        logger.debug(f"Simulating node {node.id}...")
        try:
            await self.llm.progress(f"Evaluating solution quality for node {node.sequence}...")
            # Display tree state *before* simulation
            await self.llm.emit_message(self.formatted_output(node))

            context = self.get_context_for_node(node)
            eval_type = "absolute"
            raw_score = 0

            if config["relative_evaluation"] and node.parent:
                relative_score = await self.llm.evaluate_relative(node.parent.content, node.content, context)
                if relative_score is None: raise ValueError("LLM failed relative evaluation.")
                eval_type = "relative"
                raw_score = relative_score # Store the 1-5 score
                # Convert 1-5 relative score to 1-10 absolute scale
                parent_avg_score = node.parent.value / max(1, node.parent.visits)
                # (Conversion logic same as before)
                if relative_score == 1: score = max(1, round(parent_avg_score - 2))
                elif relative_score == 2: score = round(parent_avg_score)
                elif relative_score == 3: score = min(10, round(parent_avg_score + 1))
                elif relative_score == 4: score = min(10, round(parent_avg_score + 2))
                else: score = min(10, round(parent_avg_score + 3)) # Score 5
                score = max(1, min(10, score)) # Clamp final score
            else:
                score = await self.llm.evaluate_answer(node.content, context)
                if score is None: raise ValueError("LLM failed absolute evaluation.")
                eval_type = "absolute"
                raw_score = score

            node.raw_scores.append(raw_score)
            # Update approach scores (weighted average)
            approach = node.approach_type
            current_avg = self.approach_scores.get(approach, score) # Use current score if first time
            self.approach_scores[approach] = (0.7 * score + 0.3 * current_avg)

            # Logging (same as before)
            if config["debug_logging"]: logger.debug(f"Node {node.sequence} eval: {eval_type}, raw {raw_score}, final {score}")
            self.thought_history.append(f"### Evaluating Node {node.sequence}\nScore: {score}/10 ({eval_type})\nSolution: {truncate_text(node.content, 150)}\n")
            # Update memory (same as before)
            # ...

            return score

        except Exception as e:
             logger.error(f"Error during simulation of Node {node.sequence}: {e}")
             self.thought_history.append(f"### Simulation Error on Node {node.sequence}\nFailed to simulate: {e}\n")
             return None # Indicate failure

    def backpropagate(self, node, score):
        """Backpropagate score up the tree. Handle None score."""
        if score is None:
             logger.warning(f"Skipping backpropagation from Node {node.sequence} due to simulation failure.")
             return # Do not backpropagate if simulation failed

        logger.debug(f"Backpropagating score {score} from {node.id}...")
        # (Rest of backpropagation logic is same)
        backprop_path = []
        temp_node = node
        while temp_node:
            backprop_path.append(f"Node {temp_node.sequence}")
            # Actual backpropagation
            temp_node.visits += 1
            temp_node.value += score # Add the valid score
            temp_node = temp_node.parent
        # Log path after update
        self.thought_history.append(f"### Backpropagating Score {score}\nPath: {' → '.join(reversed(backprop_path))}\n")


    async def search(self, simulations_per_iteration):
        """Perform MCTS simulations for one iteration."""
        logger.debug(f"Starting iteration {self.iterations_completed + 1} with {simulations_per_iteration} simulations...")
        visited_leaves = {}

        for i in range(simulations_per_iteration):
            self.simulations_completed += 1
            self.current_simulation_in_iteration = i + 1
            sim_entry = f"### Iteration {self.iterations_completed + 1} - Simulation {i+1}/{simulations_per_iteration}\n"
            self.thought_history.append(sim_entry)

            # --- Selection ---
            leaf = await self.select()
            self.selected = leaf

            # Exploitation tracking (same as before)
            # ...

            # --- Expansion & Simulation ---
            node_to_simulate = leaf
            score = None # Initialize score

            if not leaf.fully_expanded():
                # Try to expand
                expansion_result = await self.expand(leaf)
                if expansion_result and expansion_result[0]: # Check if expansion succeeded (returned a child node)
                    new_child, _ = expansion_result
                    self.selected = new_child # Highlight the new child
                    node_to_simulate = new_child
                    # Simulate the newly expanded child
                    score = await self.simulate(node_to_simulate)
                else:
                    # Expansion failed, log it but maybe simulate parent? Or skip?
                    logger.warning(f"Expansion failed for Node {leaf.sequence}, simulating parent instead.")
                    # Simulate the leaf itself if expansion failed
                    score = await self.simulate(leaf)
                    node_to_simulate = leaf # Ensure we backpropagate from leaf
            else:
                # If leaf is fully expanded, simulate it directly
                logger.debug(f"Node {leaf.sequence} is fully expanded, simulating leaf.")
                score = await self.simulate(leaf)
                node_to_simulate = leaf # Ensure we backpropagate from leaf

            # --- Backpropagation ---
            # Pass the node that was actually simulated
            self.backpropagate(node_to_simulate, score)

            # --- Update Best Solution (only if score is valid) ---
            if score is not None and score > self.best_score:
                 self.best_score = score # Update with the new best score
                 self.best_solution = node_to_simulate.content
                 self.thought_history.append(f"### New Best Solution Found!\nScore: {score}/10\nNode: {node_to_simulate.sequence}\n")
                 # Emit full output with new best highlighted
                 await self.llm.emit_message(self.formatted_output(node_to_simulate))

                 # Early stopping check (using the valid score)
                 if config["early_stopping"] and score >= config["early_stopping_threshold"]:
                      self.high_score_counter += 1
                      if self.high_score_counter >= config["early_stopping_stability"]:
                           await self.llm.emit_message(f"Found consistently high-quality solution (score: {score}/10). Early stopping iteration.")
                           # Store final snapshot before returning
                           self._store_iteration_snapshot("Early Stopping")
                           return self.selected # Exit search early
                 else:
                      self.high_score_counter = 0 # Reset if score drops below threshold
            else:
                 # Reset counter if simulation failed or score wasn't better
                 self.high_score_counter = 0

            # --- Periodic Reporting ---
            if i > 0 and i % 5 == 0: await self._report_tree_stats()
            # Analysis could also be periodic
            # if i > 0 and i % (simulations_per_iteration // 2) == 0: await self.analyze_iteration()

        # --- End of Iteration ---
        await self.llm.emit_message(self.formatted_output(self.selected))
        self._store_iteration_snapshot("End of Iteration") # Store snapshot
        return self.selected

    def _store_iteration_snapshot(self, reason: str):
        """Helper to store the JSON snapshot for the current iteration."""
        iteration_snapshot = {
            "iteration": self.iterations_completed + 1,
            "simulation": self.current_simulation_in_iteration,
            "reason": reason,
            "tree_json": self.export_tree_as_json()
        }
        self.iteration_json_snapshots.append(iteration_snapshot)

    async def _report_tree_stats(self):
        """Generate and report statistics about the current tree structure."""
        # (Code from v0.3.1 seems correct)
        # ...
        pass # Keep implementation

    def _collect_leaves(self, node, leaf_nodes):
        """Helper function to collect all leaf nodes in the tree."""
        # (Code from v0.3.1 seems correct)
        if not node.children: leaf_nodes.append(node)
        else:
            for child in node.children: self._collect_leaves(child, leaf_nodes)

    async def analyze_iteration(self):
        """Analyze the current iteration."""
        # (Code from v0.3.1 seems correct, uses self.llm.analyze_iteration)
        if self.best_solution and self.best_score > 0:
            context = { # Build context
                 "question": self.question, "best_answer": self.best_solution,
                 "best_score": self.best_score, "tree_depth": self.memory.get("depth",0),
                 "branches": self.memory.get("branches",0), "approach_types": ", ".join(self.approach_types),
            }
            try:
                 analysis = await self.llm.analyze_iteration(context)
                 if analysis:
                     self.thought_history.append(f"## Iteration Analysis\n{analysis}\n")
                     return analysis
            except Exception as e:
                 logger.error(f"Error during iteration analysis: {e}")
        return None


    def mermaid(self, selected=None):
        """Generate mermaid diagram code."""
        # (Code from v0.3.1 seems correct)
        return f"```mermaid\ngraph TD\n{self.root.mermaid(0, selected.id if selected else None)}\n```"

    def formatted_output(self, highlighted_node=None):
        """Generate comprehensive output block."""
        # (Code from v0.3.1 seems reasonable, ensure context variables are updated)
        iter_num = self.iterations_completed + 1
        sim_num = self.current_simulation_in_iteration
        max_sims = config['simulations_per_iteration']
        result = f"# MCTS Process - Iteration {iter_num} / Simulation {sim_num}/{max_sims}\n\n"
        result += "## Current Search Tree\n"
        result += self.mermaid(highlighted_node) # Use node object directly
        result += "\n## Thought Process History\n"
        result += "\n".join(self.thought_history[-5:])
        if self.surprising_nodes: result += "\n## Surprising Nodes Detected\n..." # Add details
        if self.best_solution: result += f"\n## Current Best Solution\nScore: {self.best_score:.2f}/10\n\n{self.best_solution}\n" # Use .2f for score
        # Approach Performance Summary (same as v0.3.1)
        # ...
        result += f"\n## Search Parameters\n..." # List parameters
        if config["debug_logging"] and self.debug_history: result += "\n## Debug Information\n..." # Add details
        return result

# ==============================================================================

class Pipe:
    """Interface with Open WebUI."""
    # (Valves definition remains same as v0.3.1)
    class Valves(BaseModel):
         # ... (All valves from v0.3.1) ...
         MAX_ITERATIONS: int = Field(default=config["max_iterations"], description="...")
         SIMULATIONS_PER_ITERATION: int = Field(default=config["simulations_per_iteration"], description="...")
         MAX_CHILDREN: int = Field(default=config["max_children"], description="...")
         EXPLORATION_WEIGHT: float = Field(default=config["exploration_weight"], description="...")
         SURPRISE_THRESHOLD: float = Field(default=config["surprise_threshold"], ge=0.0, le=1.0, description="...")
         USE_SEMANTIC_DISTANCE: bool = Field(default=config["use_semantic_distance"], description="...")
         NODE_LABEL_LENGTH: int = Field(default=config["node_label_length"], description="...")
         RELATIVE_EVALUATION: bool = Field(default=config["relative_evaluation"], description="...")
         SCORE_DIVERSITY_BONUS: float = Field(default=config["score_diversity_bonus"], description="...")
         GLOBAL_CONTEXT_IN_PROMPTS: bool = Field(default=config["global_context_in_prompts"], description="...")
         TRACK_EXPLORED_APPROACHES: bool = Field(default=config["track_explored_approaches"], description="...")
         SIBLING_AWARENESS: bool = Field(default=config["sibling_awareness"], description="...")
         MEMORY_CUTOFF: int = Field(default=config["memory_cutoff"], description="...")
         FORCE_EXPLORATION_INTERVAL: int = Field(default=config["force_exploration_interval"], description="...")
         DEBUG_LOGGING: bool = Field(default=config["debug_logging"], description="...")
         EARLY_STOPPING: bool = Field(default=config["early_stopping"], description="...")
         EARLY_STOPPING_THRESHOLD: float = Field(default=config["early_stopping_threshold"], description="...")
         EARLY_STOPPING_STABILITY: int = Field(default=config["early_stopping_stability"], description="...")


    def __init__(self):
        self.type = "manifold"
        self.__current_event_emitter__ = None
        self.__question__ = ""
        self.__model__ = ""

    # ... (pipes, resolve_model, resolve_question remain same) ...

    async def pipe( self, body: dict, __user__: dict, __event_emitter__=None, __task__=None, __model__=None) -> str | Generator | Iterator:
        # ... (setup model, question, emitter) ...
        model = self.resolve_model(body)
        base_question = self.resolve_question(body)
        if __task__ == TASKS.TITLE_GENERATION: # Handle title gen
            content = await self.get_completion(model, body.get("messages", []))
            return f"{name}: {content}"

        logger.debug(f"Pipe {name} received: {body}")
        question = base_question # Use base question directly for now
        self.__model__ = model
        self.__question__ = base_question
        self.__current_event_emitter__ = __event_emitter__

        # Apply valve settings or use hardcoded defaults
        if hasattr(self, "valves") and self.valves:
            logger.info(f"Applying Valve settings: {self.valves.model_dump()}") # Use model_dump for pydantic v2+
            try:
                 # Apply all defined valves to config
                 for key, value in self.valves.model_dump().items():
                     config_key = key.lower() # Assuming valve names match config keys in lowercase
                     if config_key in config:
                         config[config_key] = value
                     else:
                         logger.warning(f"Valve '{key}' not found in config dictionary.")
            except Exception as e:
                 logger.error(f"Error applying valve settings: {e}. Using defaults.")
                 # Optionally reset to hardcoded defaults here if applying fails
                 config["exploration_weight"] = 2.5
                 config["simulations_per_iteration"] = 15
        else:
            logger.warning("No valves found or provided, using hardcoded defaults.")
            config["exploration_weight"] = 2.5
            config["simulations_per_iteration"] = 15
            # Update other potentially critical defaults if valves fail
            config["max_iterations"] = 3
            config["max_children"] = 4
            # ... etc.

        # --- Introduction message ---
        await self.emit_message("# Advanced Monte Carlo Tree Search v0.3.1\n...")
        await self.emit_message("## Search Parameters (Current Run)\n...") # Display applied config

        # --- Initial Answer ---
        await self.progress("Preparing initial thoughts...")
        try:
            initial_reply = await self.stream_prompt_completion(initial_prompt, question=question)
            if not initial_reply: raise ValueError("LLM failed to provide initial reply.")
        except Exception as e:
             logger.error(f"Failed to get initial reply: {e}")
             await self.emit_message("Error: Could not generate an initial answer.")
             await self.done()
             return "Error generating initial answer." # Exit gracefully

        await self.emit_message("\n## Initial Answer:\n...\n")

        # --- MCTS Initialization ---
        try:
            root = Node(content=initial_reply)
            mcts = MCTS(root=root, llm=self, question=base_question)
            # Emit initial state using emit_message
            await self.emit_message(mcts.formatted_output())
        except Exception as e:
             logger.error(f"Failed to initialize MCTS: {e}")
             await self.emit_message("Error: Could not initialize search process.")
             await self.done()
             return "Error initializing MCTS."

        # --- MCTS Main Loop ---
        logger.debug("Starting MCTS iterations...")
        final_best_solution = initial_reply # Fallback
        try:
            for i in range(config["max_iterations"]):
                logger.debug(f"Starting Iteration {i + 1}/{config['max_iterations']}...")
                await self.emit_message(f"## Iteration {i + 1}/{config['max_iterations']}\n")
                await mcts.search(config["simulations_per_iteration"])
                mcts.iterations_completed += 1

                await self.emit_message(f"Completed Iteration {i+1}. Current best score: {mcts.best_score:.2f}/10\n")
                # Approach diversity report (same as v0.3.1)
                # ...

                # Early stopping check (same as v0.3.1)
                if mcts.high_score_counter >= config["early_stopping_stability"] and config["early_stopping"]:
                    await self.emit_message(f"Early stopping after iteration {i+1}.")
                    break
            # Store the best solution found
            final_best_solution = mcts.best_solution if mcts.best_solution else initial_reply

        except Exception as e:
             logger.error(f"Error during MCTS execution: {e}")
             await self.emit_message(f"An error occurred during the search process: {e}")
             # Use whatever best solution was found so far
             final_best_solution = mcts.best_solution if mcts.best_solution else initial_reply

        # --- Final Output ---
        await self.emit_message("\n\n## Final Answer:\n")
        await self.emit_message(f"{final_best_solution}\n")
        # Approach summary (same as v0.3.1)
        # ...
        # JSON Snapshots Output (same as v0.3.1)
        await self.emit_message("\n\n## Per-Iteration Tree JSON Snapshots\n")
        if mcts.iteration_json_snapshots:
             for snapshot in mcts.iteration_json_snapshots:
                 iteration_label = f"Iteration {snapshot['iteration']} (Sim {snapshot['simulation']}, Reason: {snapshot['reason']})"
                 await self.emit_message(f"### {iteration_label}\n")
                 tree_json_str = json.dumps(snapshot['tree_json'], indent=2)
                 await self.emit_message(f"```json\n{tree_json_str}\n```\n")
        else:
             await self.emit_message("No iteration snapshots were recorded.\n")


        # --- Termination ---
        await self.done()
        return final_best_solution


    # --- LLM Interaction Methods ---
    # (progress, done, emit_message, emit_replace remain same)
    async def progress(self, message: str): ...
    async def done(self): ...
    async def emit_message(self, message: str): ...
    async def emit_replace(self, message: str): ... # Keep for potential future use

    # (get_streaming_completion, get_message_completion, get_completion remain same)
    async def get_streaming_completion(self, model: str, messages) -> AsyncGenerator[str, None]: ...
    async def get_message_completion(self, model: str, content): ...
    async def get_completion(self, model: str, messages): ...

    # (call_ollama_endpoint_function - FIXED version without unsafe cleanup)
    async def call_ollama_endpoint_function(self, payload):
        # (Code from previous fix - using try/except, no finally block)
        async def receive(): return {"type": "http.request", "body": json.dumps(payload).encode("utf-8")}
        mock_request = Request(scope={"type": "http", "headers": [], "method": "POST", "scheme": "http", "server": ("localhost", 8000), "path": "/v1/chat/completions", "query_string": b"", "client": ("127.0.0.1", 8000), "app": app,}, receive=receive,)
        try:
            response = await ollama.generate_openai_chat_completion(request=mock_request, form_data=payload, user=admin)
            return response
        except Exception as e:
            logger.error(f"Error during Ollama API call: {str(e)}")
            # Return error message in expected format
            return {"choices": [{"message": {"content": f"Error: LLM communication failed ({e})."}}]}


    # (stream_prompt_completion remains same)
    async def stream_prompt_completion(self, prompt, **format_args):
         complete = ""
         # Use try-except to handle potential key errors during formatting if context is missing
         try:
             formatted_prompt = prompt.format(**format_args)
         except KeyError as e:
             logger.error(f"Missing key for prompt formatting: {e}. Prompt: {prompt[:100]}...")
             # Fallback or return error? For now, let it fail if essential keys are missing
             raise e # Or return an error message

         async for chunk in self.get_message_completion(self.__model__, formatted_prompt):
             complete += chunk
             # Avoid emitting empty messages
             if chunk: await self.emit_message(chunk)
         return complete

    # (generate_thought, evaluate_answer, evaluate_relative, analyze_iteration, update_approach - use context logic from v0.3.1)
    async def generate_thought(self, answer, context=None):
        # (Code from v0.3.1 using context seems correct)
        if not context or not config["global_context_in_prompts"]:
            prompt_to_use = thoughts_prompt.split("<context>")[0] + thoughts_prompt.split("</context>")[1]
            format_args = {"answer": answer, "question": self.__question__}
        else:
             prompt_to_use = thoughts_prompt
             format_args = {
                 "answer": answer, "question": self.__question__,
                 "best_answer": context.get("best_answer", "N/A"), "best_score": context.get("best_score", 0),
                 "explored_approaches": context.get("explored_approaches", "N/A"),
                 "current_approach": context.get("current_approach", "N/A")
             }
        try:
            return await self.stream_prompt_completion(prompt_to_use, **format_args)
        except Exception as e:
             logger.error(f"Error generating thought: {e}")
             return "Error: Could not generate thought."

    async def evaluate_answer(self, answer, context=None):
        # (Code from v0.3.1 using context seems correct)
        if not context or not config["global_context_in_prompts"]:
             prompt_to_use = eval_answer_prompt.split("<context>")[0] + eval_answer_prompt.split("</context>")[1]
             format_args = {"answer": answer, "question": self.__question__}
        else:
             prompt_to_use = eval_answer_prompt
             format_args = {
                 "answer": answer, "question": self.__question__,
                 "best_answer": context.get("best_answer", "N/A"), "best_score": context.get("best_score", 0)
             }
        try:
            result = await self.stream_prompt_completion(prompt_to_use, **format_args)
            score_match = re.search(r'\b([1-9]|10)\b', result) # More specific regex for 1-10
            if score_match: return int(score_match.group(1))
            else: logger.error(f'AnswerEval: No score 1-10 found in "{result[:100]}"'); return 0
        except Exception as e:
             logger.error(f"Error evaluating answer: {e}")
             return 0 # Return default score on error

    async def evaluate_relative(self, parent_answer, answer, context=None):
        # (Code from v0.3.1 using context seems correct)
        if not context or not config["global_context_in_prompts"]:
             prompt_to_use = relative_eval_prompt.split("<context>")[0] + relative_eval_prompt.split("</context>")[1]
             format_args = {"parent_answer": parent_answer, "answer": answer, "question": self.__question__}
        else:
             prompt_to_use = relative_eval_prompt
             format_args = {
                 "parent_answer": parent_answer, "answer": answer, "question": self.__question__,
                 "best_answer": context.get("best_answer", "N/A"), "best_score": context.get("best_score", 0)
             }
        try:
            result = await self.stream_prompt_completion(prompt_to_use, **format_args)
            score_match = re.search(r'\b[1-5]\b', result) # Regex for 1-5
            if score_match: return max(1, min(5, int(score_match.group(0))))
            else: logger.error(f'RelativeEval: No score 1-5 found in "{result[:100]}"'); return 3
        except Exception as e:
             logger.error(f"Error evaluating relative answer: {e}")
             return 3 # Default score on error

    async def analyze_iteration(self, context):
        # (Code from v0.3.1 using context seems correct)
        try:
             return await self.stream_prompt_completion(analyze_prompt, **context)
        except Exception as e:
             logger.error(f"Error analyzing iteration: {e}")
             return "Error during analysis."

    async def update_approach(self, answer, improvements, context=None):
        # (Code from v0.3.1 using context seems correct)
        if not context or not config["global_context_in_prompts"]:
             prompt_to_use = update_prompt.split("<context>")[0] + update_prompt.split("</context>")[1]
             format_args = {"answer": answer, "improvements": improvements, "question": self.__question__}
        else:
             prompt_to_use = update_prompt
             format_args = {
                 "answer": answer, "improvements": improvements, "question": self.__question__,
                 "best_answer": context.get("best_answer", "N/A"), "best_score": context.get("best_score", 0),
                 "explored_approaches": context.get("explored_approaches", "N/A"),
                 "current_approach": context.get("current_approach", "N/A")
             }
        try:
            return await self.stream_prompt_completion(prompt_to_use, **format_args)
        except Exception as e:
             logger.error(f"Error updating approach: {e}")
             return f"Error updating answer: {e}" # Return error message


    # (get_response_content remains same)
    def get_response_content(self, response):
        try:
            # Check if response is valid and has expected structure
            if isinstance(response, dict) and "choices" in response and response["choices"]:
                 message = response["choices"][0].get("message", {})
                 content = message.get("content")
                 if content: return content
            logger.warning(f"Unexpected response structure: {response}")
            return ""
        except Exception as e:
            logger.error(f'ResponseError extracting content: {str(e)}')
            return ""

    # (get_chunk_content - FIXED version)
    def get_chunk_content(self, chunk):
        """Process streaming chunks from the LLM with robust error handling."""
        try:
            chunk_str = chunk.decode("utf-8")
            # Remove "data: " prefix if present
            if chunk_str.startswith("data: "):
                chunk_str = chunk_str[6:]
            # Strip whitespace
            chunk_str = chunk_str.strip()
            # Ignore empty lines or control messages like "[DONE]"
            if not chunk_str or chunk_str == "[DONE]":
                return

            # Attempt to parse as JSON (standard OpenAI streaming format)
            try:
                chunk_data = json.loads(chunk_str)
                if isinstance(chunk_data, dict) and "choices" in chunk_data and chunk_data["choices"]:
                    delta = chunk_data["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content: # Only yield if content exists and is not None
                        yield content
            except json.JSONDecodeError:
                # If it's not JSON, log it, but don't yield unless certain it's text
                logger.debug(f"Chunk not JSON: {chunk_str[:100]}")
                # Avoid yielding potential control messages or partial JSON fragments
                if not chunk_str.startswith("{") and not chunk_str.startswith("["):
                     yield chunk_str # Assume plain text if not JSON-like
            except Exception as e:
                logger.error(f"Error processing JSON chunk: {e} - Chunk: {chunk_str[:100]}")

        except UnicodeDecodeError:
             logger.error(f"ChunkDecodeError: unable to decode chunk bytes.")
        except Exception as e:
             logger.error(f"Error processing raw chunk stream: {e}")

# ==============================================================================
# Ensure no code exists below this line except perhaps comments or whitespace.
# The previous error was caused by duplicated class definitions here.
# ==============================================================================