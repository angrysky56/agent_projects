"""
title: advanced_mcts
author: angrysky56
author_url: https://github.com/angrysky56
description: Advanced Monte Carlo Tree Search with persistent visualization
version: 0.1.1
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
)
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
import open_webui.routers.ollama as ollama
from open_webui.main import app

# ==============================================================================

name = "advanced_mcts"

# Configurable parameters with default values
config = {
    "max_children": 3,  # Maximum number of children per node
    "exploration_weight": 1.414,  # UCB exploration parameter (√2)
    "max_iterations": 3,  # Number of MCTS iterations to run
    "simulations_per_iteration": 3,  # Number of simulations per iteration (renamed for clarity)
    "thoughts_per_node": 2,  # Number of thoughts to generate per node (legacy, used differently now)
    "surprise_threshold": 0.7,  # Threshold for detecting surprising nodes
    "use_semantic_distance": True,  # Whether to use semantic distance in node selection
    "use_llm_embedding_for_distance": False,  # Whether to use LLM embeddings for semantic distance
}

# ==============================================================================

# Prompt templates
thoughts_prompt = """
<instruction>
Give a suggestion on how this answer can be improved.
WRITE ONLY AN IMPROVEMENT SUGGESTION AND NOTHING ELSE.
YOUR REPLY SHOULD BE A SINGLE SENTENCE.
</instruction>

<question>
{question}
</question>

<draft>
{answer}
</draft>
""".strip()

eval_answer_prompt = """
Given the following text:
"{answer}"

How well does it answers this question:
"{question}"

Rate the answer from 1 to 10, where 1 is completely wrong or irrelevant and 10 is a perfect answer.
Reply with a single number between 1 and 10 only. Do not write anything else, it will be discarded.
THINK CAREFULLY AND USE BEST PRACTICES.
""".strip()

analyze_prompt = """
Iteration Analysis:

Original question: {question}
Best answer found: {best_answer}
Best score achieved: {best_score}

Analyze this iteration of the thought process. Consider the following:
1. What aspects of the best answer made it successful?
2. What patterns or approaches led to higher-scoring thoughts?
3. Were there any common pitfalls or irrelevant tangents in lower-scoring thoughts?
4. How can the thought generation process be improved for the next iteration?

Provide a concise analysis and suggest one specific improvement strategy for the next iteration.
""".strip()

update_prompt = """
<instruction>
Your task is to read the question and the answer below, then analyse the given critique.
When you are done - think about how the answer can be improved based on the critique.
WRITE A REVISED ANSWER THAT ADDRESSES THE CRITIQUE. DO NOT WRITE ANYTHING ELSE.
</instruction>
<question>
{question}
</question>
<draft>
{answer}
</draft>
<critique>
{improvements}
</critique>
""".strip()

initial_prompt = """
<instruction>
Answer the question below. Do not pay attention to, unexpected casing, punctuation or accent marks.
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

# ==============================================================================


def setup_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class AdminUserMock:
    def __init__(self):
        self.role = "admin"


admin = AdminUserMock()
logger = setup_logger()

# ==============================================================================

mods = [
    "capitalize",
    "diacritic",
    "leetspeak",
    "remove_vowel",
]


def modify_text(text, percentage):
    if not text:
        return "", {}  # Return empty string and empty mapping if input is empty

    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")

    words = text.split()
    chars = list(text)
    num_chars_to_modify = max(1, int(len(chars) * (percentage / 100)))
    indices_to_modify = random.sample(range(len(chars)), num_chars_to_modify)
    word_mapping = {}

    for idx in indices_to_modify:
        modification = random.choice(mods)

        # Find the word that contains the current character
        current_length = 0
        for word_idx, word in enumerate(words):
            if current_length <= idx < current_length + len(word):
                original_word = word
                word_start_idx = current_length
                break
            current_length += len(word) + 1  # +1 for the space
        else:
            # If we're here, we're likely dealing with a space or the last character
            continue

        if modification == "capitalize":
            chars[idx] = chars[idx].swapcase()
        elif modification == "diacritic":
            if chars[idx].isalpha():
                diacritics = ["̀", "́", "̂", "̃", "̈", "̄", "̆", "̇", "̊", "̋"]
                chars[idx] = chars[idx] + random.choice(diacritics)
        elif modification == "leetspeak":
            leetspeak_map = {
                "a": "4",
                "e": "3",
                "i": "1",
                "o": "0",
                "s": "5",
                "t": "7",
                "b": "8",
                "g": "9",
                "l": "1",
            }
            chars[idx] = leetspeak_map.get(chars[idx].lower(), chars[idx])
        elif modification == "remove_vowel":
            if chars[idx].lower() in "aeiou":
                chars[idx] = ""

        modified_word = "".join(
            chars[word_start_idx : word_start_idx + len(original_word)]
        )

        if modified_word != original_word:
            # Clean up both the modified word and the original word
            cleaned_modified_word = modified_word.rstrip(".,!?")
            cleaned_original_word = original_word.rstrip(".,!?")
            word_mapping[cleaned_modified_word] = cleaned_original_word

    modified_text = "".join(chars)
    return modified_text, word_mapping


def replace_with_mapping(text, mapping):
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text


# ==============================================================================


def truncate_text(text, max_length=60):
    """Truncate text and add ellipsis if needed."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def escape_mermaid(text):
    """Escape special characters for Mermaid diagrams."""
    if not text:
        return ""
    # Escape quotes, parentheses, and other special characters
    escaped = html.escape(text)
    escaped = escaped.replace("(", "&#40;").replace(")", "&#41;")
    return escaped


def calculate_semantic_distance(text1, text2, llm=None):
    """Calculate semantic distance between two text blocks."""
    # Use LLM embeddings if specified and available
    if config["use_llm_embedding_for_distance"] and llm:
        # This would use the embedding_prompt to generate semantic representations
        # and then calculate distance. For now, fall back to TF-IDF.
        pass
    
    # Otherwise use TF-IDF vectorization
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return 1.0 - similarity  # Convert similarity to distance
    except:
        # Fallback to simple word overlap if TF-IDF fails
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 1.0
        overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
        return 1.0 - overlap  # Convert similarity to distance


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
        self.value = 0
        self.sequence = kwargs.get("sequence", 0)
        self.embedding = kwargs.get("embedding", "")
        self.is_surprising = False
        self.surprise_explanation = ""
        # Give surprising nodes an optimistic initial value bias
        if kwargs.get("is_surprising", False):
            self.is_surprising = True
            self.value = 2.0  # Small positive bias

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

        # Exploitation component (average value)
        exploitation = self.value / (self.visits + epsilon)

        # Exploration component (UCB1 formula)
        exploration = self.exploration_weight * math.sqrt(
            math.log(self.parent.visits) / (self.visits + epsilon)
        )

        # Add bonus for surprising nodes
        surprise_bonus = 0.2 if self.is_surprising else 0

        return exploitation + exploration + surprise_bonus

    def mermaid(self, offset=0, selected=None):
        """Generate mermaid diagram code for this node and its children."""
        padding = " " * offset
        # Use a more descriptive node label with score
        score = self.value / (self.visits or 1)  # Avoid division by zero
        score_text = f"{score:.2f}" if self.visits > 0 else "N/A"

        # Create a meaningful display with node content preview
        display_text = truncate_text(self.content, 40)

        # Add special indicators for surprising nodes
        surprise_indicator = "⭐ " if self.is_surprising else ""

        node_label = f"{self.id}[\"{surprise_indicator}Node {self.sequence}: Score={score_text}<br/>'{escape_mermaid(display_text)}'\"]"

        msg = f"{padding}{node_label}\n"

        # Highlight the selected node
        if selected == self.id:
            msg += f"{padding}style {self.id} fill:#d4f0fd,stroke:#0099ff,stroke-width:2px\n"

        # Add different styles based on scores and properties
        elif self.visits > 0:
            if self.is_surprising:
                # Surprising nodes get a special highlight
                msg += f"{padding}style {self.id} fill:#fcf8d3,stroke:#f1c40f,stroke-width:2px\n"
            elif score > 7:
                msg += f"{padding}style {self.id} fill:#d4ffd4,stroke:#00cc00\n"  # Good nodes (green)
            elif score < 4:
                msg += f"{padding}style {self.id} fill:#ffd4d4,stroke:#cc0000\n"  # Poor nodes (red)

        # Generate diagram code for children
        for child in self.children:
            msg += child.mermaid(offset + 4, selected)
            # Add edge with visit count
            msg += f"{padding}{self.id} -->|{child.visits}| {child.id}\n"

        return msg

    def best_child(self):
        """Return the best child node based on visit count."""
        if not self.children:
            return self

        return max(self.children, key=lambda child: child.visits).best_child()


class MCTS:
    """Monte Carlo Tree Search implementation."""

    def __init__(self, **kwargs):
        self.question = kwargs.get("question")
        self.root = kwargs.get("root")
        # Initialize node_sequence BEFORE using it
        self.node_sequence = 0
        # Now we can safely give the root node a sequence number
        self.root.sequence = self.get_next_sequence()
        self.llm = kwargs.get("llm")
        self.selected = None
        self.exploration_weight = config["exploration_weight"]
        self.thought_history = []  # Track thought process history
        self.surprising_nodes = []
        self.best_solution = None
        self.best_score = 0

        # Initialize with a welcome message to preserve in history
        self.thought_history.append(
            f"# MCTS Analysis for Question: {self.question}\n\nStarting exploration with initial answer...\n"
        )

    def get_next_sequence(self):
        """Get the next sequence number for nodes."""
        # Add safety check to prevent attribute errors
        if not hasattr(self, 'node_sequence'):
            self.node_sequence = 0
        self.node_sequence += 1
        return self.node_sequence

    async def select(self):
        """Select a node to expand using UCT."""
        logger.debug("Selecting node...")
        node = self.root
        selection_path = [node]

        while node.children:
            node = self.uct_select(node)
            selection_path.append(node)

        # Add selection path to history
        path_str = " → ".join([f"Node {n.sequence}" for n in selection_path])
        self.thought_history.append(f"### Selection Path\n{path_str}\n")

        return node

    async def expand(self, node) -> Tuple[Node, bool]:
        """
        Expand a node by generating ONE thought and creating ONE child node.
        
        Returns:
            Tuple[Node, bool]: The newly created child node and a flag indicating
                              if the node is surprising.
        """
        logger.debug(f"Expanding node {node.id}...")
        await self.llm.progress(f"Exploring a new thought path from node {node.sequence}...")

        # Generate a thought history entry
        thought_entry = f"### Expanding Node {node.sequence}\nCurrent answer: {truncate_text(node.content, 100)}\n\n"

        # Display the current tree with the selected node highlighted
        await self.llm.emit_replace(self.formatted_output(node))

        # Generate a single thought (instead of multiple)
        await self.llm.emit_message(f"Generating thought for Node {node.sequence}: ")
        thought = await self.llm.generate_thought(node.content)
        thought_entry += f"**Thought**: {thought}\n\n"

        # Update approach based on thought
        await self.llm.emit_message(f"\n\n---\n\nRefining solution based on thought:\n")
        new_content = await self.llm.update_approach(node.content, thought)

        # Check if this node is surprising based on semantic distance
        is_surprising = False
        if config["use_semantic_distance"]:
            semantic_distance = calculate_semantic_distance(
                node.content, new_content, self.llm
            )

            if semantic_distance > config["surprise_threshold"]:
                is_surprising = True
                surprise_explanation = f"This solution takes a significantly different approach from its parent (semantic distance: {semantic_distance:.2f})"
                thought_entry += f"**SURPRISE DETECTED!** Semantic distance: {semantic_distance:.2f}\n"
                thought_entry += f"**Explanation**: {surprise_explanation}\n\n"

        # Add the new node with a sequence number
        child = Node(
            content=new_content,
            parent=node,
            sequence=self.get_next_sequence(),
            is_surprising=is_surprising,  # Pass the surprising flag directly
        )

        if is_surprising:
            child.surprise_explanation = surprise_explanation
            self.surprising_nodes.append(child)

        node.add_child(child)
        thought_entry += f"**New solution {child.sequence}**: {truncate_text(new_content, 100)}\n\n"

        # Add thought history
        self.thought_history.append(thought_entry)

        # Show updated tree
        await self.llm.emit_replace(self.formatted_output(child))

        return child, is_surprising

    async def simulate(self, node):
        """Simulate and evaluate a node."""
        logger.debug(f"Simulating node {node.id}...")
        await self.llm.progress(
            f"Evaluating solution quality for node {node.sequence}..."
        )
        await self.llm.emit_replace(self.formatted_output(node))

        score = await self.llm.evaluate_answer(node.content)

        # Log evaluation in thought history
        eval_entry = f"### Evaluating Node {node.sequence}\nScore: {score}/10\nSolution: {truncate_text(node.content, 100)}\n\n"
        self.thought_history.append(eval_entry)

        return score

    def backpropagate(self, node, score):
        """Backpropagate the result up the tree."""
        logger.debug(f"Backpropagating from {node.id}...")
        # Create a backpropagation log
        backprop_path = []
        temp_node = node
        while temp_node:
            backprop_path.append(f"Node {temp_node.sequence}")
            temp_node = temp_node.parent

        backprop_entry = f"### Backpropagating Score {score}\nPath: {' → '.join(reversed(backprop_path))}\n\n"
        self.thought_history.append(backprop_entry)

        # Actual backpropagation
        while node:
            node.visits += 1
            node.value += score
            node = node.parent

    def uct_select(self, node):
        """Select a child node using UCT with Bayesian and semantic components."""
        logger.debug(f"Selecting uct {node.id}...")
        return max(node.children, key=lambda child: child.uct_value())

    def best_child(self):
        """Return the best child based on visit count."""
        return self.root.best_child()

    async def search(self, simulations_per_iteration):
        """
        Perform Monte Carlo Tree Search for a specified number of simulations.
        
        Now follows standard MCTS pattern of:
        1. Select a promising leaf node
        2. Expand the leaf node by creating ONE child
        3. Simulate the new child
        4. Backpropagate the results
        """
        logger.debug(f"Starting search for {simulations_per_iteration} simulations...")

        for i in range(simulations_per_iteration):
            iteration_entry = f"## Simulation {i+1}/{simulations_per_iteration}\n\n"
            self.thought_history.append(iteration_entry)

            # --- Selection Phase ---
            leaf = await self.select()
            self.selected = leaf

            node_to_simulate = leaf  # Default to simulating the leaf

            # --- Expansion Phase (if possible) ---
            if not leaf.fully_expanded():
                # Expand with one child instead of multiple
                new_child, is_surprising = await self.expand(leaf)
                if new_child:  # Check if expansion was successful
                    node_to_simulate = new_child  # Simulate the newly created child
                    self.selected = new_child  # Update selected node for visualization

            # --- Simulation Phase ---
            score = await self.simulate(node_to_simulate)

            # --- Backpropagation Phase ---
            self.backpropagate(node_to_simulate, score)

            # --- Update Best Solution Found So Far ---
            # Track best solution inside MCTS instead of in the Pipe class
            if score > self.best_score:
                self.best_score = score
                self.best_solution = node_to_simulate.content
                
                # Add to history
                self.thought_history.append(
                    f"### New Best Solution Found!\nScore: {score}/10\nNode: {node_to_simulate.sequence}\n"
                )
                
                # Update visualization with the new best solution
                await self.llm.emit_replace(self.formatted_output(self.selected))

            # Between iteration analysis (using the previously unused analyze_prompt)
            if i > 0 and i % 2 == 0:  # Add periodic analysis (every other iteration)
                await self.analyze_iteration()

        return self.selected

    async def analyze_iteration(self):
        """Analyze the current iteration and suggest improvements."""
        if self.best_solution and self.best_score > 0:
            analysis = await self.llm.analyze_iteration(self.best_solution, self.best_score)
            analysis_entry = f"## Iteration Analysis\n{analysis}\n\n"
            self.thought_history.append(analysis_entry)
            return analysis
        return None

    def mermaid(self, selected=None):
        """Generate mermaid diagram with improved styling."""
        return f"""
```mermaid
graph TD
{self.root.mermaid(0, selected.id if selected else (self.selected.id if self.selected else None))}
```
"""

    def formatted_output(self, highlighted_node=None):
        """Generate a comprehensive output including mermaid diagram and thought history."""
        result = "# Advanced Monte Carlo Tree Search Process\n\n"

        # Add the mermaid diagram
        result += "## Current Search Tree\n"
        result += self.mermaid(highlighted_node)

        # Add the thought history
        result += "\n## Thought Process History\n"
        result += "\n".join(
            self.thought_history[-5:]
        )  # Show last 5 entries to keep it manageable

        # Add summary of surprising nodes if any
        if self.surprising_nodes:
            result += "\n## Surprising Nodes Detected\n"
            for i, node in enumerate(
                self.surprising_nodes[-3:]
            ):  # Show last 3 surprising nodes
                result += f"### Surprising Node {node.sequence}\n"
                result += f"Explanation: {node.surprise_explanation}\n\n"
                result += f"Content: {truncate_text(node.content, 150)}\n\n"

        # Add summary if available
        if self.best_solution:
            result += f"\n## Current Best Solution\nScore: {self.best_score}/10\n\n{self.best_solution}\n"

        # Add current parameters
        result += f"\n## Search Parameters\n"
        result += f"- Exploration weight: {config['exploration_weight']:.2f}\n"
        result += f"- Max children per node: {config['max_children']}\n"
        result += f"- Surprise threshold: {config['surprise_threshold']:.2f}\n"
        result += f"- Simulations per iteration: {config['simulations_per_iteration']}\n"

        return result


class Pipe:
    """Interface with Open WebUI."""

    # Define configurable parameters (valves)
    class Valves(BaseModel):
        MAX_ITERATIONS: int = Field(
            default=3, description="Number of MCTS iterations to run"
        )
        SIMULATIONS_PER_ITERATION: int = Field(
            default=3, description="Number of simulations per iteration"
        )
        MAX_CHILDREN: int = Field(
            default=3, description="Maximum number of children per node"
        )
        EXPLORATION_WEIGHT: float = Field(
            default=1.414, description="UCB exploration parameter (sqrt(2))"
        )

    def __init__(self):
        self.type = "manifold"
        self.__current_event_emitter__ = None
        self.__question__ = ""
        self.__model__ = ""

    def pipes(self) -> list[dict[str, str]]:
        """List available models."""
        ollama.get_all_models()
        models = app.state.OLLAMA_MODELS

        out = [
            {"id": f"{name}-{key}", "name": f"{name} {models[key]['name']}"}
            for key in models
        ]
        logger.debug(f"Available models: {out}")

        return out

    def resolve_model(self, body: dict) -> str:
        """Extract model name from request body."""
        model_id = body.get("model")
        without_pipe = ".".join(model_id.split(".")[1:])
        return without_pipe.replace(f"{name}-", "")

    def resolve_question(self, body: dict) -> str:
        """Extract question from request body."""
        return body.get("messages")[-1].get("content").strip()

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
    ) -> str | Generator | Iterator:
        """Main entry point for processing requests."""
        model = self.resolve_model(body)
        base_question = self.resolve_question(body)

        if __task__ == TASKS.TITLE_GENERATION:
            content = await self.get_completion(model, body.get("messages"))
            return f"{name}: {content}"

        logger.debug(f"Pipe {name} received: {body}")
        question, mapping = modify_text(base_question, 0)
        logger.debug(f"Question: {question}")

        # Setup
        self.__model__ = model
        self.__question__ = base_question
        self.__current_event_emitter__ = __event_emitter__

        # Apply valve settings
        if hasattr(self, "valves"):
            config["max_iterations"] = self.valves.MAX_ITERATIONS
            config["simulations_per_iteration"] = self.valves.SIMULATIONS_PER_ITERATION
            config["max_children"] = self.valves.MAX_CHILDREN
            config["exploration_weight"] = self.valves.EXPLORATION_WEIGHT

        # Introduction message
        await self.emit_message(
            "# Advanced Monte Carlo Tree Search for Question Answering\n\n"
        )
        await self.emit_message(f"**Question:** {base_question}\n\n")
        await self.emit_message(
            "*Starting the intelligent exploration process with surprise detection...*\n\n"
        )

        # Initial config display
        await self.emit_message("## Search Parameters\n")
        await self.emit_message(f"- Max iterations: {config['max_iterations']}\n")
        await self.emit_message(
            f"- Simulations per iteration: {config['simulations_per_iteration']}\n"
        )
        await self.emit_message(
            f"- Exploration weight: {config['exploration_weight']}\n\n"
        )

        await self.progress("Preparing initial thoughts...")
        initial_reply = await self.stream_prompt_completion(
            initial_prompt, question=question
        )

        # Display initial answer
        await self.emit_message("\n## Initial Answer:\n\n")
        await self.emit_message(f"{initial_reply}\n\n---\n\n")
        await self.emit_message("Now refining through Monte Carlo Tree Search...\n\n")

        # Create root node
        root = Node(content=initial_reply)

        # Initialize MCTS with root node
        mcts = MCTS(root=root, llm=self, question=base_question)

        logger.debug("Starting MCTS...")
        for i in range(config["max_iterations"]):
            logger.debug(f"Iteration {i + 1}/{config['max_iterations']}...")

            await self.emit_message(
                f"## Iteration {i + 1}/{config['max_iterations']}\n\n"
            )
            await mcts.search(config["simulations_per_iteration"])

            # No need to evaluate best_child here - MCTS already tracks best solution
            await self.emit_message(
                f"Completed iteration {i+1}. Current best score: {mcts.best_score}/10\n\n"
            )

            # Show current state with full history
            await self.emit_replace(mcts.formatted_output())

        # Special section for surprising nodes if any were found
        if mcts.surprising_nodes:
            await self.emit_message("\n## Surprising Insights Discovered\n\n")
            for node in mcts.surprising_nodes:
                await self.emit_message(f"### Surprising Node {node.sequence}\n")
                await self.emit_message(f"{node.surprise_explanation}\n\n")
                await self.emit_message(f"**Content:** {node.content[:200]}...\n\n")

        # Final output using best solution tracked by MCTS
        await self.emit_message("\n\n## Final Answer:\n\n")
        await self.emit_message(f"{mcts.best_solution}\n\n")

        # Final tree visualization
        await self.emit_replace(mcts.formatted_output())

        await asyncio.sleep(0.2)
        await self.done()

        return mcts.best_solution

    async def progress(self, message: str):
        """Send a progress status message."""
        logger.debug(f"Progress: {message}")
        await self.__current_event_emitter__(
            {
                "type": "status",
                "data": {
                    "level": "info",
                    "description": message,
                    "done": False,
                },
            }
        )

    async def done(self):
        """Send a completed status message."""
        await self.__current_event_emitter__(
            {
                "type": "status",
                "data": {
                    "level": "info",
                    "description": "Fin.",
                    "done": True,
                },
            }
        )

    async def emit_message(self, message: str):
        """Send a message to be appended to the current output."""
        await self.__current_event_emitter__(
            {
                "type": "message",
                "data": {"content": message},
            }
        )

    async def emit_replace(self, message: str):
        """Replace the visualization with a new version."""
        await self.__current_event_emitter__(
            {
                "type": "replace",
                "data": {"content": message},
            }
        )

    async def get_streaming_completion(
        self,
        model: str,
        messages,
    ) -> AsyncGenerator[str, None]:
        """Stream completions from the model."""
        response = await self.call_ollama_endpoint_function(
            {"model": model, "messages": messages, "stream": True}
        )

        async for chunk in response.body_iterator:
            for part in self.get_chunk_content(chunk):
                yield part

    async def get_message_completion(self, model: str, content):
        """Get a streaming completion for a single message."""
        async for chunk in self.get_streaming_completion(
            model, [{"role": "user", "content": content}]
        ):
            yield chunk

    async def get_completion(self, model: str, messages):
        """Get a completion from the model."""
        response = await self.call_ollama_endpoint_function(
            {"model": model, "messages": messages, "stream": False}
        )

        return self.get_response_content(response)

    async def call_ollama_endpoint_function(self, payload):
        """Call the Ollama endpoint."""

        async def receive():
            return {"type": "http.request", "body": json.dumps(payload).encode("utf-8")}

        mock_request = Request(
            scope={
                "type": "http",
                "headers": [],
                "method": "POST",
                "scheme": "http",
                "server": ("localhost", 8000),
                "path": "/v1/chat/completions",
                "query_string": b"",
                "client": ("127.0.0.1", 8000),
                "app": app,
            },
            receive=receive,
        )

        return await ollama.generate_openai_chat_completion(
            request=mock_request, form_data=payload, user=admin
        )

    async def stream_prompt_completion(self, prompt, **format_args):
        """Stream a completion for a prompt with formatting arguments."""
        complete = ""
        async for chunk in self.get_message_completion(
            self.__model__,
            prompt.format(**format_args),
        ):
            complete += chunk
            await self.emit_message(chunk)
        return complete

    async def generate_thought(self, answer):
        """Generate a thought about how to improve the answer."""
        return await self.stream_prompt_completion(
            thoughts_prompt, answer=answer, question=self.__question__
        )

    async def evaluate_answer(self, answer):
        """Evaluate an answer on a scale of 1-10."""
        result = await self.stream_prompt_completion(
            eval_answer_prompt,
            answer=answer,
            question=self.__question__,
        )
        try:
            score = re.search(r"\d+", result).group()
            return int(score)
        except AttributeError:
            logger.error(f'AnswerEval: unable to parse "{result[:100]}"')
            return 0

    async def analyze_iteration(self, best_answer, best_score):
        """Analyze the current iteration and suggest improvements."""
        return await self.stream_prompt_completion(
            analyze_prompt,
            question=self.__question__,
            best_answer=best_answer,
            best_score=best_score,
        )

    async def update_approach(self, answer, improvements):
        """Update an answer based on suggested improvements."""
        return await self.stream_prompt_completion(
            update_prompt,
            question=self.__question__,
            answer=answer,
            improvements=improvements,
        )

    def get_response_content(self, response):
        """Extract content from the LLM response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.error(
                f'ResponseError: unable to extract content from "{response[:100]}"'
            )
            return ""

    def get_chunk_content(self, chunk):
        """Process streaming chunks from the LLM."""
        chunk_str = chunk.decode("utf-8")
        if chunk_str.startswith("data: "):
            chunk_str = chunk_str[6:]

        chunk_str = chunk_str.strip()

        if chunk_str == "[DONE]" or not chunk_str:
            return

        try:
            chunk_data = json.loads(chunk_str)
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
        except json.JSONDecodeError:
            logger.error(f'ChunkDecodeError: unable to parse "{chunk_str[:100]}"')