Project Blueprint: The Cognitively-Grounded Reasoning Agent (CGRA)

Introduction: Moving Beyond Mimicry to True Machine Reasoning

Modern large language models (LLMs) exhibit a perplexing form of cognitive dissonance: they solve highly complex problems yet fail on simpler variants. This dissociation is not a scientific curiosity but a critical architectural liability, revealing that their success stems from mechanisms fundamentally different from the robust cognitive structures of human reasoning. This reliance on brittle shortcuts and memorization creates a source of unpredictable, high-stakes failures that precludes their use in mission-critical systems. This blueprint presents a high-level architectural design for a new class of general reasoning agent—the Cognitively-Grounded Reasoning Agent (CGRA)—engineered to eliminate this liability. The design is directly informed by a novel, comprehensive taxonomy that synthesizes decades of cognitive science research into an actionable engineering framework.

This document proceeds as a logical engineering argument. It first diagnoses the core deficits in contemporary models, then establishes the theoretical requirements for a solution by introducing our four-dimensional cognitive framework. Subsequently, it proposes the CGRA's modular architecture, a concrete system designed to meet these requirements. Finally, it outlines a training and evaluation strategy to manufacture this system's capabilities. To build an agent that can truly reason, we must first understand the specific and systematic gaps that separate human cognition from machine performance.


--------------------------------------------------------------------------------


1. The Cognitive Chasm: Diagnosing the Limitations of Contemporary LLM Reasoners

To engineer more capable AI reasoners, we must move beyond simple outcome-based evaluation. The current paradigm, which rewards models solely for correct final answers, has created a "measurement crisis" by incentivizing memorization and spurious reasoning shortcuts over the cultivation of genuine cognitive processes. A deep analysis of how models behave during reasoning reveals a profound disconnect between their default strategies and the approaches that actually lead to success, accumulating a form of cognitive technical debt that makes them brittle and unreliable.

A meta-analysis of 1,598 LLM reasoning research papers quantifies a significant imbalance in the community's focus. Research has overwhelmingly concentrated on easily measured behaviors while neglecting the complex, meta-cognitive capabilities crucial for robust reasoning.

Dominant Research Focus (Top 5)	Neglected Capabilities (Bottom 5)
Context Awareness (70%)	Backward Chaining (8%)
Decomposition & Integration (60%)	Evaluation (8%)
Knowledge Structure Alignment (56%)	Spatial Organization (10%)
Sequential Organization (55%)	Self-Awareness (16%)
Pattern Recognition (51%)	Productivity (16%)

This research bias is mirrored in model behavior. Analysis of 170,000 reasoning traces confirms that models consistently deploy a narrow set of familiar behaviors even when the problem demands a more adaptive approach. The combined findings show a clear misalignment between the behaviors models most frequently exhibit and those most strongly correlated with successful outcomes. In essence, models deploy behaviors inversely to what success requires, defaulting to rigid strategies precisely when flexibility is most critical.

This analysis crystallizes into several key deficiencies:

* Operational Rigidity: Models overwhelmingly default to linear, step-by-step processing (Sequential Organization) and forward-chaining logic. This occurs even when problems demand alternative knowledge structures (e.g., hierarchical, spatial) and more sophisticated navigation strategies like working backward from a goal (Backward Chaining).
* Ineffective Execution of Core Behaviors: Foundational reasoning invariants like Logical Coherence are frequently present in model outputs. However, their weak correlation with success indicates a critical execution failure; models may attempt to maintain consistency but often fail to recognize or effectively resolve contradictions.
* Limited Meta-Cognitive Control: Models demonstrate a profound struggle with self-assessment (Evaluation), a deficit that becomes more pronounced on non-verifiable problems where ground truth is ambiguous. This lack of effective self-monitoring prevents them from identifying errors and adapting their strategies.

These failures are not isolated bugs but symptoms of a flawed paradigm. Overcoming them requires an architecture built from first principles, grounded in a comprehensive model of cognition.


--------------------------------------------------------------------------------


2. The Cognitive Blueprint: A Four-Dimensional Framework for General Reasoning

To build this agent, we first need a blueprint. Our architecture is grounded in a unified taxonomy of 28 cognitive elements derived from foundational theories of problem-solving, mental representation, and meta-cognition. Organized by Marr's levels of analysis, this framework provides the "analytical vocabulary" needed to design, build, and evaluate more robust reasoners. Effective reasoning, much like building a complex LEGO model, emerges from the dynamic interplay of four distinct dimensions: the fundamental rules of valid construction (Invariants), the executive plan guiding the build (Controls), the mental scaffolds used to organize sections (Representations), and the specific actions used to connect pieces (Operations).

Dimension 1: Reasoning Invariants (The Computational Constraints) These are the "always-true" constraints that define valid reasoning at a computational level, ensuring that the process remains consistent, coherent, and generative.

* Logical Coherence: Maintain consistency across reasoning steps and contexts.
* Compositionality: Build complex ideas from simpler components.
* Productivity: Formulate an indefinite number of thoughts or solutions using a finite set of elements.
* Conceptual Processing: Operate over abstract representations before linguistic expression.

Dimension 2: Meta-Cognitive Controls (The Executive Regulator) These are the higher-order executive functions that select, monitor, and adapt reasoning strategies in response to the task and environment.

* Self-awareness: Assess own knowledge state, capabilities, and task solvability.
* Context awareness: Perceive, understand, and navigate one’s circumstances.
* Strategy selection: Choose and explore reasoning approaches suited to task and domain demands.
* Goal management: Establish, maintain, and adjust goals throughout the reasoning process.
* Evaluation: Assess and adapt to the quality, efficiency, and progress of one’s reasoning.

Dimension 3: Reasoning Representations (The Knowledge Scaffolds) These are the organizational structures and formats used to encode knowledge and guide the reasoning process, providing the mental workspace for thought.

* Structural Organization: Sequential, Hierarchical, Network, Ordinal.
* Conceptual Organization: Causal, Temporal, Spatial.

Dimension 4: Reasoning Operations (The Transformation Toolkit) These are the goal-directed procedures that construct, evaluate, modify, and navigate the knowledge representations.

* Representation Selection: Context alignment, Knowledge alignment.
* Representation Evaluation: Verification.
* Representation Modification: Selective attention, Adaptive detail management, Decomposition and integration, Representational restructuring, Pattern recognition, Abstraction.
* Representation Navigation: Forward chaining, Backward chaining, Backtracking.

This comprehensive cognitive blueprint moves beyond a narrow focus on isolated behaviors and serves as the direct architectural specification for the proposed reasoning agent.


--------------------------------------------------------------------------------


3. Architectural Design: The Cognitively-Grounded Reasoning Agent (CGRA)

The Cognitively-Grounded Reasoning Agent (CGRA) is envisioned not as a monolithic model but as a modular, integrated system. Each component is explicitly designed to instantiate one of the four dimensions of the cognitive framework, promoting high cohesion and low coupling between functions. This architecture directly addresses the identified failures of current models, particularly their over-reliance on shallow, sequential forward chaining, by building in the capacity for meta-cognitive oversight and representational flexibility.

The CGRA is composed of four primary, interacting modules:

1. The Constraint Governor (Invariants Module): The Constraint Governor enforces computational invariants across the entire reasoning stack. It is a continuous validation layer responsible for maintaining Logical Coherence and Compositionality, ensuring that all intermediate and final outputs remain consistent and generative.
2. The Executive Controller (Meta-Cognition Module): This is the central orchestrator of the CGRA, instantiating the meta-cognitive controls. Its core functions include Strategy Selection, Goal Management, and continuous Evaluation. This module directly addresses the critical lack of meta-cognitive control identified in our analysis, where models failed to perform effective Evaluation on non-verifiable problems.
3. The Dynamic Workspace (Representations Module): This module provides a flexible representational system capable of constructing, maintaining, and switching between diverse knowledge structures. By enabling the construction of Hierarchical and Network structures on demand, this module overcomes the Operational Rigidity that locks current models into ineffective sequential processing.
4. The Procedural Toolkit (Operations Module): This module provides a suite of callable reasoning operators that manipulate the representations in the Dynamic Workspace. It includes a rich set of operations such as Forward Chaining, Backward Chaining, Backtracking, and Representational Restructuring, which are selected and deployed by the Executive Controller.

This modular design enables a dynamic and adaptive reasoning flow. For an ill-structured Dilemma problem, the Executive Controller would initiate a Self-Awareness check, select a Hierarchical Organization strategy in the Dynamic Workspace, and deploy the Decomposition operation from the Procedural Toolkit to break the problem down, all while the Constraint Governor ensures logical consistency. The effectiveness of this architectural vision is supported by empirical evidence showing that eliciting such structured cognitive patterns can dramatically improve performance.


--------------------------------------------------------------------------------


4. From Theory to Performance: Elicitation and Training Strategy for the CGRA

Our training philosophy is a direct response to the community's research imbalance. Where current methods have neglected capabilities like Evaluation and Backward Chaining (both studied in only 8% of research), our protocol will explicitly target, incentivize, and cultivate them. This strategy moves beyond simple outcome-based rewards to a process-oriented approach, transforming latent capabilities into spontaneously deployed skills.

This approach is grounded in powerful empirical evidence. Experiments with "test-time reasoning guidance" demonstrate that providing existing models with explicit cognitive scaffolding improves performance by up to 66.7% on complex, ill-structured problems. This serves as a proof-of-concept that capable models possess latent reasoning structures that can be activated with targeted intervention. Our training protocol is designed to internalize this scaffolding, making robust cognitive processes an intrinsic part of the agent's behavior.

The training protocol will proceed through a multi-stage process:

1. Foundation via Cognitive Scaffolding: The initial training phase will use process supervision, where the agent is rewarded for generating reasoning traces that follow optimal cognitive structures derived from the "consensus subgraphs" identified as most correlated with success.
2. Reward Shaping for Meta-Cognition: We will move to a reinforcement learning stage where the reward model is explicitly augmented to value meta-cognitive behaviors like effective Evaluation (error detection) and adaptive Strategy Selection, independently of final answer correctness.
3. Curriculum Design for Representational Diversity: To develop representational flexibility, the training curriculum will force the agent to generalize underlying principles by training on sets of problems that are structurally similar but have diverse surface features, promoting schema abstraction.
4. Environment Design for Robustness: The agent will be trained in adversarial environments specifically designed to penalize brittleness and "spurious reasoning shortcuts," forcing the development of more generalized and adaptable problem-solving capabilities.

This integrated approach, combining a cognitively-grounded architecture with a behavior-focused training strategy, is designed to produce a new generation of AI reasoners capable of robust and flexible thought.


--------------------------------------------------------------------------------


5. Conclusion: A New Paradigm for Artificial General Reasoning

Achieving robust, human-like reasoning requires a paradigm shift away from simply scaling current architectures toward a new model of development grounded in cognitive science. The Cognitively-Grounded Reasoning Agent (CGRA) embodies this new paradigm. Its modular design—featuring a Constraint Governor, Executive Controller, Dynamic Workspace, and Procedural Toolkit—is a direct implementation of a comprehensive cognitive framework, built to overcome the brittleness and operational rigidity that characterize today's models.

The vision for this work extends beyond a single agent; we are proposing a reference architecture for a new class of reasoning systems. The goal is to develop AI that exhibits flexible, adaptive, and truly general reasoning—a stark contrast to the brittle, in-distribution performance common today. This framework provides the shared vocabulary and measurement infrastructure needed to transform AI development, enabling a paradigm shift from post-hoc observation to theory-driven engineering in our pursuit of robust, generalizable intelligence.

Taxonomy of cognitive reasoning behaviors, organized along two main dimensions: Cognitive
Properties (stable internal constraints) and Cognitive Capabilities (deployable context-sensitive abilities).
A. Reasoning Invariants: "Always-true" constraints or quality criteria the system maintains across reasoning steps.
Logical coherence Maintain consistency across reasoning steps and contexts (Fodor & Pylyshyn, 1988).
Compositionality Build complex ideas from simpler components (Fodor, 1975).
Productivity Formulate an indefinite number of thoughts or solutions using a finite set of elements (Halford, 1989).
Conceptual processing Operating over abstract representations before linguistic expression (Halford, 1989).
B. Meta-Cognitive Controls: Higher-order abilities that select, monitor, and adapt processes.
Self-awareness Assess own knowledge state, capabilities, and task solvability (Wicklund, 1979).
Context awareness Perceive, understand, and navigate one’s circumstances (including other agents) (Frith & Frith, 2007).
Strategy selection Choose & explore reasoning approaches suited to task and domain demands (Lieder & Griffiths, 2017).
Goal management Establish, maintain, and adjust goals throughout the reasoning process (Griffiths et al., 2019).
Evaluation Assess & adapt to the quality, efficiency, and progress of one’s reasoning (Fleming & Daw, 2017).
C. Reasoning Representations: The formats and organizational patterns used to encode and relate knowledge and steps.
Structural
Sequential organization Order steps where sequence matters (Skinner, 1953).
Organization
Hierarchical organization Nest concepts in parent–child relationships (Galanter et al., 1960).
Network organization Link concepts through multiple relationship types (Quillan, 1966).
Ordinal organization Arrange elements by relative order or rank (Stevens, 1946).
Conceptual Causal organization Connect elements through cause–effect relations (Heider, 1958).
Organization Temporal organization Order elements by before–after relations (Ebbinghaus, 1885).
Spatial organization Structure elements by spatial relationships (Tolman, 1948).
D. Reasoning Operations: Goal-directed actions that construct, evaluate, modify, and navigate reasoning representations.
Representation Context alignment Align to task and situational demands (Gick & Holyoak, 1980).
Selection Knowledge alignment Align to domain-specific structures & relations (Chi et al., 1981).
Representation
Verification Check reasoning steps against pre-determined criteria (Flavell, 1979).
Evaluation
Representation
Selective attention Focus on relevant details and filtering noise (Broadbent, 1958).
Modification
Adaptive detail mgmt. Adjust granularity based on task demands (Rosch, 1978).
Decomposition and integration Divide problems and synthesizing subsolutions (Newell et al., 1959).
Representational restructuring Reformulate problems for new insights (Wertheimer, 1945).
Pattern recognition Detect recurring structures across contexts (Selfridge, 1959).
Abstraction Generalize from specific cases (Hull, 1920).
Representation
Forward chaining Reason from known facts toward goals (Huys et al., 2012).
Navigation
Backward chaining Work backward from goals to prerequisites (Park et al., 2017).
Backtracking Revisit and correcting prior reasoning paths (Nilsson, 1971).

Concept Source: https://arxiv.org/pdf/2511.16660v1

Cognitive Foundations for Reasoning and Their Manifestation in LLMs

Priyanka Kargupta1♡, Shuyue Stella Li2♡, Haocheng Wang3
Jinu Lee1
Shan Chen4
Orevaoghene Ahia2
Dean Light2
Thomas L. Griffiths3
Max Kleiman-Weiner2
Jiawei Han1
Asli Celikyilmaz2
Yulia Tsvetkov2