# COMPASS Framework

**Cognitive Orchestration & Metacognitive Planning for Adaptive Semantic Systems**

A sophisticated AI reasoning framework that integrates six powerful cognitive systems into a unified architecture for advanced decision-making, planning, and problem-solving.

## 🌟 Overview

COMPASS synthesizes multiple cognitive frameworks to create an intelligent system that can:
- 🧠 **Think metacognitively** about resource allocation and confidence
- 📊 **Plan strategically** with SMART objectives
- 🔄 **Learn continuously** through self-reflection
- 🎯 **Reason systematically** through semantic logic progression
process user inputs intelligently
- 🤖 **Decide optimally** using multi-modal intelligence

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│              (SHAPE + Semantic Processing)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 METACOGNITIVE CONTROL LAYER                  │
│        (oMCD Resource Allocation + Self-Discover)            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Resource   │  │ Confidence   │  │  Self-Reflection │   │
│  │ Optimizer   │  │  Evaluator   │  │     Engine       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 REASONING & PLANNING LAYER                   │
│          (SLAP Logical Flow + SMART Goals)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   SLAP      │  │    SMART     │  │  Reasoning       │   │
│  │  Pipeline   │  │   Planner    │  │   Modules        │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              INTELLIGENCE EXECUTION LAYER                    │
│             (Integrated Intelligence Core)                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────────┐  │
│  │Learning │ │Reasoning│ │  NLU    │ │  Uncertainty     │  │
│  └─────────┘ └─────────┘ └─────────┘ └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Integrated Frameworks

| Framework | Purpose | Key Features |
|-----------|---------|--------------|
| **SLAP** | Semantic Logic Auto Progression | 8-stage pipeline: C→R→F→S→D→RB→M→SF |
| **SHAPE** | Shorthand Assisted Prompt Engineering | Input processing, expansion, semantic mapping |
| **SMART** | Strategic Management & Resource Tracking | Goal-oriented planning with measurable objectives |
| **oMCD** | Online Metacognitive Control of Decisions | Optimal resource allocation via confidence-cost tradeoff |
| **Self-Discover** | Reinforcement via Self-Reflection | Actor-evaluator-reflection loop with 39 reasoning modules |
| **Integrated Intelligence** | Multi-Modal Intelligence | Universal intelligence combining 6 modalities |

## 🚀 Quick Start

### Installation

```bash
cd /home/ty/Repositories/ai_workspace/agent_projects/unified_cognitive_system
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install numpy
```

### Basic Usage

```python
from compass_framework import quick_solve

# Solve a task with one line
result = quick_solve("Optimize the algorithm for better performance")

print(f"Success: {result['success']}")
print(f"Solution: {result['solution']}")
print(f"Score: {result['score']:.3f}")
```

### Advanced Usage

```python
from compass_framework import create_compass
from config import create_custom_config

# Create custom configuration
config = create_custom_config(
    omcd={'alpha': 0.2, 'R': 15.0},
    slap={'alpha': 0.3, 'beta': 0.7},
    self_discover={'max_trials': 15}
)

# Initialize COMPASS
compass = create_compass(config)

# Process a complex task
result = compass.process_task(
    task_description="Create an efficient sorting algorithm",
    context={'domain': 'algorithms', 'constraints': ['time complexity']},
    max_iterations=10
)

# View results
print(f"Iterations: {result['iterations']}")
print(f"Reflections: {len(result['reflections'])}")
print(f"Resources used: {result['resources_used']:.2f}")

# Check status
status = compass.get_status()
print(f"Average score: {status['average_score']:.3f}")
```

## 📚 Components

### SHAPE Processor
Processes user input with shorthand expansion and semantic enrichment.

```python
from shape_processor import SHAPEProcessor

processor = SHAPEProcessor(config.shape)
processed = processor.process_user_input("opt the algo for perf")
expanded = processor.expand_shorthand(processed)
semantic = processor.map_semantics(expanded)
```

### oMCD Controller
Optimizes cognitive resource allocation based on confidence-cost tradeoffs.

```python
from omcd_controller import oMCDController

controller = oMCDController(config.omcd)
allocation = controller.determine_resource_allocation(
    current_state={'value_difference': 0.5, 'variance': 0.3},
    importance=10.0,
    available_resources=100.0
)
```

### Self-Discover Engine
Manages continuous improvement through self-reflection.

```python
from self_discover_engine import SelfDiscoverEngine

engine = SelfDiscoverEngine(config.self_discover)
modules = engine.select_reasoning_modules(task, reflections)
reflection = engine.generate_reflection(trajectory, score, objectives)
```

### SLAP Pipeline
Processes information through 8-stage semantic logic pipeline.

```python
from slap_pipeline import SLAPPipeline

pipeline = SLAPPipeline(config.slap)
plan = pipeline.create_reasoning_plan(task, objectives)
advancement = plan['advancement']  # Truth + α·Scrutiny + β·Improvement
```

### SMART Planner
Creates and manages SMART objectives.

```python
from smart_planner import SMARTPlanner

planner = SMARTPlanner(config.smart)
objectives = planner.create_objectives_from_task(task, context)
progress = planner.monitor_progress(objectives)
```

### Integrated Intelligence
Synthesizes decisions using multi-modal intelligence.

```python
from integrated_intelligence import IntegratedIntelligence

intelligence = IntegratedIntelligence(config.intelligence)
decision = intelligence.make_decision(task, plan, modules, resources, context)
```

## ⚙️ Configuration

All framework parameters are configured through dataclasses:

```python
from config import COMPASSConfig, oMCDConfig, SLAPConfig

# Modify specific components
config = COMPASSConfig()
config.omcd.alpha = 0.15  # Unitary effort cost
config.omcd.R = 12.0      # Decision importance
config.slap.alpha = 0.5   # Scrutiny weight
config.slap.beta = 0.5    # Improvement weight
```

## 📊 Examples

See the `examples/` directory for comprehensive examples:

- **`example_basic_task.py`** - Simple task execution
- **`example_complex_reasoning.py`** - Multi-stage reasoning
- **`example_adaptive_learning.py`** - Learning from feedback
- **`example_resource_optimization.py`** - Resource-constrained scenarios

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 📖 Documentation

### Key Formulas

**oMCD Resource Allocation:**
- Benefit: `B(z) = R × P_c(z)`
- Cost: `C(z) = α × z^ν`
- Net Benefit: `max(B(z) - C(z))`

**SLAP Advancement:**
- `Advancement = Truth + (α × Scrutiny) + (β × Improvement)`
- Constraint: `α + β = 1.0`

**Universal Intelligence:**
- `U(x) = Σ(ωᵢ × Fᵢ(x)) + Σ(ωⱼₖ × Fⱼ(x) × Fₖ(x))`

### Reasoning Modules

Self-Discover includes 39 reasoning modules covering:
- Critical thinking and creativity
- Systems thinking and risk analysis
- Problem decomposition
- Constraint identification
- Metrics and evaluation
- Step-by-step planning

## 🔬 Research Background

This framework integrates concepts from:
- **Self-Discover:** Reinforcement learning via self-reflection
- **oMCD Model:** Metacognitive control of decisions (computational neuroscience)
- **Semantic Logic:** Formal reasoning systems
- **SMART Goals:** Project management and planning
- **Universal Intelligence:** Multi-modal AI reasoning

## 📝 License

MIT License - feel free to use and modify for your projects!

## 🤝 Contributing

This is a research framework developed for exploring integrated cognitive architectures. Contributions, ideas, and feedback are welcome!

## 🎯 Roadmap

- [ ] Add neural-symbolic hybrid reasoning
- [ ] Implement advanced MCTS for entity identification
- [ ] ML-based shorthand discovery in SHAPE
- [ ] Performance benchmarking suite
- [ ] Interactive visualization dashboard
- [ ] Integration with external knowledge bases

## 📧 Author

Built with 🧠 by Ty

---

**COMPASS** - Where metacognition meets semantic reasoning for intelligent decision-making.
