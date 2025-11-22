# COMPASS Architecture Visualization

This document provides visual representations of the COMPASS framework architecture using Mermaid diagrams.

## System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[User Input] --> SHAPE[SHAPE Processor]
        SHAPE --> |Shorthand Expansion| SE[Semantic Engine]
        SE --> |Enriched Prompt| META
    end

    subgraph "Metacognitive Layer"
        META[Metacognitive Control]
        META --> OMCD[oMCD Controller]
        META --> SD[Self-Discover Engine]
        OMCD --> |Resource Allocation| EXEC
        SD --> |Module Selection| EXEC
        SD --> |Self-Reflection| MEM[(Memory)]
    end

    subgraph "Reasoning Layer"
        EXEC[Reasoning Executor]
        EXEC --> SLAP[SLAP Pipeline]
        EXEC --> SMART[SMART Planner]
        SLAP --> |Advancement Score| DEC
        SMART --> |Objectives| DEC
    end

    subgraph "Intelligence Layer"
        DEC[Decision Synthesizer]
        DEC --> II[Integrated Intelligence]
        II --> LEARN[Learning]
        II --> REASON[Reasoning]
        II --> NLU[NLU]
        II --> UNC[Uncertainty]
        II --> EVOL[Evolution]
        II --> NEURAL[Neural]
    end

    DEC --> |Solution| OUTPUT[Output]
    OUTPUT --> |Feedback| SHAPE

    style UI fill:#e1f5ff
    style OUTPUT fill:#e1f5ff
    style META fill:#fff4e1
    style EXEC fill:#ffe1f5
    style DEC fill:#e1ffe1
```

## SLAP Pipeline Flow

```mermaid
flowchart LR
    C[Conceptualization] --> R[Representation]
    R --> F[Facts]
    F --> S[Scrutiny]
    S --> D[Derivation]
    D --> RB[Rule-Based]
    RB --> M[Model]
    M --> SF[Semantic Formalization]

    S -.-> ADV[Advancement Score]
    M -.-> ADV
    ADV -.-> |"Truth + α·Scrutiny + β·Improvement"| EVAL[Evaluation]

    style C fill:#ffcccc
    style R fill:#ffddcc
    style F fill:#ffeecc
    style S fill:#ffffcc
    style D fill:#eeffcc
    style RB fill:#ddffcc
    style M fill:#ccffcc
    style SF fill:#ccffdd
    style ADV fill:#cce5ff
```

## Self-Discover Loop

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> SelectModules: Start Task
    SelectModules --> AllocateResources: Modules Selected
    AllocateResources --> Execute: Resources Allocated
    Execute --> Evaluate: Action Complete
    Evaluate --> Reflect: Score Generated
    Reflect --> CheckThreshold: Reflection Created

    CheckThreshold --> [*]: Pass ✓
    CheckThreshold --> SelectModules: Continue

    Reflect --> Memory: Store
    Memory --> SelectModules: Inform Next Iteration

    note right of SelectModules
        Adaptive: Uses past
        reflections & task type
    end note

    note right of AllocateResources
        oMCD: Optimizes
        confidence-cost tradeoff
    end note
```

## Component Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant COMPASS
    participant SHAPE
    participant SMART
    participant SLAP
    participant oMCD
    participant SelfDiscover
    participant Intelligence

    User->>COMPASS: process_task(description)
    COMPASS->>SHAPE: process_user_input()
    SHAPE->>SHAPE: expand_shorthand()
    SHAPE->>SHAPE: map_semantics()
    SHAPE-->>COMPASS: enriched_prompt

    COMPASS->>SMART: create_objectives()
    SMART-->>COMPASS: objectives[]

    COMPASS->>SLAP: create_reasoning_plan()
    SLAP-->>COMPASS: reasoning_plan

    loop Until Pass or Max Iterations
        COMPASS->>SelfDiscover: select_modules()
        SelfDiscover-->>COMPASS: modules[]

        COMPASS->>oMCD: determine_allocation()
        oMCD-->>COMPASS: resources

        COMPASS->>Intelligence: make_decision()
        Intelligence-->>COMPASS: decision

        COMPASS->>SelfDiscover: evaluate_trajectory()
        SelfDiscover-->>COMPASS: score

        COMPASS->>SelfDiscover: generate_reflection()
        SelfDiscover-->>COMPASS: reflection

        COMPASS->>oMCD: should_stop()
        oMCD-->>COMPASS: continue/stop
    end

    COMPASS-->>User: result
```

## oMCD Resource Allocation

```mermaid
graph LR
    subgraph "Input"
        STATE[Current State] --> CALC
        IMP[Importance R] --> CALC
        AVAIL[Available Resources] --> CALC
    end

    subgraph "Calculation"
        CALC[Calculate Optimal z]
        CALC --> PREC[Update Precision]
        PREC --> CONF["Confidence P_c(z)"]
        CONF --> BEN["Benefit: R × P_c"]
        CALC --> COST["Cost: α × z^ν"]
    end

    subgraph "Optimization"
        BEN --> NET[Net Benefit]
        COST --> NET
        NET --> OPT[Maximize]
        OPT --> |"ẑ = argmax(B - C)"| ALLOC
    end

    subgraph "Output"
        ALLOC[Allocation Decision]
        ALLOC --> AMT[Amount]
        ALLOC --> CONFIDENCE[Confidence]
        ALLOC --> NETBEN[Net Benefit]
    end

    style STATE fill:#ffe6e6
    style IMP fill:#ffe6e6
    style AVAIL fill:#ffe6e6
    style ALLOC fill:#e6ffe6
    style AMT fill:#e6ffe6
    style CONFIDENCE fill:#e6ffe6
    style NETBEN fill:#e6ffe6
```

## Integrated Intelligence Components

```mermaid
mindmap
    root((Universal Intelligence))
        Learning
            Q-Learning
            Transfer Learning
            Adaptability
        Reasoning
            Bayesian
            SLAP Advancement
            Logical Inference
        NLU
            Semantic Analysis
            Intent Extraction
            Context Understanding
        Uncertainty
            Entropy Calculation
            Confidence Estimation
            Risk Quantification
        Evolution
            Fitness Evaluation
            Diversity Metrics
            Selection
        Neural
            Activation Functions
            Feature Weighting
            Non-linear Mapping
```

## Data Flow

```mermaid
flowchart TD
    START([User Task]) --> INPUT[Raw Input]

    INPUT --> SHAPE_PROC[SHAPE Processing]
    SHAPE_PROC --> EXPAND[Expansion]
    EXPAND --> SEMANTIC[Semantic Mapping]

    SEMANTIC --> OBJ_GEN[Objective Generation]
    OBJ_GEN --> REASON_PLAN[Reasoning Plan]

    REASON_PLAN --> ITER_START{Start Iteration}

    ITER_START --> MOD_SEL[Module Selection]
    MOD_SEL --> RES_ALLOC[Resource Allocation]
    RES_ALLOC --> EXEC[Execute Reasoning]

    EXEC --> EVAL[Evaluate]
    EVAL --> REFLECT[Generate Reflection]

    REFLECT --> STOP_CHECK{Stop?}
    STOP_CHECK -->|No| ITER_START
    STOP_CHECK -->|Yes| RESULTS

    RESULTS[Compile Results] --> OUTPUT([Output with Score, Reflections, Trajectory])

    REFLECT -.-> MEM[(Memory Store)]
    MEM -.-> MOD_SEL

    style START fill:#e1f5ff
    style OUTPUT fill:#e1ffe1
    style ITER_START fill:#fff4e1
    style STOP_CHECK fill:#ffe1e1
```

## SMART Objective Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Define: Task Input
    Define --> Assess: Objectives Created
    Assess --> Feasible: Feasibility Check
    Feasible --> Relevant: Relevance Check
    Relevant --> Active: Validated ✓

    Active --> Monitor: Progress Tracking
    Monitor --> OnTrack: Status Check
    Monitor --> OffTrack: Status Check

    OnTrack --> Completed: Target Reached
    OnTrack --> Monitor: Continue

    OffTrack --> Adjust: Refinement Needed
    Adjust --> Active: Updated

    Completed --> [*]

    note right of Assess
        Specific, Measurable,
        Achievable, Relevant,
        Time-bound
    end note
```

## Configuration Hierarchy

```mermaid
classDiagram
    class COMPASSConfig {
        +oMCDConfig omcd
        +SLAPConfig slap
        +SMARTConfig smart
        +SelfDiscoverConfig self_discover
        +SHAPEConfig shape
        +IntegratedIntelligenceConfig intelligence
        +bool enable_logging
        +str log_level
        +int max_workers
    }

    class oMCDConfig {
        +float alpha
        +float nu
        +float beta
        +float gamma
        +float lambda_conf
        +float R
        +float max_resources
    }

    class SLAPConfig {
        +float alpha
        +float beta
        +List stages
        +int mcts_iterations
    }

    class SMARTConfig {
        +List objective_categories
        +Dict metrics
        +int default_timeline_days
    }

    class SelfDiscoverConfig {
        +int max_trials
        +List enabled_reasoning_modules
        +str module_selection_strategy
        +float pass_threshold
    }

    class SHAPEConfig {
        +Dict shorthand_dict
        +int context_window_size
        +bool enable_ml_expansion
    }

    class IntegratedIntelligenceConfig {
        +Dict linear_weights
        +float interaction_weight
        +float gamma_discounting
    }

    COMPASSConfig --> oMCDConfig
    COMPASSConfig --> SLAPConfig
    COMPASSConfig --> SMARTConfig
    COMPASSConfig --> SelfDiscoverConfig
    COMPASSConfig --> SHAPEConfig
    COMPASSConfig --> IntegratedIntelligenceConfig
```

## Execution Timeline

```mermaid
gantt
    title COMPASS Task Execution Timeline
    dateFormat X
    axisFormat %L

    section SHAPE
    Input Processing       :a1, 0, 10
    Shorthand Expansion    :a2, 10, 20
    Semantic Mapping       :a3, 20, 30

    section Planning
    Objective Creation     :b1, 30, 40
    Reasoning Plan (SLAP)  :b2, 40, 60

    section Iteration 1
    Module Selection       :c1, 60, 65
    Resource Allocation    :c2, 65, 70
    Decision Synthesis     :c3, 70, 85
    Evaluation            :c4, 85, 90
    Reflection            :c5, 90, 100

    section Iteration 2
    Module Selection       :d1, 100, 105
    Resource Allocation    :d2, 105, 110
    Decision Synthesis     :d3, 110, 120
    Evaluation            :d4, 120, 125
    Reflection            :d5, 125, 135

    section Finalization
    Result Compilation     :e1, 135, 145
    Feedback Collection    :e2, 145, 150
```

---

## Legend

- **Solid Lines**: Direct data/control flow
- **Dashed Lines**: Feedback or memory access
- **Colors**:
  - Blue: Input/Output
  - Yellow: Metacognitive layer
  - Pink: Reasoning layer
  - Green: Intelligence layer
