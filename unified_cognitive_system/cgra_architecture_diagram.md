# CGRA-Enhanced COMPASS Architecture

Visual representation of how CGRA integrates into the COMPASS framework.

## Unified System Architecture

```mermaid
graph TB
    subgraph "USER INTERFACE LAYER"
        UI[User Input] --> SHAPE[SHAPE Processor]
        SHAPE --> |Enriched Prompt| META
    end

    subgraph "CGRA CONSTRAINT LAYER - NEW"
        CG[Constraint Governor]
        CG --> |Validates| INV[Invariants]
        INV --> LC[Logical Coherence]
        INV --> COMP[Compositionality]
        INV --> PROD[Productivity]
        INV --> CP[Conceptual Processing]
    end

    subgraph "META-COGNITIVE CONTROL LAYER - ENHANCED"
        META[Executive Controller - NEW]
        META --> OMCD[oMCD Resource Allocation]
        META --> SD[Self-Discover Reflection]
        META --> GM[Goal Management - NEW]
        META --> EVAL[Continuous Evaluation - NEW]
        META --> SA[Self-Awareness - NEW]
        META --> CA[Context Awareness - NEW]
    end

    subgraph "DYNAMIC WORKSPACE LAYER - ENHANCED"
        DW[Dynamic Workspace - NEW]
        DW --> RS[Representation Selector - NEW]
        RS --> |Selects Type| REP[Representation Type]
        REP --> SEQ[Sequential - Existing]
        REP --> HIER[Hierarchical - NEW]
        REP --> NET[Network - NEW]
        REP --> SPATIAL[Spatial - NEW]
        REP --> CAUSAL[Causal - NEW]

        DW --> SLAP[SLAP Pipeline - Enhanced]
        DW --> SMART[SMART Planner]
    end

    subgraph "PROCEDURAL TOOLKIT LAYER - NEW"
        PT[Procedural Toolkit - NEW]
        PT --> BC[Backward Chaining - NEW]
        PT --> BT[Backtracking - NEW]
        PT --> RS2[Representational Restructuring - NEW]
        PT --> PR[Pattern Recognition - NEW]
        PT --> ABS[Abstraction - NEW]
        PT --> FC[Forward Chaining - Existing]
    end

    subgraph "EXECUTION LAYER"
        II[Integrated Intelligence]
        II --> LEARN[Learning]
        II --> REASON[Reasoning]
        II --> NLU[NLU]
        II --> UNC[Uncertainty]
    end

    SHAPE --> META
    META --> |Coordinates| DW
    DW --> |Uses| PT
    PT --> |Executes| II
    II --> |Result| OUTPUT[Output]

    CG -.-> |Validates All Steps| META
    CG -.-> |Validates| DW
    CG -.-> |Validates| PT
    CG -.-> |Validates| II

    OUTPUT --> |Feedback| SHAPE

    style UI fill:#e1f5ff
    style OUTPUT fill:#e1f5ff
    style CG fill:#ffe6e6
    style META fill:#fff4e1
    style DW fill:#ffe1f5
    style PT fill:#e6f5ff
    style II fill:#e1ffe1
```

## CGRA Module Details

### Constraint Governor (Reasoning Invariants)

```mermaid
flowchart LR
    INPUT[Reasoning Step] --> CG[Constraint Governor]
    CG --> LC{Logical Coherence?}
    CG --> COMP{Compositionality?}
    CG --> PROD{Productivity?}
    CG --> CP{Conceptual Processing?}

    LC -->|Pass| NEXT
    LC -->|Fail| VIOL[Violation Report]
    COMP -->|Pass| NEXT
    COMP -->|Fail| VIOL
    PROD -->|Pass| NEXT
    PROD -->|Fail| VIOL
    CP -->|Pass| NEXT[Continue]
    CP -->|Fail| VIOL

    VIOL --> LOG[Log & Alert]

    style LC fill:#ffcccc
    style COMP fill:#ffcccc
    style PROD fill:#ffcccc
    style CP fill:#ffcccc
    style NEXT fill:#ccffcc
    style VIOL fill:#ff9999
```

### Executive Controller (Meta-Cognitive Control)

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> EstablishGoals: Task Received
    EstablishGoals --> SelectStrategy: Goals Set
    SelectStrategy --> AllocateResources: Strategy Chosen

    AllocateResources --> ExecuteStep: Resources Allocated
    ExecuteStep --> ContinuousEval: Step Complete

    ContinuousEval --> AssessSelfAwareness: Evaluate Quality
    AssessSelfAwareness --> CheckGoals: Know Capabilities

    CheckGoals --> AdjustGoals: Goals Not Met
    CheckGoals --> SelectStrategy: Goals Adjusted
    CheckGoals --> [*]: Goals Achieved

    note right of EstablishGoals
        Goal Management (NEW)
        Creates goal stack
    end note

    note right of ContinuousEval
        Continuous Evaluation (NEW)
        Beyond simple scoring
    end note

    note right of AssessSelfAwareness
        Self-Awareness (NEW)
        Track knowledge state
    end note
```

### Dynamic Workspace (Representation Flexibility)

```mermaid
graph TD
    TASK[Task Input] --> ANALYZE[Analyze Task Structure]

    ANALYZE --> DECOMP{Requires Decomposition?}
    ANALYZE --> GOAL{Goal-Directed?}
    ANALYZE --> CAUSE{Causal Reasoning?}
    ANALYZE --> SPACE{Spatial Relations?}

    DECOMP -->|Yes| HIER[Hierarchical Representation]
    GOAL -->|Yes| NET[Network Representation]
    CAUSE -->|Yes| CAUSAL[Causal Representation]
    SPACE -->|Yes| SPATIAL[Spatial Representation]
    DECOMP -->|No| SEQ[Sequential Representation]

    HIER --> BUILD[Build Representation]
    NET --> BUILD
    CAUSAL --> BUILD
    SPATIAL --> BUILD
    SEQ --> BUILD

    BUILD --> PLAN[Reasoning Plan]

    style HIER fill:#cce5ff
    style NET fill:#cce5ff
    style CAUSAL fill:#cce5ff
    style SPATIAL fill:#cce5ff
    style SEQ fill:#e6e6e6
```

### Procedural Toolkit Operations

```mermaid
mindmap
    root((Procedural Toolkit))
        Representation Navigation
            Forward Chaining
                Existing SLAP
            Backward Chaining
                Goal to Prerequisites
                NEW
            Backtracking
                Correct Errors
                NEW
        Representation Modification
            Restructuring
                Reformulate Problem
                NEW
            Pattern Recognition
                Detect Recurring Structures
                NEW
            Abstraction
                Generalize from Cases
                NEW
            Selective Attention
                Filter Relevant Details
            Adaptive Detail Management
                Adjust Granularity
        Representation Evaluation
            Verification
                Check Against Criteria
        Representation Selection
            Context Alignment
                Task Demands
            Knowledge Alignment
                Domain Structures
```

## Integration Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant SHAPE
    participant ExecCtrl as Executive Controller (NEW)
    participant RepSel as Representation Selector (NEW)
    participant SLAP
    participant ProcTool as Procedural Toolkit (NEW)
    participant ConstGov as Constraint Governor (NEW)
    participant oMCD
    participant SelfDiscover
    participant Intelligence

    User->>SHAPE: Task Description
    SHAPE->>SHAPE: Expand Shorthand
    SHAPE->>ExecCtrl: Enriched Prompt

    ExecCtrl->>ExecCtrl: Establish Goals (NEW)
    ExecCtrl->>RepSel: Analyze Task Structure (NEW)
    RepSel-->>ExecCtrl: Representation Type (NEW)

    ExecCtrl->>SLAP: Create Plan (Type-Aware)
    SLAP-->>ExecCtrl: Reasoning Plan

    loop Reasoning Iterations
        ExecCtrl->>SelfDiscover: Select Modules
        SelfDiscover-->>ExecCtrl: Modules

        ExecCtrl->>oMCD: Allocate Resources
        oMCD-->>ExecCtrl: Resources

        ExecCtrl->>ProcTool: Apply Operation (NEW)
        alt Backward Chaining Needed
            ProcTool->>ProcTool: Backward Chain (NEW)
        else Forward Chaining
            ProcTool->>SLAP: Forward Chain
        end
        ProcTool-->>ExecCtrl: Step Result

        ExecCtrl->>ConstGov: Validate Step (NEW)
        ConstGov-->>ExecCtrl: Validation Result (NEW)

        alt Constraint Violated
            ConstGov->>ProcTool: Backtrack (NEW)
        end

        ExecCtrl->>ExecCtrl: Continuous Evaluation (NEW)
        ExecCtrl->>ExecCtrl: Check Goals (NEW)

        alt Goals Met
            ExecCtrl->>User: Solution
        end
    end
```

## CGRA Cognitive Elements Coverage

### Mapped to COMPASS Components

| CGRA Element | Implementation | COMPASS Component |
|--------------|----------------|-------------------|
| **Reasoning Invariants** | | |
| Logical Coherence | Constraint Governor | `constraint_governor.py` |
| Compositionality | Constraint Governor | `constraint_governor.py` |
| Productivity | Constraint Governor | `constraint_governor.py` |
| Conceptual Processing | Constraint Governor | `constraint_governor.py` |
| **Meta-Cognitive Controls** | | |
| Self-Awareness | Executive Controller | `executive_controller.py` |
| Context Awareness | Executive Controller | `executive_controller.py` |
| Strategy Selection | Executive Controller + Self-Discover | Enhanced module selection |
| Goal Management | Executive Controller + oMCD | Enhanced `omcd_controller.py` |
| Evaluation | Executive Controller | `executive_controller.py` |
| **Reasoning Representations** | | |
| Sequential Organization | SLAP Pipeline (existing) | `slap_pipeline.py` |
| Hierarchical Organization | Dynamic Workspace | Enhanced `slap_pipeline.py` |
| Network Organization | Dynamic Workspace | Enhanced `slap_pipeline.py` |
| Ordinal Organization | Dynamic Workspace | Enhanced `slap_pipeline.py` |
| Causal Organization | Dynamic Workspace | Enhanced `slap_pipeline.py` |
| Temporal Organization | Dynamic Workspace | Enhanced `slap_pipeline.py` |
| Spatial Organization | Dynamic Workspace | Enhanced `slap_pipeline.py` |
| **Reasoning Operations** | | |
| Context Alignment | Representation Selector | `representation_selector.py` |
| Knowledge Alignment | Representation Selector | `representation_selector.py` |
| Verification | Procedural Toolkit | `procedural_toolkit.py` |
| Selective Attention | Procedural Toolkit + SLAP | Enhanced operations |
| Adaptive Detail Mgmt | Procedural Toolkit | `procedural_toolkit.py` |
| Decomposition/Integration | Self-Discover (existing) | Enhanced `self_discover_engine.py` |
| Representational Restructuring | Procedural Toolkit | `procedural_toolkit.py` |
| Pattern Recognition | Procedural Toolkit | `procedural_toolkit.py` |
| Abstraction | Procedural Toolkit | `procedural_toolkit.py` |
| Forward Chaining | SLAP Pipeline (existing) | `slap_pipeline.py` |
| Backward Chaining | Procedural Toolkit | `procedural_toolkit.py` |
| Backtracking | Procedural Toolkit | `procedural_toolkit.py` |

**Total: 28/28 cognitive elements mapped ✓**

## Key Enhancements Summary

### 🔒 Constraint Enforcement
- **Before**: No explicit validation of reasoning invariants
- **After**: Continuous validation via Constraint Governor catching logical inconsistencies, compositional errors, and conceptual processing issues

### 🧠 Meta-Cognitive Control
- **Before**: oMCD handles resources, Self-Discover handles reflection
- **After**: Executive Controller unifies both + adds goal management, continuous evaluation, self-awareness, and context awareness

### 📊 Representation Flexibility
- **Before**: SLAP only sequential C→R→F→S→D→RB→M→SF
- **After**: Dynamic Workspace supports hierarchical, network, spatial, causal, and temporal representations

### 🔧 Operational Richness
- **Before**: Only forward chaining and basic decomposition
- **After**: Procedural Toolkit adds backward chaining, backtracking, restructuring, pattern recognition, abstraction, and more

---

## File Structure After Integration

```
unified_cognitive_system/
├── config.py (MODIFIED - adds CGRA configs)
├── compass_framework.py (MODIFIED - integrates CGRA)
├── omcd_controller.py (MODIFIED - adds goal management)
├── self_discover_engine.py (MODIFIED - enhanced context awareness)
├── slap_pipeline.py (MODIFIED - multiple representations)
├── shape_processor.py (existing)
├── smart_planner.py (existing)
├── integrated_intelligence.py (existing)
├── utils.py (MODIFIED - CGRA data structures)
│
├── constraint_governor.py (NEW)
├── executive_controller.py (NEW)
├── representation_selector.py (NEW)
├── procedural_toolkit.py (NEW)
│
├── cgra_architecture_diagram.md (NEW - this file)
│
└── tests/
    ├── unit/
    │   ├── test_constraint_governor.py (NEW)
    │   ├── test_executive_controller.py (NEW)
    │   ├── test_representation_selector.py (NEW)
    │   └── test_procedural_toolkit.py (NEW)
    ├── integration/
    │   └── test_cgra_compass_integration.py (NEW)
    └── e2e/
        ├── test_hierarchical_reasoning.py (NEW)
        ├── test_backward_chaining.py (NEW)
        └── test_enhanced_evaluation.py (NEW)
```
