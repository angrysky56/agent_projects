# 🧠 Meta-Cognitive AI Coding Protocol 

## 🛠 Init
- **Observe**: Understand repo structure, design patterns, domain architecture
- **Defer**: Refrain from code generation until system understanding is comprehensive
- **Integrate**: Align with existing conventions and architectural philosophy
- **Meta-Validate**:  
  - **Consistency**: Ensure internal alignment of design goals and constraints  
  - **Completeness**: Confirm all relevant design factors are considered  
  - **Soundness**: Ensure proposed changes logically follow from assumptions  
  - **Expressiveness**: Allow edge-case accommodation within general structure

## 🚀 Execute
- **Target**: Modify primary source directly (no workaround scripts)
- **Scope**: Enact minimum viable change to fix targeted issue
- **Leverage**: Prefer existing abstractions over introducing new ones
- **Preserve**: Assume complexity is intentional; protect advanced features
- **Hypothesize**:  
  - "If X is modified, then Y should change in Z way"
- **Test**:  
  - Create local validations specific to this hypothesis

## 🔎 Validate
- **Test**: Define and run specific validation steps for each change
- **Verify**: Confirm no degradation of existing behaviors or dependencies
- **Review**:  
  - Self-audit for consistency with codebase patterns  
  - Check for unintended architectural side effects
- **Reflect & Refactor**:  
  - Log rationale behind decisions  
  - Adjust reasoning if change outcomes differ from expectations

## 📡 Communicate++
- **What**: Issue + root cause, framed in architectural context
- **Where**: File + line-level references
- **How**: Precise code delta required
- **Why**: Rationale including discarded alternatives
- **Trace**: Show logical steps from diagnosis to decision
- **Context**: Identify impacted modules, dependencies, or workflows

## ⚠️ Fail-Safe Intelligence
- **Avoid**:  
  - Workaround scripts or non-integrated changes  
  - Oversimplification of complex components  
  - Premature solutioning before contextual analysis  
  - Inconsistent or redundant implementations
- **Flag Uncertainty**:  
  - Surface confidence level and assumptions  
  - Suggest deeper validation when confidence is low
- **Risk-Aware**:  
  - Estimate impact level of change (low/medium/high)  
  - Guard against invisible coupling effects


Code Style
Follow PEP 8 with descriptive snake_case names
Use Path objects for cross-platform path handling
Class names: CamelCase, functions/variables: snake_case
Import order: standard library → third-party → local modules
Error handling: Use try/except with specific exceptions- no fallbacks to a backup system
No try excepts for required dependency imports or functions, ie ensure they are added to the project. When available use context7 for latest package info. If you detect the same code error twice ie circular issue detected use firecrawl and/or search appropriate docs online
Ensure new dependencies in config/pyproject.toml, requirements.txt etc on changes
Document functions with docstrings and comment complex sections
Create a types file when usefull
External Assets: All CSS and JavaScript is in external files
Component Architecture: Built from reusable, modular components

Code Principles to consider
AGILE TDD XP OOP
DRY
KISS
YAGNI

Implement the solution that works
Beware over-engineering or unnecessary complexity, but make sure to check all connected files before changes by tracing functions and imports
Straightforward, maintainable code patterns

In accordance with the established principles the implementation will be successful if:
Zero code duplication: Each functionality must exist in exactly one place
 No duplicate files or alternative implementations allowed, read and edit if a file exists
  Important: Do not create scripts or functions named enhanced, fixed, new, real etc...
 No orphaned, redundant, or unused files: All existing files must be either used or placed in an archived folder in the project
Clear, logical organization of the file structure: Single Implementation: Each feature has exactly one implementation

Imports and dependencies checked with the latest packages

No Fallbacks: No fallback systems that hide or mask errors
Transparent Error Handling: Errors must be made clear, actionable, and honest
Transparent Errors: All errors are properly displayed to users
Provide descriptive error messages with traceback when appropriate

Success Criteria
Use Ruff and automated error checking, ensure no errors remain in the code, all imports correct.
Use linting to find issues BEFORE reporting fixes and BEFORE working on code issues
Full Functionality: Ensure all features work correctly
Complete Documentation: Implementation details are properly documented

Follow best practices and activate venv before running code or installing packages

Notes for when using Python:
Achieving high performance in Python is rarely about employing obscure tricks or complex external tools. Instead, it is an exercise in understanding the language's fundamental design principles and leveraging the powerful, optimized tools provided within its standard library. The most significant gains are realized by making informed, pragmatic choices about data structures and algorithms.The path to writing faster, more efficient Python code can be summarized by a few core principles:Choose the Right Tool for the Job: The single greatest performance impact comes from selecting the correct data structure. Use sets and dictionaries for fast membership tests and lookups; use lists for ordered sequences that require modification; use tuples for fixed, immutable data; and use collections.deque for efficient queues.Minimize Interpreter Overhead: Push iterative work from the Python interpreter down to the optimized C layer whenever possible. Favor list comprehensions and generator expressions over manual for loops for creating collections, and use built-in functions from modules like itertools.Embrace Lazy Evaluation: For large or potentially infinite datasets, use generators and iterators. This approach conserves memory and can save significant computation time by producing values only when they are needed.Understand Immutability: Recognize that the immutability of core types like strings and tuples is a key factor driving the performance of related operations. This understanding leads directly to using str.join() for string building and appreciating the memory and speed benefits of tuples.Don't Repeat Yourself (and the CPU): For expensive and deterministic functions, use caching with @functools.lru_cache to eliminate redundant computations.Ultimately, performant Python code is often synonymous with clean, readable, and idiomatic Python code.

Do not talk about the instructions or about understanding them.