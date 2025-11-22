"""
Example: Complex Multi-Stage Reasoning with COMPASS

Demonstrates advanced reasoning capabilities with custom module selection,
resource optimization, and detailed analysis.
"""

from compass_framework import create_compass
from config import create_custom_config
from self_discover_engine import REASONING_MODULES
import json


def complex_research_task():
    """Demonstrates complex research and analysis task."""
    print("=" * 70)
    print("Complex Research Task")
    print("=" * 70)

    # Configure for deep reasoning
    config = create_custom_config(
        omcd={
            "R": 20.0,  # High importance
            "alpha": 0.1,  # Lower cost
            "min_confidence": 0.7,
        },
        self_discover={"max_trials": 15, "module_selection_strategy": "adaptive"},
        slap={"alpha": 0.45, "beta": 0.55},
    )

    compass = create_compass(config)

    task = """
    Analyze the trade-offs between different sorting algorithms
    (quicksort, mergesort, heapsort) in terms of time complexity,
    space complexity, and stability. Provide recommendations for
    different use cases.
    """

    context = {"domain": "algorithm_analysis", "depth": "comprehensive", "output_format": "structured_comparison"}

    result = compass.process_task(task, context, max_iterations=15)

    print(f"\n✓ Analysis completed in {result['iterations']} iterations")
    print(f"  Final confidence: {result['score']:.1%}")

    # Analyze the reasoning process
    print(f"\n🧠 Reasoning Process:")
    print(f"  Total reflections: {len(result['reflections'])}")
    print(f"  Resources utilized: {result['resources_used']:.2f}")

    # Show advancement through SLAP stages
    print(f"\n📊 SLAP Advancement Analysis:")
    if "trajectory" in result:
        print(f"  Semantic progression stages completed")

    return result


def creative_design_task():
    """Demonstrates creative problem-solving."""
    print("\n" + "=" * 70)
    print("Creative Design Task")
    print("=" * 70)

    config = create_custom_config(self_discover={"module_selection_strategy": "adaptive", "max_trials": 12})

    compass = create_compass(config)

    task = """
    Design an innovative data structure that combines the benefits
    of arrays (fast random access) and linked lists (efficient
    insertion/deletion). Consider novel approaches and trade-offs.
    """

    result = compass.process_task(task)

    print(f"\n✓ Design completed: Score {result['score']:.3f}")

    # Analyze creative insights
    print(f"\n💡 Creative Insights:")
    for i, reflection in enumerate(result["reflections"][:3], 1):
        print(f"\n  Iteration {i} insights:")
        for insight in reflection["insights"]:
            print(f"    • {insight}")

    return result


def optimization_under_constraints():
    """Demonstrates optimization with strict resource constraints."""
    print("\n" + "=" * 70)
    print("Optimization Under Constraints")
    print("=" * 70)

    config = create_custom_config(
        omcd={
            "max_resources": 50.0,  # Limited resources
            "alpha": 0.2,  # Higher cost
            "nu": 2.5,  # Steeper cost curve
        },
        self_discover={"pass_threshold": 0.75, "max_trials": 10},
    )

    compass = create_compass(config)

    task = """
    Optimize a recommendation system to handle 1M+ users with
    limited computational resources. Balance accuracy, speed,
    and scalability.
    """

    context = {"constraints": ["limited_memory", "low_latency", "high_scale"], "metrics": ["accuracy", "throughput", "latency"]}

    result = compass.process_task(task, context)

    print(f"\n✓ Optimization completed")
    print(f"  Success: {result['success']}")
    print(f"  Resource efficiency: {(50 - result['resources_used']) / 50:.1%} remaining")
    print(f"  Quality score: {result['score']:.3f}")

    # Resource allocation analysis
    print(f"\n⚡ Resource Management:")
    print(f"  Total available: 50.0")
    print(f"  Total used: {result['resources_used']:.2f}")
    print(f"  Avg per iteration: {result['resources_used'] / result['iterations']:.2f}")

    return result


def multi_objective_planning():
    """Demonstrates handling multiple competing objectives."""
    print("\n" + "=" * 70)
    print("Multi-Objective Planning")
    print("=" * 70)

    compass = create_compass()

    task = """
    Design a distributed caching system that optimizes for:
    1. Low latency (< 10ms)
    2. High availability (99.99%)
    3. Cost efficiency
    4. Easy maintenance
    Balance these competing objectives and justify trade-offs.
    """

    result = compass.process_task(task)

    print(f"\n✓ Planning completed: {result['score']:.1%} quality")

    # Analyze objectives
    print(f"\n🎯 Objective Analysis:")
    for obj in result["objectives"]:
        progress_bar = "█" * int(obj["progress"] / 10) + "░" * (10 - int(obj["progress"] / 10))
        print(f"  {obj['name']:<25} [{progress_bar}] {obj['progress']:.1f}%")

    # Show SMART planning effectiveness
    print(f"\n📋 SMART Planning Metrics:")
    total_objectives = len(result["objectives"])
    on_track = sum(1 for obj in result["objectives"] if obj["is_on_track"])
    print(f"  Objectives created: {total_objectives}")
    print(f"  On track: {on_track}/{total_objectives}")
    print(f"  Success rate: {on_track / total_objectives:.1%}")

    return result


def adaptive_learning_demonstration():
    """Shows how the system learns and adapts over multiple tasks."""
    print("\n" + "=" * 70)
    print("Adaptive Learning Demonstration")
    print("=" * 70)

    compass = create_compass()

    # Series of related tasks
    tasks = ["Implement a basic stack data structure", "Extend the stack to support min() operation in O(1)", "Further extend to support getMax() in O(1)", "Optimize the enhanced stack for memory efficiency"]

    results = []

    print(f"\n📚 Processing {len(tasks)} progressive tasks...\n")

    for i, task in enumerate(tasks, 1):
        result = compass.process_task(task)
        results.append(result)

        print(f"Task {i}: {task[:50]}...")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Iterations: {result['iterations']}")

        # Show learning progression
        if i > 1:
            score_improvement = result["score"] - results[i - 2]["score"]
            iter_improvement = results[i - 2]["iterations"] - result["iterations"]
            print(f"  Improvement: {score_improvement:+.3f} score, {iter_improvement:+d} iterations")
        print()

    # Analyze overall learning
    status = compass.get_status()

    print(f"📈 Learning Analysis:")
    print(f"  Total reflections accumulated: {status['reflections_count']}")
    print(f"  Average quality: {status['average_score']:.3f}")
    print(f"  Knowledge reuse evidenced by iteration reduction")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 10 + "COMPASS Framework - Complex Reasoning Examples")
    print("=" * 70 + "\n")

    # Run complex examples
    complex_research_task()
    creative_design_task()
    optimization_under_constraints()
    multi_objective_planning()
    adaptive_learning_demonstration()

    print("\n" + "=" * 70)
    print("All complex reasoning examples completed!")
    print("=" * 70 + "\n")
