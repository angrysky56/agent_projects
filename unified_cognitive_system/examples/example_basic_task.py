"""
Example: Basic Task Execution with COMPASS

This example demonstrates the simplest way to use COMPASS for task solving.
"""

from compass_framework import quick_solve, create_compass
import json


def example_1_quick_solve():
    """Example 1: One-liner task solving."""
    print("=" * 70)
    print("Example 1: Quick Solve")
    print("=" * 70)

    result = quick_solve("Design a simple sorting algorithm")

    print(f"\nSuccess: {result['success']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nSolution Preview:")
    print(json.dumps(result["solution"], indent=2)[:500])
    print("\n")


def example_2_basic_configuration():
    """Example 2: Basic task with custom configuration."""
    print("=" * 70)
    print("Example 2: Basic Configuration")
    print("=" * 70)

    from config import create_custom_config

    # Create custom config with higher importance weight
    config = create_custom_config(omcd={"R": 15.0, "alpha": 0.15}, self_discover={"max_trials": 8})

    compass = create_compass(config)

    result = compass.process_task("Optimize a search algorithm for large datasets")

    print(f"\nTask processed in {result['iterations']} iterations")
    print(f"Final score: {result['score']:.3f}")
    print(f"Resources used: {result['resources_used']:.2f}")
    print(f"Number of reflections: {len(result['reflections'])}")
    print("\n")


def example_3_with_context():
    """Example 3: Task with additional context."""
    print("=" * 70)
    print("Example 3: Task with Context")
    print("=" * 70)

    compass = create_compass()

    context = {"domain": "data_structures", "constraints": ["memory efficient", "fast lookups"], "requirements": ["thread-safe", "scalable"]}

    result = compass.process_task(task_description="Design a hash table implementation", context=context, max_iterations=12)

    print(f"\nContext-aware processing completed")
    print(f"Success: {result['success']}")
    print(f"Score: {result['score']:.3f}")

    print(f"\nObjectives created: {len(result['objectives'])}")
    for obj in result["objectives"]:
        print(f"  - {obj['name']}: {obj['progress']:.1f}% complete")

    print("\n")


def example_4_multiple_tasks():
    """Example 4: Processing multiple tasks with same instance."""
    print("=" * 70)
    print("Example 4: Multiple Tasks")
    print("=" * 70)

    compass = create_compass()

    tasks = ["Create a linked list data structure", "Implement binary search", "Design a priority queue"]

    results = []
    for task in tasks:
        result = compass.process_task(task)
        results.append(result)
        print(f"✓ {task}: Score={result['score']:.3f}")

    # Get overall status
    status = compass.get_status()

    print(f"\nOverall Statistics:")
    print(f"  Total trajectories: {status['trajectories_count']}")
    print(f"  Total reflections: {status['reflections_count']}")
    print(f"  Average score: {status['average_score']:.3f}")
    print(f"  Resource utilization: {status['resource_utilization']:.1%}")
    print("\n")


def example_5_inspect_results():
    """Example 5: Detailed result inspection."""
    print("=" * 70)
    print("Example 5: Detailed Result Inspection")
    print("=" * 70)

    compass = create_compass()

    result = compass.process_task("Analyze the time complexity of merge sort")

    print(f"\n📊 Result Details:")
    print(f"  Success: {result['success']}")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Iterations: {result['iterations']}")

    print(f"\n🎯 Objectives:")
    for obj in result["objectives"]:
        status = "✓" if obj["is_on_track"] else "✗"
        print(f"  {status} {obj['name']}: {obj['progress']:.1f}%")

    print(f"\n💭 Reflections:")
    for i, reflection in enumerate(result["reflections"][:3], 1):
        print(f"\n  Reflection {i}:")
        print(f"    Insights: {len(reflection['insights'])}")
        for insight in reflection["insights"][:2]:
            print(f"      - {insight}")

    print(f"\n📈 Trajectory:")
    print(f"  Total steps: {result['trajectory']['length']}")
    print(f"  Final score: {result['trajectory']['score']:.3f}")

    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 15 + "COMPASS Framework - Basic Examples")
    print("=" * 70 + "\n")

    # Run all examples
    example_1_quick_solve()
    example_2_basic_configuration()
    example_3_with_context()
    example_4_multiple_tasks()
    example_5_inspect_results()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")
