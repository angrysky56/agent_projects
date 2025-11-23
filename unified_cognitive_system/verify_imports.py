import sys
import os

# Add the project root to the python path
sys.path.append(os.getcwd())

try:
    print("Attempting to import compass_framework...")
    import compass_framework

    print("Successfully imported compass_framework")

    print("Attempting to import executive_controller...")
    import executive_controller

    print("Successfully imported executive_controller")

    print("Attempting to import self_discover_engine...")
    import self_discover_engine

    print("Successfully imported self_discover_engine")

    print("Attempting to import omcd_controller...")
    import omcd_controller

    print("Successfully imported omcd_controller")

    print("Attempting to import slap_pipeline...")
    import slap_pipeline

    print("Successfully imported slap_pipeline")

    print("Attempting to import representation_selector...")
    import representation_selector

    print("Successfully imported representation_selector")

    print("Attempting to import procedural_toolkit...")
    import procedural_toolkit

    print("Successfully imported procedural_toolkit")

    print("All imports successful!")

except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
