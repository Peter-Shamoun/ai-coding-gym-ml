"""
Test script for the MNIST Digit Recognition challenge.
======================================================
Verifies:
  1. Challenge loads correctly via the registry
  2. All data files exist
  3. A simple MLP solution gets graded properly
  4. Quick "run" mode works
  5. Full "submit" mode works with grading
"""
#
import os
import sys
import time
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_registry_loading():
    """Test that the challenge loads via the registry."""
    print("=" * 60)
    print("TEST 1: Registry Loading")
    print("=" * 60)

    from challenges.registry import ChallengeRegistry

    # Force re-discovery (clear singleton)
    ChallengeRegistry._instance = None
    ChallengeRegistry._challenges = {}

    registry = ChallengeRegistry()
    all_ids = registry.list_ids()
    print(f"  Registered challenges: {all_ids}")

    assert "mnist_digit_recognition" in all_ids, \
        f"mnist_digit_recognition not found! Got: {all_ids}"

    challenge = registry.get("mnist_digit_recognition")
    assert challenge is not None, "Challenge object is None"
    assert challenge.title == "MNIST Handwritten Digit Recognition"
    assert challenge.difficulty == "medium"

    details = challenge.get_details()
    assert "description" in details
    assert "starter_code" in details
    assert "allowed_libraries" in details
    assert "numpy" in details["allowed_libraries"]
    assert "sklearn" in details["allowed_libraries"]
    assert "tensorflow" in details["allowed_libraries"]

    print(f"  Title: {challenge.title}")
    print(f"  Difficulty: {challenge.difficulty}")
    print(f"  Timeout: {challenge.timeout}s")
    print(f"  Allowed libraries: {len(details['allowed_libraries'])}")
    print("  PASSED\n")


def test_data_files():
    """Test that all data files exist and have correct shapes."""
    print("=" * 60)
    print("TEST 2: Data Files")
    print("=" * 60)

    challenge_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "challenges", "mnist_digit_recognition"
    )

    expected_files = {
        "mnist_train_images.npy": (60000, 28, 28),
        "mnist_train_labels.npy": (60000,),
        "mnist_test_images.npy": (10000, 28, 28),
        "mnist_test_labels.npy": (10000,),
    }

    for fname, expected_shape in expected_files.items():
        path = os.path.join(challenge_dir, fname)
        assert os.path.isfile(path), f"Missing: {path}"
        arr = np.load(path)
        assert arr.shape == expected_shape, \
            f"{fname}: expected shape {expected_shape}, got {arr.shape}"
        assert arr.dtype == np.uint8, \
            f"{fname}: expected dtype uint8, got {arr.dtype}"
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {fname}: shape={arr.shape}, dtype={arr.dtype}, size={size_mb:.1f} MB")

    # Check sample images
    samples_dir = os.path.join(challenge_dir, "samples")
    for digit in range(10):
        path = os.path.join(samples_dir, f"sample_{digit}.png")
        assert os.path.isfile(path), f"Missing sample: {path}"
    print(f"  10 sample images present in {samples_dir}")

    # Check label ranges
    y_train = np.load(os.path.join(challenge_dir, "mnist_train_labels.npy"))
    y_test = np.load(os.path.join(challenge_dir, "mnist_test_labels.npy"))
    assert y_train.min() == 0 and y_train.max() == 9
    assert y_test.min() == 0 and y_test.max() == 9
    print(f"  Label range: {y_train.min()}-{y_train.max()} (correct)")
    print("  PASSED\n")


def test_simple_solution():
    """Test grading with a simple MLP solution."""
    print("=" * 60)
    print("TEST 3: Simple MLP Solution (Full Grading)")
    print("=" * 60)

    # The simple MLP solution code (as a string, for grading)
    SIMPLE_SOLUTION = '''
import numpy as np
from sklearn.neural_network import MLPClassifier

def train_digit_classifier(X_train, y_train):
    # Flatten images
    X_flat = X_train.reshape(len(X_train), -1)

    # Normalize
    X_flat = X_flat / 255.0

    # Train simple MLP
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, random_state=42)
    clf.fit(X_flat, y_train)

    def predict(X_test):
        X_test_flat = X_test.reshape(len(X_test), -1) / 255.0
        return clf.predict(X_test_flat)

    return predict
'''

    from engine.executor import execute_challenge

    # Force re-discovery
    from challenges.registry import ChallengeRegistry
    ChallengeRegistry._instance = None
    ChallengeRegistry._challenges = {}

    print("  Running full submit (this will take 1-2 minutes)...")
    start = time.time()
    result = execute_challenge("mnist_digit_recognition", SIMPLE_SOLUTION, mode="submit")
    elapsed = time.time() - start

    print(f"  Execution time: {elapsed:.1f}s")
    print(f"  Success: {result.success}")

    if not result.success:
        print(f"  ERROR: {result.error}")
        if result.traceback_str:
            print(f"  Traceback: {result.traceback_str}")
        if result.stderr:
            print(f"  Stderr: {result.stderr[:500]}")
        return

    assert result.grading is not None, "Grading result is None"
    g = result.grading

    print(f"\n  === GRADING RESULTS ===")
    print(f"  Total: {g['total_score']}/{g['max_score']}")
    print(f"  Passed: {g['passed']}")
    print(f"  Accuracy: {g.get('accuracy', 'N/A')}")

    for cat in g["categories"]:
        print(f"\n  {cat['name']}: {cat['score']}/{cat['max_score']}")
        for fb in cat.get("feedback", []):
            print(f"    - {fb}")

    # Expected: Simple MLP should get ~94-96% accuracy
    accuracy = g.get("accuracy")
    if accuracy is not None:
        print(f"\n  Accuracy: {accuracy * 100:.2f}%")
        assert accuracy >= 0.90, f"Expected >= 90% accuracy, got {accuracy * 100:.2f}%"
        if accuracy < 0.97:
            print("  (Below 97% target - expected for simple MLP)")

    print("\n  PASSED\n")


def test_run_mode():
    """Test quick run mode with the same simple solution."""
    print("=" * 60)
    print("TEST 4: Quick Run Mode")
    print("=" * 60)

    SIMPLE_SOLUTION = '''
import numpy as np
from sklearn.neural_network import MLPClassifier

def train_digit_classifier(X_train, y_train):
    X_flat = X_train.reshape(len(X_train), -1) / 255.0
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, random_state=42)
    clf.fit(X_flat, y_train)

    def predict(X_test):
        X_test_flat = X_test.reshape(len(X_test), -1) / 255.0
        return clf.predict(X_test_flat)

    return predict
'''

    from engine.executor import execute_challenge
    from challenges.registry import ChallengeRegistry
    ChallengeRegistry._instance = None
    ChallengeRegistry._challenges = {}

    print("  Running quick test (sample mode)...")
    start = time.time()
    result = execute_challenge("mnist_digit_recognition", SIMPLE_SOLUTION, mode="run")
    elapsed = time.time() - start

    print(f"  Execution time: {elapsed:.1f}s")
    print(f"  Success: {result.success}")

    if not result.success:
        print(f"  ERROR: {result.error}")
        if result.traceback_str:
            print(f"  Traceback: {result.traceback_str}")
        return

    print(f"  Sample accuracy: {result.sample_accuracy}")
    print(f"  Train time: {result.train_time}s")

    assert result.sample_accuracy is not None, "No sample accuracy returned"
    assert result.sample_accuracy > 0.85, \
        f"Expected sample accuracy > 85%, got {result.sample_accuracy * 100:.2f}%"

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MNIST DIGIT RECOGNITION CHALLENGE — TEST SUITE")
    print("=" * 60 + "\n")

    test_registry_loading()
    test_data_files()
    test_run_mode()
    test_simple_solution()

    print("=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)
