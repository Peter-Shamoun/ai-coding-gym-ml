"""
Local submission checker for Email Spam Detection challenge.

Usage:
  python check_submission.py

This script checks if your submission files are ready:
- classifier.py exists
- model.pkl (or model/ directory) exists
- Can generate predictions.csv

Run this before submitting to catch common issues.
"""

import os
import sys
import pickle
import pandas as pd
import importlib.util


def check_files_exist():
    """Check if required files are present."""
    print("=" * 50)
    print("CHECKING FILES...")
    print("=" * 50)

    required_files = ['classifier.py']
    optional_files = ['model.pkl', 'model/', 'predictions.csv']

    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
            print(f"  MISSING: {f}")
        else:
            print(f"  Found:   {f}")

    # Check for model file or directory
    if not (os.path.exists('model.pkl') or os.path.exists('model/')):
        print(f"  MISSING: model.pkl or model/ directory")
        missing.append('model')
    else:
        if os.path.exists('model.pkl'):
            size_mb = os.path.getsize('model.pkl') / (1024 * 1024)
            print(f"  Found:   model.pkl ({size_mb:.1f} MB)")
        else:
            n_files = len(os.listdir('model/'))
            print(f"  Found:   model/ directory ({n_files} file(s))")

    # Check predictions.csv (optional but recommended)
    if os.path.exists('predictions.csv'):
        print(f"  Found:   predictions.csv")
    else:
        print(f"  Info:    predictions.csv not found (optional, generated during grading)")

    if missing:
        print(f"\n  FAILED: Missing required files: {', '.join(missing)}")
        return False

    print(f"\n  All required files present!")
    return True


def check_model_loads():
    """Try to load the model."""
    print("\n" + "=" * 50)
    print("CHECKING MODEL...")
    print("=" * 50)

    try:
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            model_type = type(model).__name__
            print(f"  Model loaded successfully from model.pkl")
            print(f"  Model type: {model_type}")

            # Check for common sklearn model attributes
            if hasattr(model, 'predict'):
                print(f"  Model has predict() method")
            else:
                print(f"  WARNING: Model does not have predict() method")

        elif os.path.exists('model/'):
            contents = os.listdir('model/')
            if not contents:
                print(f"  FAILED: model/ directory is empty")
                return False
            print(f"  Found model/ directory with {len(contents)} file(s)")
            for item in contents[:5]:
                print(f"    - {item}")
            if len(contents) > 5:
                print(f"    ... and {len(contents) - 5} more")
        else:
            print(f"  FAILED: No model file or directory found")
            return False

        return True

    except Exception as e:
        print(f"  FAILED: Could not load model: {e}")
        return False


def check_classifier_syntax():
    """Check that classifier.py has valid Python syntax."""
    print("\n" + "=" * 50)
    print("CHECKING CLASSIFIER CODE...")
    print("=" * 50)

    if not os.path.exists('classifier.py'):
        print(f"  SKIPPED: classifier.py not found")
        return False

    try:
        with open('classifier.py', 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()

        import ast
        ast.parse(source)
        loc = len([l for l in source.splitlines() if l.strip() and not l.strip().startswith('#')])
        print(f"  Syntax OK ({loc} lines of code)")
    except SyntaxError as e:
        print(f"  FAILED: SyntaxError at line {e.lineno}: {e.msg}")
        return False

    # Check for key patterns the grader looks for
    import re

    checks = {
        'Text preprocessing': [r'\bclean', r'\blower\s*\(', r'\bre\.sub\b',
                                r'\bstemm', r'\blemmat', r'\bstopwords\b',
                                r'\bpreprocess', r'\btokeniz'],
        'Class imbalance handling': [r'\bclass_weight\b', r'\bbalanced\b',
                                      r'\bSMOTE\b', r'\bover_?sampl',
                                      r'\bunder_?sampl', r'\bresample\b'],
        'Feature engineering': [r'\bngram_range\b', r'\bGridSearchCV\b',
                                 r'\bcross_val', r'\bsublinear_tf\b',
                                 r'\bPipeline\b', r'\bmax_features\b',
                                 r'\bmin_df\b', r'\bmax_df\b',
                                 r'\bensemble\b'],
    }

    for category, patterns in checks.items():
        found = any(re.search(p, source, re.IGNORECASE) for p in patterns)
        if found:
            print(f"  Found:   {category} (good for code quality score)")
        else:
            print(f"  Missing: {category} (costs you code quality points)")
            if category == 'Text preprocessing':
                print(f"           Tip: Add lowercasing, regex cleaning, stop-word removal")
            elif category == 'Class imbalance handling':
                print(f"           Tip: Use class_weight='balanced' or SMOTE")
            elif category == 'Feature engineering':
                print(f"           Tip: Try ngram_range, GridSearchCV, sublinear_tf")

    return True


def check_predictions_format():
    """Check if predictions.csv has correct format."""
    print("\n" + "=" * 50)
    print("CHECKING PREDICTIONS FORMAT...")
    print("=" * 50)

    if not os.path.exists('predictions.csv'):
        print("  predictions.csv not found (will be generated during grading)")
        return True

    try:
        df = pd.read_csv('predictions.csv')
    except Exception as e:
        print(f"  FAILED: Error reading predictions.csv: {e}")
        return False

    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Check for accepted prediction column names (matches grader logic)
    accepted_cols = ['prediction', 'label_num', 'predicted', 'pred', 'spam', 'label']
    pred_col = None
    for col in accepted_cols:
        if col in df.columns:
            pred_col = col
            break

    if pred_col is None:
        print(f"  FAILED: No recognized prediction column found")
        print(f"           Expected one of: {accepted_cols}")
        return False

    print(f"  Using prediction column: '{pred_col}'")

    # Check values
    unique_vals = df[pred_col].unique()
    print(f"  Unique values: {sorted(unique_vals)}")

    # Validate values are 0/1 or spam/ham
    valid_numeric = set(unique_vals).issubset({0, 1})
    valid_string = set(str(v).strip().lower() for v in unique_vals).issubset({'spam', 'ham'})

    if not (valid_numeric or valid_string):
        print(f"  WARNING: Values should be 0/1 or 'spam'/'ham'")

    # Check row count against test set
    if os.path.exists('spam_test.csv'):
        test_df = pd.read_csv('spam_test.csv')
        if len(df) != len(test_df):
            print(f"  FAILED: Row count mismatch!")
            print(f"           predictions.csv has {len(df)} rows")
            print(f"           spam_test.csv has {len(test_df)} rows")
            return False
        else:
            print(f"  Row count matches test set ({len(test_df)} rows)")

    # Show class distribution
    if valid_numeric:
        spam_count = (df[pred_col] == 1).sum()
        ham_count = (df[pred_col] == 0).sum()
    elif valid_string:
        spam_count = df[pred_col].str.lower().str.strip().eq('spam').sum()
        ham_count = df[pred_col].str.lower().str.strip().eq('ham').sum()
    else:
        spam_count = ham_count = '?'

    print(f"  Predicted: {ham_count} ham, {spam_count} spam")

    print(f"\n  Predictions format looks good!")
    return True


def check_quick_accuracy():
    """If both predictions.csv and spam_test.csv exist, compute accuracy."""
    print("\n" + "=" * 50)
    print("QUICK ACCURACY CHECK...")
    print("=" * 50)

    if not os.path.exists('predictions.csv') or not os.path.exists('spam_test.csv'):
        print("  SKIPPED: Need both predictions.csv and spam_test.csv")
        return True

    try:
        import numpy as np
        pred_df = pd.read_csv('predictions.csv')
        test_df = pd.read_csv('spam_test.csv')

        # Find prediction column
        pred_col = None
        for col in ['prediction', 'label_num', 'predicted', 'pred', 'spam', 'label']:
            if col in pred_df.columns:
                pred_col = col
                break

        if pred_col is None or len(pred_df) != len(test_df):
            print("  SKIPPED: Cannot compute accuracy")
            return True

        y_pred = pred_df[pred_col].values
        y_true = test_df['label_num'].values

        # Handle string labels
        if y_pred.dtype == object:
            mapping = {'spam': 1, 'ham': 0}
            y_pred = np.array([mapping[str(v).strip().lower()] for v in y_pred])
        else:
            y_pred = y_pred.astype(int)

        accuracy = np.mean(y_pred == y_true)
        pct = accuracy * 100

        print(f"  Accuracy: {pct:.2f}%")
        print()

        # Show expected grade based on grading thresholds
        if accuracy >= 0.95:
            pts = 50
        elif accuracy >= 0.93:
            pts = 40
        elif accuracy >= 0.91:
            pts = 30
        elif accuracy >= 0.89:
            pts = 20
        elif accuracy >= 0.85:
            pts = 10
        else:
            pts = 0

        print(f"  Expected accuracy score: {pts}/50 pts")
        print(f"  Thresholds: >=95%->50  >=93%->40  >=91%->30  >=89%->20  >=85%->10")

        if accuracy < 0.93:
            print(f"\n  TIP: Target is >93% accuracy for a passing grade (80+ total)")

    except Exception as e:
        print(f"  Could not compute accuracy: {e}")

    return True


def main():
    print("\n" + "=" * 50)
    print("  EMAIL SPAM DETECTION - SUBMISSION CHECKER")
    print("=" * 50 + "\n")

    all_good = True

    # Run all checks
    if not check_files_exist():
        all_good = False

    if not check_model_loads():
        all_good = False

    if not check_classifier_syntax():
        all_good = False

    if not check_predictions_format():
        all_good = False

    # Bonus: quick accuracy check
    check_quick_accuracy()

    # Final summary
    print("\n" + "=" * 50)
    if all_good:
        print("  SUBMISSION READY!")
        print("=" * 50)
        print("\nNext step: Upload your files to the grading system")
        print("Required files:")
        print("  - classifier.py")
        print("  - model.pkl (or model/ directory)")
        print("  - predictions.csv (optional, will be generated)")
    else:
        print("  ISSUES FOUND - FIX BEFORE SUBMITTING")
        print("=" * 50)
        print("\nPlease fix the issues above before submitting.")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
