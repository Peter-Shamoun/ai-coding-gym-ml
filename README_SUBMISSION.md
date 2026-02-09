# Email Spam Detection - Submission Guide

## Required Files

| File | Required? | Description |
|------|-----------|-------------|
| `classifier.py` | **Yes** | Your training and prediction code |
| `model.pkl` or `model/` | **Yes** | Your saved trained model |
| `predictions.csv` | Recommended | Predictions on the test set |

## How to Check Your Submission

Before uploading, run the local checker:

```
python check_submission.py
```

This will verify:
- All required files are present
- Your model loads without errors
- `classifier.py` has valid syntax
- `predictions.csv` has the correct format and row count
- A quick accuracy estimate (if test data is available)

## What the Grading System Tests

Your submission is scored out of **100 points**. You need **80+ to pass**.

| Section | Points | What's Checked |
|---------|--------|----------------|
| **1. Validation** | 20 | `classifier.py` exists (5), model loads (10), `predictions.csv` exists (5) |
| **2. Accuracy** | 50 | Prediction accuracy on the test set (tiered scoring below) |
| **3. Precision/Recall** | 20 | Spam precision > 0.90 (10), spam recall > 0.90 (10) |
| **4. Code Quality** | 10 | Text preprocessing (3), class imbalance handling (3), feature engineering (4) |

### Accuracy Scoring Tiers

| Accuracy | Points |
|----------|--------|
| >= 95% | 50 |
| >= 93% | 40 |
| >= 91% | 30 |
| >= 89% | 20 |
| >= 85% | 10 |
| < 85% | 0 |

### Code Quality Checklist

The grader scans `classifier.py` for evidence of:

- **Text preprocessing** (3 pts): lowercasing, regex cleaning, stemming/lemmatization, stopword removal
- **Class imbalance handling** (3 pts): `class_weight='balanced'`, SMOTE, oversampling, undersampling
- **Feature engineering** (4 pts): `ngram_range`, `GridSearchCV`, `sublinear_tf`, `Pipeline`, ensembles, `min_df`/`max_df`

You need at least 2 feature engineering indicators to get the full 4 points.

## Predictions Format

`predictions.csv` should contain a column with one of these names (checked in order):

`prediction`, `label_num`, `predicted`, `pred`, `spam`, `label`

Values should be:
- Numeric: `0` (ham) or `1` (spam)
- Or string: `ham` or `spam`

Row count **must match** the test set (1,035 rows).

## Running the Grader Locally

```
python grade_submission.py ./ --test-data spam_test.csv
```

## Common Mistakes to Avoid

1. **Wrong row count** - Make sure `predictions.csv` has exactly 1,035 rows (one per test email). Don't include a header row in the count, but do include a header row in the file.

2. **Missing prediction column** - Name your prediction column `prediction` or `label_num`. Other names may not be recognized.

3. **Model won't load** - If you use `pickle.dump()` to save, make sure all dependencies are available when loading. Consider using `joblib` for sklearn models.

4. **String vs numeric labels** - Either `0`/`1` or `ham`/`spam` work, but don't mix them.

5. **Forgetting code quality patterns** - The grader does static analysis on `classifier.py`. Even if your model is great, you'll lose 10 points if your code doesn't show preprocessing, imbalance handling, and feature engineering.

6. **Empty model directory** - If using `model/` instead of `model.pkl`, make sure it actually contains files.

7. **SyntaxError in classifier.py** - Run `python classifier.py` to make sure it executes without errors before submitting.

8. **Not reaching 93% accuracy** - This is the threshold for 40/50 accuracy points. Combined with validation (20) and code quality (10), you need at least 93% accuracy and decent code to pass with 80+.
