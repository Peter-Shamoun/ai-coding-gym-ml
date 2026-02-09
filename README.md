# Email Spam Detection

## Problem Statement

Build a binary classifier that identifies whether an email is **spam** or **ham** (not spam).

The dataset contains approximately 5,000 real emails from the Enron corpus. Your goal is to train a model that achieves **greater than 93% accuracy** on the held-out test set.

## Dataset

| File | Description |
|------|-------------|
| `spam_train.csv` | Training set (~4,136 emails) |
| `spam_test.csv` | Test set (~1,035 emails) |

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Full email text |
| `label` | string | `spam` or `ham` |
| `label_num` | int | `1` = spam, `0` = ham |

## Requirements

- Build a binary classifier (spam vs ham)
- Train on `spam_train.csv`
- Generate predictions for `spam_test.csv`
- Achieve **>93% accuracy** on the test set

## Deliverables

| File | Description |
|------|-------------|
| `classifier.py` | Training and prediction code |
| `model.pkl` or `model/` | Saved trained model |
| `predictions.csv` | Test set predictions with columns: `text`, `predicted_label` |
| `report.txt` | Single line with your test accuracy score |

## Evaluation

Your submission will be evaluated on **accuracy** — the percentage of correctly classified emails in the test set. The target is **>93%**.
