# Running the Email Spam Detection Challenge

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place these files in the same directory:
- index.html
- app.py
- grade_submission.py
- check_submission.py
- spam_train.csv
- spam_test.csv

## Start the Server
```bash
python app.py
```

Then open http://localhost:5000 in your browser.

## How It Works

1. User downloads datasets from the webpage
2. User builds their classifier locally
3. User uploads classifier.py and model.pkl
4. Backend runs grading script and returns results
5. Results shown immediately on webpage

All grading happens server-side in isolated temporary directories.
