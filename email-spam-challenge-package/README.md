# Email Spam Detection Challenge

## Quick Start

### Option 1: One-Click Start

**Mac/Linux:**
```bash
./start.sh
```

**Windows:**
```
start.bat
```

### Option 2: Manual Start
```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Files Included

- `index.html` - Challenge webpage
- `app.py` - Grading server
- `grade_submission.py` - Automated grading logic
- `check_submission.py` - Local validation script
- `spam_train.csv` - Training dataset (~4000 emails)
- `spam_test.csv` - Test dataset (~1000 emails)
- `requirements.txt` - Python dependencies

## For Challenge Takers

1. Download datasets from the webpage
2. Build your classifier
3. Test locally with `check_submission.py`
4. Upload to get graded

## For Challenge Hosts

Run `python app.py` to start the grading server.

Users access http://localhost:5000 or your deployed URL.

## System Requirements

- Python 3.8+
- 2GB RAM minimum
- Internet connection (for pip install)

## Support

Target accuracy: >= 93%
Estimated time: 2 hours
Difficulty: Intermediate
