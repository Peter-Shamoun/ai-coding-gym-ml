import os
import shutil

def create_package():
    """
    Assemble the complete Email Spam Detection challenge package.
    """
    
    package_dir = 'email-spam-challenge-package'
    
    # Create clean directory
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)
    
    # Required files
    files_to_copy = [
        'index.html',
        'app.py',
        'grade_submission.py',
        'check_submission.py',
        'spam_train.csv',
        'spam_test.csv',
        'requirements.txt',
    ]
    
    print("Copying files...")
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, package_dir)
            print(f"  [OK] {file}")
        else:
            print(f"  [MISSING] {file}")
    
    # Create startup script (Unix/Mac)
    startup_sh_path = os.path.join(package_dir, 'start.sh')
    with open(startup_sh_path, 'w', newline='\n') as f:
        f.write('''#!/bin/bash
echo "=========================================="
echo "Email Spam Detection Challenge"
echo "=========================================="
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting server..."
python app.py
''')
    # Set executable permission (no-op on Windows, works on Unix)
    try:
        os.chmod(startup_sh_path, 0o755)
    except OSError:
        pass
    
    # Create startup script (Windows)
    with open(os.path.join(package_dir, 'start.bat'), 'w') as f:
        f.write('''@echo off
echo ==========================================
echo Email Spam Detection Challenge
echo ==========================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting server...
python app.py
pause
''')
    
    # Create comprehensive README
    with open(os.path.join(package_dir, 'README.md'), 'w') as f:
        f.write('''# Email Spam Detection Challenge

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
''')
    
    print(f"\n[OK] Package created: {package_dir}/")
    print(f"\nTo distribute:")
    print(f"  zip -r email-spam-challenge.zip {package_dir}/")
    print(f"\nOr deploy to a server and share the URL.")

if __name__ == '__main__':
    create_package()
