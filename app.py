from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import tempfile
import shutil

# ── Configuration ───────────────────────────────────────────
DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')
PORT = int(os.environ.get('PORT', 5000))
MAX_CONTENT_LENGTH = int(os.environ.get('MAX_UPLOAD_MB', 16)) * 1024 * 1024  # default 16 MB

app = Flask(__name__, static_folder='.')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
CORS(app)

UPLOAD_FOLDER = 'submissions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add project root so we can import the grader directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade_submission import grade_submission as run_grader

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, datasets)."""
    return send_from_directory('.', path)

@app.route('/api/grade', methods=['POST'])
def grade_submission():
    """
    Handle file upload and grading.
    
    Expected files:
    - classifier.py
    - model.pkl (or model files)
    """
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    # Create temporary directory for this submission
    temp_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
    
    try:
        # Save uploaded files -- sanitize filenames to prevent path traversal
        saved_files = []
        for file in files:
            if file.filename:
                safe_name = os.path.basename(file.filename)
                if not safe_name:
                    continue
                filepath = os.path.join(temp_dir, safe_name)
                file.save(filepath)
                saved_files.append(safe_name)
        
        # Validate required files
        if 'classifier.py' not in saved_files:
            return jsonify({'error': 'Missing classifier.py'}), 400
        
        has_model = any('model' in f for f in saved_files)
        if not has_model:
            return jsonify({'error': 'Missing model file'}), 400
        
        # Path to the test CSV (in project root)
        test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'spam_test.csv'
        )
        
        # Run grading directly (no subprocess needed)
        results = run_grader(temp_dir, test_data_path)
        
        # Remove the verbose feedback string to keep the JSON response lean
        results.pop('feedback', None)
        return jsonify(results)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("=" * 50)
    print("AI Coding Gym - Email Spam Detection")
    print("=" * 50)
    print(f"\nServer starting (debug={DEBUG})...")
    print(f"Open http://localhost:{PORT} in your browser")
    print("\nPress Ctrl+C to stop")
    print("=" * 50 + "\n")
    
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
