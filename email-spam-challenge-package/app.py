from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import shutil
import subprocess
import json

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for local development

UPLOAD_FOLDER = 'submissions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        # Save uploaded files
        saved_files = []
        for file in files:
            if file.filename:
                filepath = os.path.join(temp_dir, file.filename)
                file.save(filepath)
                saved_files.append(file.filename)
        
        # Validate required files
        if 'classifier.py' not in saved_files:
            return jsonify({'error': 'Missing classifier.py'}), 400
        
        has_model = any('model' in f for f in saved_files)
        if not has_model:
            return jsonify({'error': 'Missing model file'}), 400
        
        # Copy test data to temp directory
        shutil.copy('spam_test.csv', os.path.join(temp_dir, 'spam_test.csv'))
        
        # Copy grading script
        shutil.copy('grade_submission.py', os.path.join(temp_dir, 'grade_submission.py'))
        
        # Run grading
        result = subprocess.run(
            ['python', 'grade_submission.py'],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Try to load results
        results_file = os.path.join(temp_dir, 'results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            # Grading failed
            return jsonify({
                'error': 'Grading failed',
                'stdout': result.stdout,
                'stderr': result.stderr
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Grading timed out (>2 minutes)'}), 500
    
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
    print("\nServer starting...")
    print("Open http://localhost:5000 in your browser")
    print("\nPress Ctrl+C to stop")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
