from flask import Flask, render_template, jsonify, request
import os

app = Flask(__name__)

# Define the base directory you want to open
BASE_DIRECTORY = "./experiments"  # <-- Set this to your actual directory
FIXED_FILE = "fixed_file.txt"  # Replace with the file name you want to display

def safe_join(base, *paths):
    """Safely join paths and prevent path traversal outside the base directory."""
    # Convert base and target path to absolute paths
    base = os.path.abspath(base)
    target_path = os.path.abspath(os.path.join(base, *paths))
    print(f"Joining path: base={base}, paths={paths}, target_path={target_path}")
    
    # Check if target_path is within the base directory
    if os.path.commonpath([target_path, base]) == base:
        return target_path
    return None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files')
def list_files():
    # Get and sanitize the subdirectory parameter
    subdirectory = request.args.get('subdir', '').strip('/')
    directory_path = safe_join(BASE_DIRECTORY, subdirectory) if subdirectory else BASE_DIRECTORY
    
    print(f"Requested directory path: {directory_path}")
    if directory_path is None or not os.path.isdir(directory_path):
        print("Directory path is invalid or does not exist.")
        return jsonify({"error": "Invalid directory path"}), 400

    try:
        items = []
        for entry in os.listdir(directory_path):
            full_path = os.path.join(directory_path, entry)
            items.append({
                "name": entry,
                "is_dir": os.path.isdir(full_path),
            })
        return jsonify({"path": subdirectory, "items": items})
    except Exception as e:
        print(f"Error listing files: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/file-content/<path:filename>')
def get_file_content(filename):
    file_path = safe_join(BASE_DIRECTORY, filename)
    if file_path is None or not os.path.isfile(file_path):
        return jsonify({"error": "Invalid file path"}), 400

    try:
        with open(file_path, 'r') as f:
            content = f.read()
        words = content.split()
        return jsonify(words)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/fixed-file-content')
def fixed_file_content():
    fixed_file_path = safe_join(BASE_DIRECTORY, FIXED_FILE)
    print(f"Fixed file path: {fixed_file_path}")
    
    if fixed_file_path is None or not os.path.isfile(fixed_file_path):
        print("Fixed file not found or path is invalid.")
        return jsonify({"error": "Fixed file not found"}), 400

    try:
        with open(fixed_file_path, 'r') as f:
            content = f.read()
        words = content.split()
        return jsonify(words)
    except Exception as e:
        print(f"Error reading fixed file: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

