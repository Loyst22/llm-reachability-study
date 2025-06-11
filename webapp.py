from flask import Flask, render_template, jsonify, request, send_file, g
import os

app = Flask(__name__)

# Define the base directory you want to open
BASE_DIRECTORY = "./experiments"  # <-- Set this to your actual directory
FIXED_FILE = "fixed_file.txt"  # Replace with the file name you want to display

fixed_file_path = None

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

import urllib.parse
import os

import os
import urllib.parse

def update_fixed_file_path(subdir):
    """
    Search for 'theClass.java' in the specified subdirectory.
    If found, update the global fixed_file_path to point to this file.
    """
    global fixed_file_path
    target_file = 'theClass.java'

    # Decode any URL encoding in the path and join with base directory
    decoded_subdir = urllib.parse.unquote(subdir)
    combined_path = os.path.join(BASE_DIRECTORY, decoded_subdir)  # Join with base directory
    absolute_path = os.path.abspath(combined_path)  # Convert to absolute path
    
    print(f"Checking absolute directory path: {absolute_path}")
    
    # Verify that absolute_path is a valid directory
    if not os.path.isdir(absolute_path):
        print(f"Provided path is not a valid directory: {absolute_path}")
        return
    
    # Traverse the subdirectory to find the target file
    for root, dirs, files in os.walk(absolute_path):
        print(f"Inspecting {root} - Files: {files}")
        
        if target_file in files:
            g.fixed_file_path = os.path.join(root, target_file)
            fixed_file_path = os.path.join(root, target_file)
            print(f"!!!! \n\nFixed file path updated to: {g.fixed_file_path}")
            return  # Exit after finding the first instance

    print("No 'theClass.java' file found in the specified directory.")


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
    
    update_fixed_file_path(subdirectory)

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
            words = []
            content = f.readlines()
            for line in content:
                words.extend(line.split())
                words.append("\n")
            return jsonify(words)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/fixed-file-content', methods=['GET'])
def fixed_file_content():
    """
    Return the content of the fixed file if it exists, otherwise return an error.
    """
    print("!!!!! Updating file contents")
    file_path = getattr(g, 'fixed_file_path', fixed_file_path)
    print(f"!!!!! FILE PATH: {file_path}")
    if file_path is None:
        file_path = fixed_file_path

    # Ensure the path is valid and the file exists
    if file_path and os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return jsonify({"content": content})
        except Exception as e:
            print("!!!@!@!@!@!@!@!@!@")
            return jsonify({"error": str(e)}), 500
    else:
        print("!@@@@@@@@@@@@@@@@@@@@@@@@@")

        return jsonify({'error': 'Fixed file not found or path is invalid.'}), 400


if __name__ == '__main__':
    app.run(debug=True)

