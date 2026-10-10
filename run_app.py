import os
import sys

# Ensure the root directory is on Python's module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

if __name__ == '__main__':
    print("\n=======================================================")
    print("  SEEDSENSE AI WEBSITE STARTING")
    print("  Access the app at: http://localhost:5000")
    print("=======================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
