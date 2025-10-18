#!/usr/bin/env python3
"""
Startup script for the Ad Headline Optimizer.
This ensures proper Python path setup before launching the Streamlit app.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ['PYTHONPATH'] = str(project_root)

if __name__ == "__main__":
    import subprocess
    
    print("🚀 Starting Ad Headline Optimizer...")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[0]}")
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/simple_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Ad Headline Optimizer...")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Try running: streamlit run app/simple_app.py")
