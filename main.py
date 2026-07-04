"""
Root-level fallback entry point for platforms (like a manually configured
Render service) that run `python main.py` from the repo root instead of
using render.yaml's rootDir. Delegates straight into python-app/main.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python-app"))
os.chdir(os.path.join(os.path.dirname(__file__), "python-app"))

import main  # noqa: F401,E402
