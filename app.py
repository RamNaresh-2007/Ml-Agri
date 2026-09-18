#!/usr/bin/env python
"""
AgriYield AI - Root Workspace Launcher
Launch the clean agricultural dashboard from workspace root:
    python app.py
"""

import os
import sys
import importlib.util
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
PROJECT_DIR = WORKSPACE_ROOT / "AgriculturePredict"
LAUNCHER_FILE = PROJECT_DIR / "app.py"

if not LAUNCHER_FILE.exists():
    print(f"Error: Could not locate launcher file at {LAUNCHER_FILE}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("agri_launcher", str(LAUNCHER_FILE))
launcher_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher_mod)

if __name__ == "__main__":
    launcher_mod.main()
