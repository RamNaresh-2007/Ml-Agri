#!/usr/bin/env python
"""
AgriYield AI - Main Application Launcher
Runs the interactive web dashboard with unified REST API endpoints,
and automatically opens the dashboard in your default browser.

Usage:
    python app.py                     # Launch clean web dashboard (http://127.0.0.1:5000)
    python app.py --streamlit         # Launch Streamlit Plotly intelligence suite
    python app.py --port 5001         # Run on custom port
    python app.py --no-browser        # Start server without auto-opening browser
"""

import os
import sys
import argparse
import threading
import webbrowser
import time
import importlib.util
from pathlib import Path

# Safe UTF-8 console output for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
WEB_APP_DIR = CURRENT_DIR / "05_Web_Application"
WORKSPACE_DIR = CURRENT_DIR.parent

for p in [str(WEB_APP_DIR), str(CURRENT_DIR), str(WORKSPACE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def get_flask_app():
    """Dynamically and screen-tested load the Flask app from 05_Web_Application/app.py."""
    app_file = WEB_APP_DIR / "app.py"
    spec = importlib.util.spec_from_file_location("agri_web_app", str(app_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def open_browser_tab(url, delay=1.25):
    """Open user's default browser after server initializes."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def run_streamlit():
    """Launch Streamlit intelligence suite."""
    dashboard_path = WEB_APP_DIR / "dashboard.py"
    print("\n" + "=" * 62)
    print("  🌾 Launching AgriYield AI Streamlit Suite...")
    print("=" * 62 + "\n", flush=True)
    os.system(f'streamlit run "{dashboard_path}"')


def run_dashboard(port=5000, open_browser=True, debug=False):
    """Launch Flask web dashboard and REST API."""
    flask_app = get_flask_app()
    url = f"http://127.0.0.1:{port}"

    print("\n" + "=" * 62)
    print("  🌾 AgriYield AI - Precision Agriculture Dashboard")
    print("=" * 62)
    print(f"  🚀 Web Dashboard : {url}")
    print(f"  📊 REST API      : {url}/api/data")
    print(f"  🔮 AI Predictor  : {url}/api/predict")
    print(f"  🎨 UI Theme      : Simple & Clean (Emerald / Frost Light)")
    print("=" * 62)
    print("  Press Ctrl+C to stop the application server.\n", flush=True)

    if open_browser:
        threading.Thread(target=open_browser_tab, args=(url,), daemon=True).start()

    flask_app.run(host="127.0.0.1", port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser(description="AgriYield AI - Application Launcher")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind dashboard server (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    parser.add_argument("--streamlit", "-s", action="store_true", help="Launch Streamlit intelligence dashboard")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable Flask debug mode")

    args = parser.parse_args()

    if args.streamlit:
        run_streamlit()
    else:
        run_dashboard(port=args.port, open_browser=not args.no_browser, debug=args.debug)


if __name__ == "__main__":
    main()
