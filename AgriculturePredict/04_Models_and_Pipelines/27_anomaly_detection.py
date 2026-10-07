# Session 27 - Agricultural Anomaly Detection & Diagnostics
# ==============================================================================
# Wrapper delegating to anomaly_detection.py for AgriculturePredict
# multi-algorithm anomaly detection (Isolation Forest, LOF, Elliptic Envelope, OC-SVM).

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
target_script = os.path.join(current_dir, "anomaly_detection.py")

if __name__ == "__main__":
    with open(target_script, "r") as f:
        exec(f.read(), globals())
