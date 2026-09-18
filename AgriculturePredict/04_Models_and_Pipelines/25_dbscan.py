# Session 25 - Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
# ==================================================================================
# Wrapper delegating to dbscan_clustering.py for AgriculturePredict

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
target_script = os.path.join(current_dir, "dbscan_clustering.py")

if __name__ == "__main__":
    with open(target_script, "r") as f:
        exec(f.read(), globals())
