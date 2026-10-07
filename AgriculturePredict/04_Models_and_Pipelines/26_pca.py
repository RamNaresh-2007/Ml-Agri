# Session 26 - Principal Component Analysis (PCA)
# =========================================================================
# Wrapper delegating to pca_analysis.py for AgriculturePredict dimensionality
# reduction, scree analysis, factor loadings biplot, and variance decomposition.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
target_script = os.path.join(current_dir, "pca_analysis.py")

if __name__ == "__main__":
    with open(target_script, "r") as f:
        exec(f.read(), globals())
