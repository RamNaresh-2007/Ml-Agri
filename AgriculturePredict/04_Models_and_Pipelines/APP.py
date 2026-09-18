# AgriYield Preprocessing Pipeline (APP)
# Alias delegating directly to PPP.py

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
target_script = os.path.join(current_dir, "PPP.py")

if __name__ == "__main__":
    with open(target_script, "r") as f:
        exec(f.read(), globals())
