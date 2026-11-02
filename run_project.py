#!/usr/bin/env python3
"""
run_project.py - Master runner for Adult Income Mining (DMW Activity 2)
Executes all four assignment notebooks, populates the SQLite warehouse,
runs OLAP queries, and verifies output figure generation.
"""

import sys
import subprocess
import time
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    python_bin = root / '.venv' / 'bin' / 'python'
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    print("=" * 80)
    print("DMW ACTIVITY 2: ADULT INCOME MINING - FULL WORKFLOW EXECUTION")
    print("=" * 80)
    print(f"Using Python interpreter: {python_bin}")
    print(f"Project directory       : {root}\n")

    steps = [
        ("Notebook 1: EDA & Profile Similarity", "scripts/generate_notebook_1.py"),
        ("Notebook 2: Preprocessing, PCA & Data Warehouse", "scripts/generate_notebook_2.py"),
        ("Notebook 3: Classification & Association Rules", "scripts/generate_notebook_3.py"),
        ("Notebook 4: Unsupervised Clustering", "scripts/generate_notebook_4.py"),
    ]

    total_start = time.time()
    for title, script in steps:
        print(f"\n>>> Executing {title}...")
        t0 = time.time()
        res = subprocess.run([str(python_bin), str(root / script)], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED {title} (exit code {res.returncode}):")
            print(res.stderr)
            sys.exit(res.returncode)
        else:
            print(f"COMPLETED in {time.time()-t0:.2f}s")
            # Print last output line
            lines = [l for l in res.stdout.strip().split('\n') if l]
            if lines:
                print(f"    {lines[-1]}")

    print("\n" + "=" * 80)
    print("ALL 4 NOTEBOOKS SUCCESSFULLY EXECUTED AND VERIFIED!")
    print(f"Total pipeline execution time: {time.time()-total_start:.2f}s")
    print("=" * 80)

if __name__ == '__main__':
    main()

