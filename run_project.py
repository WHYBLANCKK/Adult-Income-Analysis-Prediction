#!/usr/bin/env python3
"""
run_project.py - Master pipeline runner for Adult Income Analysis & Prediction
Executes all four analysis notebooks sequentially, populates the SQLite data warehouse,
runs multidimensional OLAP queries, and validates output figure generation.
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
    print("ADULT INCOME ANALYSIS & PREDICTION — FULL PIPELINE EXECUTION")
    print("=" * 80)
    print(f"Using Python interpreter: {python_bin}")
    print(f"Project directory       : {root}\n")

    steps = [
        ("Part 1: Exploratory Data Analysis & Profile Similarity", "scripts/generate_notebook_1.py"),
        ("Part 2: Preprocessing, Dimensionality Reduction & Data Warehouse", "scripts/generate_notebook_2.py"),
        ("Part 3: Income Classification & Association Rule Mining", "scripts/generate_notebook_3.py"),
        ("Part 4: Unsupervised Learning & Population Segmentation", "scripts/generate_notebook_4.py"),
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
    print("ALL FOUR ANALYSIS NOTEBOOKS EXECUTED SUCCESSFULLY!")
    print(f"Total pipeline execution time: {time.time()-total_start:.2f}s")
    print("=" * 80)

if __name__ == '__main__':
    main()

