
from pathlib import Path
import json

NUMBER_OF_STUDENTS = 31

submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)

# Create a valid empty Jupyter notebook
notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

for i in range(1, NUMBER_OF_STUDENTS + 1):
    student_dir = submissions_dir / f"Student_{i:02d}"
    student_dir.mkdir(parents=True, exist_ok=True)

    notebook_path = student_dir / "TP1_Solutions.ipynb"

    if not notebook_path.exists():
        notebook_path.write_text(
            json.dumps(notebook, indent=2),
            encoding="utf-8"
        )

print(f"{NUMBER_OF_STUDENTS} student folders created.")



