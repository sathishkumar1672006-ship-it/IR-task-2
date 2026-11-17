"""
Executes the notebook and saves all real outputs inside the .ipynb file
"""

import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

print("Regenerating notebook structure...")
os.system("python create_notebook.py")

print("Executing notebook cells with ExecutePreprocessor...")
notebook_path = "notebooks/News_Classification_Battle.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "notebooks/"}})

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully executed and saved notebook to: {notebook_path}")

