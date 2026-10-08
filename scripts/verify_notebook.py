"""Execute the notebook in this interpreter; no additional notebook runner required.

This smoke check evaluates every code cell and captures text and rich displays.
It is deliberately not a general-purpose Jupyter execution engine.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
path = ROOT / "notebooks" / "retail_analysis.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
namespace = {"__name__": "__main__"}
execution = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    execution += 1
    output = io.StringIO()
    displays = []

    def capture_display(*objects, **kwargs):
        for obj in objects:
            bundle = {"text/plain": str(obj)}
            if hasattr(obj, "to_plotly_json"):
                bundle["application/vnd.plotly.v1+json"] = json.loads(obj.to_json())
            elif hasattr(obj, "_repr_html_"):
                rendered = obj._repr_html_()
                if rendered:
                    bundle["text/html"] = rendered
            displays.append(nbformat.v4.new_output("display_data", data=bundle, metadata={}))

    with contextlib.redirect_stdout(output), patch("IPython.display.display", capture_display):
        exec(compile(cell.source, f"notebook:cell:{cell.id}", "exec"), namespace)
        namespace["display"] = capture_display
    cell.execution_count = execution
    cell.outputs = ([nbformat.v4.new_output("stream", name="stdout", text=output.getvalue())] if output.getvalue() else []) + displays
    print(f"Cell {execution} ({cell.id}): passed", flush=True)
notebook.metadata.language_info.version = sys.version.split()[0]
nbformat.write(notebook, path)
print("Notebook validated and all code cells executed successfully.")
