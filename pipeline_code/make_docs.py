import inspect
import importlib
import sys

module_names = [
    "pipeline_code.file_formatting",
    "pipeline_code.kernels",
    "pipeline_code.modeling",
    "pipeline_code.correction",
]

with open("API_Documentation.md", "w") as f:
    f.write("# API Documentation\n\n")

    for mod_name in module_names:
        try:
            module = importlib.import_module(mod_name)
        except Exception as e:
            print(f"Skipping {mod_name}: {e}")
            continue

        f.write(f"## {mod_name.split('.')[-1]}\n\n")

        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):
                doc = inspect.getdoc(obj)
                if doc:
                    f.write(f"### {name}\n\n")
                    f.write(doc + "\n\n---\n\n")