import re
import subprocess
import toml

pyproject_file = "pyproject.toml"

pip_output = subprocess.run(["pip", "list", "--format=freeze"], capture_output=True, text=True).stdout

pip_versions = {}
for line in pip_output.splitlines():
    parts = line.split("==")
    if len(parts) == 2:
        package, version = parts
        pip_versions[package.lower()] = version

with open(pyproject_file, "r") as f:
    pyproject_raw = f.read()

match = re.search(r"dependencies\s*=\s*\[(.*?)\]", pyproject_raw, re.DOTALL)
if match:
    deps_raw = match.group(1)
    deps = [d.strip().strip('"') for d in deps_raw.split(",")]

    updated_deps = []
    for dep in deps:
        match = re.match(r"([a-zA-Z0-9_-]+)([<>=!].*)?", dep)
        if match:
            pkg_name = match.group(1).lower()
            if pkg_name in pip_versions:
                updated_deps.append(f'"{pkg_name}=={pip_versions[pkg_name]}"')
            else:
                updated_deps.append(f'"{dep}"')  # Keep as is if not found in pip list

    updated_deps_str = ",\n  ".join(updated_deps)
    pyproject_raw = re.sub(r"dependencies\s*=\s*\[.*?\]", f"dependencies = [\n  {updated_deps_str}\n]", pyproject_raw, flags=re.DOTALL)

with open(pyproject_file, "w") as f:
    f.write(pyproject_raw)
