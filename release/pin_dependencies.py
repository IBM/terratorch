import toml
import sys
import re
import subprocess


def get_pip_list():
    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    return result.stdout


def parse_pip_list(pip_list):
    package_versions = {}
    for line in pip_list.strip().split("\n")[2:]:  # Skip headers
        match = re.match(r"(\S+)\s+(\S+)", line)
        if match:
            package_versions[match.group(1).lower()] = match.group(2)
    return package_versions


def update_pyproject_toml(pyproject_path, package_versions):
    with open(pyproject_path, "r") as f:
        lines = f.readlines()

    in_dependencies = False
    updated_lines = []
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("dependencies = ["):
            in_dependencies = True
            updated_lines.append(line)
            continue

        if in_dependencies:
            if stripped == "]":
                in_dependencies = False
                updated_lines.append(line)
                continue

            match = re.match(r'\s*"([^"]+)"(.*)', stripped)
            if match:
                package_name = match.group(1)
                constraints = match.group(2)
                lower_package = package_name.lower()
                if lower_package in package_versions and "==" not in constraints and "<=" not in constraints and ">=" not in constraints and "@" not in constraints:
                    updated_line = f'  "{package_name}=={package_versions[lower_package]}",\n'
                    print(f"Updated {package_name} to version {package_versions[lower_package]}")
                else:
                    updated_line = line
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    with open(pyproject_path, "w") as f:
        f.writelines(updated_lines)
    print("pyproject.toml dependencies section has been updated.")


if __name__ == "__main__":
    pyproject_file = "pyproject.toml"
    pip_list_content = get_pip_list()
    package_versions = parse_pip_list(pip_list_content)
    update_pyproject_toml(pyproject_file, package_versions)
