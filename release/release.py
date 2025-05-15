import os
import subprocess
import tomllib

def get_version():
    return '1.0.1'

def pin_requirements():
    subprocess.run(["python", "./release/pin_dependencies.py"], check=True)

def build_package():
    subprocess.run(["python", "-m", "build"], check=True)

def upload_to_pypi():
    subprocess.run(["twine", "upload", "dist/*"], check=True)

def tag_and_push(version):
    subprocess.run(["git", "tag", version], check=True)
    subprocess.run(["git", "push", "origin", version], check=True)

def main():
    pin_requirements()
    version = get_version()
    print(f"Releasing version {version}...")
    build_package()
    upload_to_pypi()
    tag_and_push(version)
    print("Release completed.")

if __name__ == "__main__":
    main()
