#!/bin/bash

PYTHON_VERSIONS=("3.11" "3.12" "3.13")

for v in ${PYTHON_VERSIONS[@]};
do
  echo "Testing for Python $v"
  podman build --build-arg version=$v -f Dockerfile.run -t "pytest_terratorch_$v"
  podman run -t "pytest_terratorch_$v" bash run_install.sh 
done

