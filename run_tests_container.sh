#!/bin/bash

PYTHON_VERSIONS=("3.11" "3.12" "3.13")

for v in ${PYTHON_VERSIONS[@]};
do
  echo "Testing for Python $v"
  podman run -t "pytest_terratorch_$v"
done

