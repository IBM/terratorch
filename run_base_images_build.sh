#!/bin/bash

PYTHON_VERSIONS=("3.11" "3.12" "3.13")

for v in ${PYTHON_VERSIONS[@]};
do
  echo "Creating base image for Python $v"
  podman build --build-arg version=$v -f Dockerfile.pytest -t terratorch_python_$v
done

