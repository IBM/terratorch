#!/bin/bash
set -x

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
WORKING_DIR=${SCRIPT_DIR}/../../

VENV_NAME=vllm_integration_tests_venv

uv venv --clear --python 3.12 ${VENV_NAME}
source ${VENV_NAME}/bin/activate
cd ${WORKING_DIR}/terratorch

uv pip install -e .[test]

pytest -s -v integrationtests/vLLM/ 2>&1

deactivate

cd ${WORKING_DIR}
rm -r ${VENV_NAME}