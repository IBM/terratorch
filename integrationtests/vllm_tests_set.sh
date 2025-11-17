#!/bin/bash
set -x

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
WORKING_DIR=${SCRIPT_DIR}/../../

VENV_NAME=vllm_integration_tests_venv

cd ${WORKING_DIR}

uv venv --clear --python 3.12 ${VENV_NAME}
source ${VENV_NAME}/bin/activate

cd terratorch

uv pip install --no-cache-dir -e .[test,vllm,vllm_test]

pytest -s -v integrationtests/vLLM/ 2>&1
pytest_ret=$?

deactivate

cd ${WORKING_DIR}
rm -r ${VENV_NAME}

# Return the ret code of the pytest run
exit ${pytest_ret}