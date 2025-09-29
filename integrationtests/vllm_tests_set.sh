#!/bin/bash

WORKING_DIR=$1

VENV_NAME=vllm_integration_tests_venv

cd ${WORKING_DIR}
pushd ${WORKING_DIR}

uv venv --python 3.12 ${VENV_NAME}
source ${VENV_NAME}/bin/activate

cd terratorch

uv pip install -e .[test]

# TODO: Remove once the Prithvi Sen1Flods11 data modules support the latest albumentations
uv pip install albumentations==1.4.6

pytest -s -v integrationtests/vLLM/ 2>&1

deactivate

popd
rm -r ${VENV_NAME}