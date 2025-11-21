#!/bin/bash
# run_tests_lsf.sh
# Run each test (either file or individual test function) in separate LSF jobs with coverage + logs.

set -euo pipefail

TEST_DIR="/dccstor/terratorch/shared/integrationtests/terratorch"
LOG_DIR="$TEST_DIR/logs"
COV_DIR="$TEST_DIR/.coverage_jobs"
mkdir -p "$LOG_DIR" "$COV_DIR"

# 1. Get all standalone test files (tests/test_*)
test_files=$(find tests -maxdepth 1 -type f -name 'test_*')

# 2. Include individual tests inside integrationtests/test_base_set.py
extra_tests=$(cd "$TEST_DIR" && \
  pytest --collect-only -q integrationtests/test_base_set.py 2>/dev/null | \
  grep -E '^integrationtests/test_base_set\.py::' || true)

# Combine both sets
all_tests=$(printf "%s\n%s\n" "$test_files" "$extra_tests")

# 3. Submit each test as a separate LSF job
for test in $all_tests; do
    # Normalize name (pytest nodeid may include "::class::test_func")
    test_name=$(echo "$test" | tr '/:' '_')
    out="$LOG_DIR/${test_name}.out"
    err="$LOG_DIR/${test_name}.err"
    cov_file="$COV_DIR/.coverage_${test_name}"

    echo "Submitting job for $test"

    bsub -gpu num=1 -R "rusage[ngpus=1,cpu=4,mem=32GB]" \
         -J "terratorch_${test_name}" \
         -oo "$out" -eo "$err" \
         "cd $TEST_DIR && \
          source .venv2/bin/activate && \
          coverage run --parallel-mode --data-file=$cov_file -m pytest -s -v $test"
done

echo "All jobs submitted. Monitor with: bjobs -u \$USER"
