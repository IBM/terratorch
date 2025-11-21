#!/bin/bash
set -euo pipefail

TEST_DIR="/dccstor/terratorch/shared/integrationtests/terratorch"
COV_DIR="$TEST_DIR/.coverage_jobs"

echo "Merging coverage data..."
cd "$TEST_DIR"

coverage combine "$COV_DIR"/.coverage_*
coverage report -m
coverage html -d "$TEST_DIR/htmlcov"

echo "Coverage report generated at $TEST_DIR/htmlcov/index.html"
