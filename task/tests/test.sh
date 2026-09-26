#!/usr/bin/env bash
set -uo pipefail

TESTS_DIR="${TESTS_DIR:-/tests}"
LOG_DIR="${LOG_DIR:-/logs/verifier}"
CTRF_PATH="${LOG_DIR}/ctrf.json"
PYTEST_LOG="${LOG_DIR}/pytest.log"
REWARD_PATH="${LOG_DIR}/reward.txt"

mkdir -p "${LOG_DIR}"

if python3 -m pytest --help 2>/dev/null | grep -q -- "--ctrf"; then
    python3 -m pytest --ctrf "${CTRF_PATH}" "${TESTS_DIR}/test_outputs.py" -rA \
        2>&1 | tee "${PYTEST_LOG}"
else
    python3 -m pytest "${TESTS_DIR}/test_outputs.py" -rA \
        2>&1 | tee "${PYTEST_LOG}"
fi
status=${PIPESTATUS[0]}

if [ "${status}" -eq 0 ]; then
    echo "1" > "${REWARD_PATH}"
else
    echo "0" > "${REWARD_PATH}"
fi

exit "${status}"
