#!/usr/bin/env bash

# Shared destination for evaluation artifacts. Defaults to results/eval inside
# the repository; set EVAL_RESULTS_ROOT to point every launcher at shared
# storage instead.
_EVAL_PATHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-${_EVAL_PATHS_ROOT}/results/eval}"
export EVAL_RESULTS_ROOT
mkdir -p "${EVAL_RESULTS_ROOT}"
