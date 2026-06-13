#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# The submitted GCNII solution uses JittorGeometric custom CUDA ops by default.
export USE_CUDA="${USE_CUDA:-1}"

if [ -f "$REPO_ROOT/scripts/jittor_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/jittor_env.sh"
elif [ -f "/workspace/JDJT/scripts/jittor_env.sh" ]; then
  # shellcheck disable=SC1091
  source "/workspace/JDJT/scripts/jittor_env.sh"
fi

if [ "${USE_CUDA:-0}" = "1" ]; then
  for CUDA_ROOT in \
    "$HOME/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux" \
    "/usr/local/cuda"; do
    if [ -f "$CUDA_ROOT/include/cuda.h" ]; then
      export CPATH="$CUDA_ROOT/include:${CPATH:-}"
      export CPLUS_INCLUDE_PATH="$CUDA_ROOT/include:${CPLUS_INCLUDE_PATH:-}"
      export PATH="$CUDA_ROOT/bin:$PATH"
      export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi

cd "$SCRIPT_DIR"
python gcn.py "$@"
