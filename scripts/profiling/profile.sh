#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Torch profiling script for sglang.launch_server
# This script runs bench_one_batch_server with profiling enabled

model_name="deepseek-ai/DeepSeek-R1"
head_node="127.0.0.1"
head_port=30000

# Parse arguments (same as sa-bench for consistency)
n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
total_gpus=$5

echo "Torch Profiling Configuration:"
echo "  Profiling mode: ${PROFILING_MODE}"
echo "  Profiling dir: ${SGLANG_TORCH_PROFILER_DIR}"
echo "  Prefill workers: ${n_prefill}"
echo "  Decode workers: ${n_decode}"
echo "  Prefill GPUs: ${prefill_gpus}"
echo "  Decode GPUs: ${decode_gpus}"
echo "  Total GPUs: ${total_gpus}"

# Wait for server to be ready using inline wait function
echo "Waiting for server at http://${head_node}:${head_port} to be ready..."
wait_until_ready() {
    local SERVER_URL="$1"
    while true; do
        status_code=$(curl -s -o /dev/null -w "%{http_code}" "${SERVER_URL}/health" || echo "000")
        if [ "$status_code" -eq 200 ]; then
            echo "Server ${SERVER_URL} is ready"
            break
        fi
        echo "Server not ready yet (status: ${status_code}), waiting..."
        top -b -n 1 | head -10
        PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | tr -d ' ' | head -n1)
        [ -n "$PID" ] && py-spy dump -s --pid $PID > /logs/py-spy-dump-${SLURM_NODEID:-0}.txt || echo "No GPU process found"
        sleep 30
    done
}
wait_until_ready "http://${head_node}:${head_port}"

# Determine profiling parameters strictly from environment (no internal defaults)
PROFILE_STEPS_ARG=""
CLI_ARGS=""
[[ -n "${PROFILE_CONCURRENCY}" ]] && CLI_ARGS+=" --batch-size ${PROFILE_CONCURRENCY}"
[[ -n "${PROFILE_ISL}" ]] && CLI_ARGS+=" --input-len ${PROFILE_ISL}"
[[ -n "${PROFILE_OSL}" ]] && CLI_ARGS+=" --output-len ${PROFILE_OSL}"

[[ -n "${PROFILE_STOP_STEP}" ]] && PROFILE_STEPS_ARG+=" --profile-steps $((PROFILE_STOP_STEP-0))"
#[[ -n "${PROFILE_START_STEP}" ]] && PROFILE_STEPS_ARG+=" --profile-start ${PROFILE_START_STEP}" # start step is not supported yet

echo "Running ${PROFILING_MODE} profiling with${CLI_ARGS} ${PROFILE_STEPS_ARG}"

# Create profiling output directory
mkdir -p ${SGLANG_TORCH_PROFILER_DIR} 2>/dev/null || true

echo "Running torch profiler..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

set -x
python3 -m sglang.bench_one_batch_server \
    --model ${model_name} \
    --base-url http://${head_node}:${head_port} \
    ${CLI_ARGS} \
    ${PROFILE_STEPS_ARG} \
    --profile
exit_code=$?
set +x

echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "Torch profiling completed for ${PROFILING_MODE} mode with exit code ${exit_code}"
echo "Profiling results saved to ${SGLANG_TORCH_PROFILER_DIR}"

exit ${exit_code}
