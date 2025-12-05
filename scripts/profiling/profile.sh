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

# Determine profiling parameters strictly from environment 
PROFILE_STEPS_ARG=""
CLI_ARGS=""
[[ -n "${PROFILE_CONCURRENCY}" ]] && CLI_ARGS+=" --batch-size ${PROFILE_CONCURRENCY}"
# Require ISL/OSL to be provided; do not pass them as CLI args here
if [[ -z "${PROFILE_ISL}" || -z "${PROFILE_OSL}" ]]; then
    echo "Error: isl and osl must be set for profiling."
    exit 1
fi

# Configure profiling steps range; set defaults independently if missing
if [[ -z "${PROFILE_START_STEP}" ]]; then
    echo "Warning: PROFILE_START_STEP not set; defaulting to 0"
    PROFILE_START_STEP=0
fi
if [[ -z "${PROFILE_STOP_STEP}" ]]; then
    echo "Warning: PROFILE_STOP_STEP not set; defaulting to 50"
    PROFILE_STOP_STEP=50
fi


echo "Running profiler..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# Create profiling output directory only when torch profiler dir is provided
ACTIVITIES=""
if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
    ACTIVITIES='["CPU", "GPU"]'
    mkdir -p "${SGLANG_TORCH_PROFILER_DIR}" 2>/dev/null || true
    export SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR}
else
    ACTIVITIES='["CUDA_PROFILER"]'
    mkdir -p "/logs/profiles" 2>/dev/null || true
fi

set -x

curl -X POST http://${head_node}:${head_port}/start_profile -H "Content-Type: application/json" -d "{\"start_step\": \"$PROFILE_START_STEP\", \"num_steps\": $((PROFILE_STOP_STEP-PROFILE_START_STEP)), \"activities\": $ACTIVITIES}"

python3 -m sglang.bench_serving \
--backend sglang \
--model ${model_name} \
--host ${head_node} --port ${head_port} \
--dataset-name random \
--max-concurrency $PROFILE_CONCURRENCY \
--num-prompts 128 \
--random-input-len $PROFILE_ISL \
--random-output-len $PROFILE_OSL \
--random-range-ratio 1 \
--warmup-request 10

pip install lm-eval tenacity
python -m lm_eval \
--model local-completions \
--tasks gsm8k \
--model_args \
base_url=http://${head_node}:${head_port}/v1/completions,\
model=${model_name},\
tokenized_requests=False,tokenizer_backend=None,\
num_concurrent=${PROFILE_CONCURRENCY},timeout=6000,max_retries=1 \
--limit 10

exit_code=$?
set +x

echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "Torch profiling completed for ${PROFILING_MODE} mode with exit code ${exit_code}"
echo "Profiling results saved to ${SGLANG_TORCH_PROFILER_DIR}"

exit ${exit_code}
