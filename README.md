# SRT Slurm

Benchmarking toolkit for Dynamo and SGLang on SLURM.

## Run a benchmark

1. **Run `make setup`** - Downloads dynamo dependencies (nats, etcd, wheels) and interactively creates `srtslurm.yaml` with your cluster settings:

```bash
make setup
```

You'll be prompted for:

- SLURM account (e.g., `restricted`)
- SLURM partition (e.g., `batch-aqua`)
- Network interface (e.g., `enP6p9s0np0`)
- Default time limit (e.g., `4:00:00`)

This creates `srtslurm.yaml` automatically. You can also manually copy from the example:

```bash
cp srtslurm.yaml.example srtslurm.yaml
```

3. Run your first benchmark (much shorter now!):

```bash
# Minimal command (cluster settings + container from srtslurm.yaml):
cd slurm_runner
python3 submit_job_script.py \
  --model-dir /mnt/lustre01/models/deepseek-r1-0528-fp4-v2 \
  --gpu-type gb200-fp4 \
  --gpus-per-node 4 \
  --prefill-nodes 1 \
  --decode-nodes 12 \
  --prefill-workers 1 \
  --decode-workers 1 \
  --script-variant max-tpt \
  --benchmark "type=sa-bench; isl=1024; osl=1024; concurrencies=1x8x32x128x512x1024x2048x4096x8192; req-rate=inf"

# Or override anything via CLI:
python3 submit_job_script.py \
  --container-image /path/to/different/container.sqsh \
  --account nvidia \
  --partition batch \
  --network-interface enP6p9s0np0 \
  ... # (rest same as above)
```

**What's simplified (9+ args removed!):**

- ✅ `--config-dir` optional (defaults to `../configs`)
- ✅ `--log-dir` optional (defaults to `../logs`)
- ✅ `--container-image` optional (read from `srtslurm.yaml`)
- ✅ `--account`, `--partition`, `--network-interface` optional (from `srtslurm.yaml`)
- ✅ `--time-limit` optional (from `srtslurm.yaml` or `04:00:00`)
- ✅ `--use-dynamo-whls` removed (always enabled)

For more info on the submission script see [slurm_runner/README.md](slurm_runner/README.md)

## Run the UI

```bash
./run_dashboard.sh
```

The dashboard will open at http://localhost:8501 and scan the current directory for benchmark runs. You can specify your own log directory in the UI itself.

## Cloud Storage Sync

Store benchmark results in cloud storage (S3-compatible) and access them from anywhere.

### Setup

1. **Install dependencies:**

```bash
pip install boto3
```

(Note: PyYAML is in stdlib on most clusters)

2. **Edit `srtslurm.yaml`** to add cloud storage settings:

```yaml
cloud:
  endpoint_url: "https://your-s3-endpoint"
  bucket: "your-bucket-name"
  prefix: "benchmark-results/"
```

3. **Set credentials as environment variables:**

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### Usage

**On Clusters (Push results):**

```bash
# Push all runs (defaults to logs/ directory, skips existing)
./push_after_benchmark.sh

# Or specify a different logs directory
./push_after_benchmark.sh --log-dir /mnt/lustre01/users-public/slurm-shared/joblogs

# Or push a single run
./push_after_benchmark.sh 3667_1P_1D_20251110_192145
```

The script automatically skips runs that already exist in cloud storage.

**Locally (Pull results):**

Just launch the dashboard - it automatically pulls missing runs on startup:

```bash
./run_dashboard.sh
```

In the dashboard sidebar:

- Auto-sync is enabled by default (pulls missing runs)
- Use 🔄 button to manually sync anytime
- See status messages (new runs downloaded, errors, etc.)

Or pull manually:

```bash
uv run python slurm_runner/scripts/sync_results.py pull-missing
uv run python slurm_runner/scripts/sync_results.py list-remote
```

## What It Does

**Pareto Analysis** - Compare throughput efficiency (TPS/GPU) vs per-user throughput (TPS/User) across configurations

**Latency Breakdown** - Visualize TTFT, TPOT, and ITL metrics as concurrency increases

**Config Comparison** - View deployment settings (TP/DP) and hardware specs side-by-side

**Data Export** - Sort, filter, and export metrics to CSV

## Key Metrics

- **Output TPS/GPU** - Throughput per GPU (higher = more efficient)
- **Output TPS/User** - Throughput per concurrent user (higher = better responsiveness)
- **TTFT** - Time to first token (lower = faster start)
- **TPOT** - Time per output token (lower = faster generation)
- **ITL** - Inter-token latency (lower = smoother streaming)

## Directory Structure

This structure comes built into the scripts. WIP to handle other directory structures.

The app expects benchmark runs in subdirectories with:

- `{jobid}.json` - Metadata file with run configuration (required)
- `vllm_isl_*_osl_*/` containing `*.json` result files
- `*_config.json` files for node configurations
- `*_prefill_*.err` and `*_decode_*.err` files for node metrics
