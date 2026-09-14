#!/bin/bash
#SBATCH --job-name=vr-demo-smoke
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node09
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=outputs/smoke/slurm-%j.out
# GPU smoke test for the web demo, run from the repository root:
#     mkdir -p outputs/smoke && sbatch demo/tests/smoke_job.sbatch
# GPUs come from --gres only; never set CUDA_VISIBLE_DEVICES here.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

export HOME_CACHE=$HOME/.cache
export UV_CACHE_DIR=$HOME_CACHE/uv
export HF_HOME=$HOME_CACHE/huggingface
export TMPDIR=$HOME/tmp
export GRADIO_TEMP_DIR=$TMPDIR/gradio
export TORCHINDUCTOR_CACHE_DIR=$HOME_CACHE/torch/inductor
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$TMPDIR" "$GRADIO_TEMP_DIR" "$TORCHINDUCTOR_CACHE_DIR" outputs/smoke
echo "node=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} repo=$PWD"
df -h /tmp | tail -1

# uv.lock pins torch 2.10.0, which matches the cluster's CUDA 12.8 driver.
uv sync --extra demo --extra dev
PY=.venv/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$PY -m playwright install chromium

echo "=== 0. unit tests ==="
$PY -m pytest -q
node --test demo/tests/test_score_model.mjs

echo "=== 1. model-level smoke (load, generate, switch checkpoint, generate) ==="
$PY demo/tests/gpu_smoke.py outputs/smoke

echo "=== 2. browser end-to-end against a real server ==="
GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7871 $PY demo/app.py > outputs/smoke/server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7871/ | grep -q 200 && break
  sleep 5
done
$PY demo/tests/browser/generate_smoke.py

echo "=== 3. Space bundle dry-run ==="
$PY scripts/sync_space.py --dry-run | head -3
echo "=== SMOKE PASSED ==="
