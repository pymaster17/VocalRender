#!/bin/bash
#SBATCH --job-name=vr-demo-smoke
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node09
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/smoke/slurm-%j.out
# GPU smoke test for the VocalRender demo. GPUs come from --gres only; never set CUDA_VISIBLE_DEVICES here.
set -euo pipefail
cd /export/home2/n2505683d/VocalRender-demo

export HOME_CACHE=/export/home2/n2505683d/.cache
export UV_CACHE_DIR=$HOME_CACHE/uv
export HF_HOME=$HOME_CACHE/huggingface
export TMPDIR=/export/home2/n2505683d/tmp
export GRADIO_TEMP_DIR=$TMPDIR/gradio
export TORCHINDUCTOR_CACHE_DIR=$HOME_CACHE/torch/inductor
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$TMPDIR" "$GRADIO_TEMP_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "node=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
df -h /tmp | tail -1

VENV=.venv-gpu
if [ ! -x $VENV/bin/python ]; then
  uv venv --python 3.11 $VENV
fi
# Torch wheels must match the node's driver (CUDA 12.8 on this cluster); pin before the rest resolves.
uv pip install --python $VENV/bin/python torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $VENV/bin/python -r requirements.txt playwright
PY=$VENV/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== 1. model-level smoke (load, generate, switch checkpoint, generate) ==="
$PY tests/gpu_smoke.py logs/smoke

echo "=== 2. browser end-to-end against a real server ==="
GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7871 $PY app.py > logs/smoke/server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7871/ | grep -q 200 && break
  sleep 5
done
$PY tests/browser/generate_smoke.py
echo "=== SMOKE PASSED ==="
