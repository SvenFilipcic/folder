#!/bin/bash
set -e
# ===========================================================================
# Recreate the `infer` conda env on the lab PC (spconv / UV Mapper inference).
#
# Different CUDA toolkit than `fold`: this env is built on CUDA 12.1 wheels
# (torch 2.5.1+cu121, spconv-cu121, cumm-cu121). The lampa driver (580) serves
# both cu121 and cu128 — they coexist fine in separate envs.
#
# spconv lesson (folder6000 progress2): the spconv wheel MUST match the CUDA
# series. We pin -cu121 to match the local env. If the lab PC ever needs a
# different CUDA, swap cu121 → the matching series in BOTH torch and spconv/cumm.
#
# Usage:  bash env/setup_infer.sh
# ===========================================================================

ENV_NAME=infer
PYVER=3.11
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "── 1. create env (python $PYVER) ───────────────────────────────────────"
conda create -y -n "$ENV_NAME" "python=$PYVER"

echo "── 2. torch 2.5.1 + cu121 ──────────────────────────────────────────────"
conda run -n "$ENV_NAME" pip install \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "── 3. spconv + cumm (cu121) ────────────────────────────────────────────"
conda run -n "$ENV_NAME" pip install spconv-cu121==2.3.8 cumm-cu121==0.7.11

echo "── 4. everything else (pinned leftovers) ───────────────────────────────"
conda run -n "$ENV_NAME" pip install -r "$HERE/requirements.infer.rest.txt"

echo "── 5. verify ───────────────────────────────────────────────────────────"
conda run -n "$ENV_NAME" python -c \
  "import torch, spconv, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| spconv ok | numpy', numpy.__version__)"
echo "infer env ready."
