#!/bin/bash
set -e
# ===========================================================================
# Recreate the `fold` conda env on the lab PC (Isaac Sim 6.0 + Newton + IsaacLab).
#
# WHY STAGED (not `conda env create -f environment.fold.yml`):
#   The exported yml pins torch==2.10.0+cu128 and the isaacsim 6.0 wheels but
#   records NO --extra-index-url, and it pins IsaacLab as LOCAL editable paths
#   that don't exist on the lab PC. A one-shot create therefore fails. We install
#   in the order the wheels actually resolve.
#
# PREREQS on lab PC (same lampa machine):
#   - miniconda/anaconda installed
#   - NVIDIA driver new enough for CUDA 12.8 (driver >= 570; lampa is 580 → OK)
#   - This repo cloned, and IsaacLab cloned as a SIBLING dir (see PORTING.md §3)
#   - Run inside tmux/screen — large downloads.
#
# Usage:
#   bash env/setup_fold.sh /ABS/PATH/TO/IsaacLab
# ===========================================================================

ISAACLAB_DIR="${1:?Pass the absolute path to the cloned IsaacLab repo as arg 1}"
ISAACLAB_COMMIT=a4a7602f29e755e2673fe0022ea35566df6dd7d5   # v3.0.0-beta (matches local fold env)
ENV_NAME=fold
PYVER=3.12

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "── 1. create env (python $PYVER) ───────────────────────────────────────"
conda create -y -n "$ENV_NAME" "python=$PYVER"

echo "── 2. torch 2.10.0 + cu128 (CUDA 12.8 wheels) ──────────────────────────"
conda run -n "$ENV_NAME" pip install \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

echo "── 3. Isaac Sim 6.0 meta + extscache bundles (NVIDIA index) ────────────"
# Mirror the folder6000 lesson: list extscache bundles EXPLICITLY (the [extscache]
# extra is unreliable). 6.0 uses the same meta+extscache pattern as 4.2.
conda run -n "$ENV_NAME" pip install \
    isaacsim==6.0.0.0 \
    isaacsim-extscache-physics==6.0.0.0 \
    isaacsim-extscache-kit==6.0.0.0 \
    isaacsim-extscache-kit-sdk==6.0.0.0 \
    --extra-index-url https://pypi.nvidia.com

echo "── 4. IsaacLab editable install (pinned commit) ────────────────────────"
git -C "$ISAACLAB_DIR" rev-parse HEAD | grep -q "$ISAACLAB_COMMIT" \
    || { echo "WARNING: IsaacLab is not at $ISAACLAB_COMMIT — run: git -C $ISAACLAB_DIR checkout $ISAACLAB_COMMIT"; }
# IsaacLab ships its own installer that pip -e's every source/* subpackage.
( cd "$ISAACLAB_DIR" && conda run -n "$ENV_NAME" ./isaaclab.sh --install ) \
    || { echo "isaaclab.sh failed; falling back to manual pip -e of each source/* pkg"; \
         for d in "$ISAACLAB_DIR"/source/*/; do \
             [ -f "$d/setup.py" -o -f "$d/pyproject.toml" ] && conda run -n "$ENV_NAME" pip install -e "$d"; \
         done; }

echo "── 5. VCS-pinned deps (rl_games, robomimic) ────────────────────────────"
conda run -n "$ENV_NAME" pip install \
    "rl_games @ git+https://github.com/isaac-sim/rl_games.git@6b3534f29568158e9e29ec8bf83cc88fce5f0cae" \
    "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9"

echo "── 6. everything else (pinned leftovers) ───────────────────────────────"
conda run -n "$ENV_NAME" pip install -r "$HERE/requirements.fold.rest.txt"

echo "── 7. verify ───────────────────────────────────────────────────────────"
conda run -n "$ENV_NAME" python -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())"
OMNI_KIT_ACCEPT_EULA=YES conda run -n "$ENV_NAME" python -c \
  "from isaacsim import SimulationApp; a=SimulationApp({'headless':True}); a.close(); print('ISAAC SIM 6.0 LAUNCH OK')"
conda run -n "$ENV_NAME" python -c "import isaaclab, isaaclab_newton; print('isaaclab', isaaclab.__version__, 'OK')"
echo "fold env ready."
