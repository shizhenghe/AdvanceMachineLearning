#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy scipy Pillow matplotlib scikit-learn python-docx
mkdir -p data
if [ ! -d data/magnetic-tiles/.git ]; then
 git clone --depth 1 https://github.com/abin24/Magnetic-tile-defect-datasets. data/magnetic-tiles
fi
