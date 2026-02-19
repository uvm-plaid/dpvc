#!/usr/bin/env bash
# scripts/setup.sh — One-shot setup for the ControlVC pipeline.
#
# Usage (run from repo root):
#   bash scripts/setup.sh [--control-vc-dir PATH] [--venv-dir PATH]
#
# Defaults:
#   --control-vc-dir  ~/repos/control-vc
#   --venv-dir        .venv310 (inside this repo)

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BOLD}[setup]${NC} $*"; }
success() { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup] WARNING:${NC} $*"; }
die()     { echo -e "${RED}[setup] ERROR:${NC} $*" >&2; exit 1; }

# ── Argument parsing ──────────────────────────────────────────────────────────
CONTROL_VC_DIR="${CONTROL_VC_DIR:-$HOME/repos/control-vc}"
VENV_DIR=".venv310"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --control-vc-dir) CONTROL_VC_DIR="$2"; shift 2 ;;
    --venv-dir)       VENV_DIR="$2";        shift 2 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# Resolve to absolute path
CONTROL_VC_DIR="${CONTROL_VC_DIR/#\~/$HOME}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ""
info "ControlVC setup"
info "  control-vc dir : $CONTROL_VC_DIR"
info "  venv dir       : $REPO_ROOT/$VENV_DIR"
echo ""

# ── Step 1: Detect Python 3.10 ───────────────────────────────────────────────
info "Step 1/8: Detecting Python 3.10..."

PY310=""
for candidate in \
    python3.10 \
    "$HOME/.pyenv/versions/3.10.0/bin/python3.10" \
    "$HOME/.pyenv/shims/python3.10"; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c 'import sys; print(sys.version_info[:2])')
    if [[ "$ver" == "(3, 10)" ]]; then
      PY310="$candidate"
      break
    fi
  fi
done

# Try any pyenv 3.10.x
if [[ -z "$PY310" ]]; then
  for pybin in "$HOME"/.pyenv/versions/3.10.*/bin/python3.10; do
    if [[ -x "$pybin" ]]; then
      PY310="$pybin"
      break
    fi
  done
fi

if [[ -z "$PY310" ]]; then
  die "Python 3.10 not found. Install it with pyenv:\n  pyenv install 3.10.0\n  pyenv global 3.10.0"
fi

success "Found Python 3.10: $PY310"

# ── Step 2: Clone control-vc ─────────────────────────────────────────────────
info "Step 2/8: Checking control-vc repository..."

if [[ -d "$CONTROL_VC_DIR" ]]; then
  success "control-vc already exists at $CONTROL_VC_DIR — skipping clone."
else
  info "Cloning zuruoke/control-vc to $CONTROL_VC_DIR..."
  git clone https://github.com/zuruoke/control-vc.git "$CONTROL_VC_DIR"
  success "Cloned control-vc."
fi

# ── Step 3: Create venv ───────────────────────────────────────────────────────
info "Step 3/8: Setting up Python virtual environment..."

VENV_PATH="$REPO_ROOT/$VENV_DIR"
if [[ -d "$VENV_PATH" ]]; then
  success "Venv already exists at $VENV_PATH — skipping creation."
else
  "$PY310" -m venv "$VENV_PATH"
  success "Created venv at $VENV_PATH."
fi

# Activate venv for the remainder of this script
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
success "Venv activated."

# ── Step 4: Install dpvc ──────────────────────────────────────────────────────
info "Step 4/8: Installing dpvc package..."
pip install --quiet -e "$REPO_ROOT"
success "dpvc installed."

# ── Step 5: Install fairseq stack (order matters) ────────────────────────────
info "Step 5/8: Installing fairseq and dependencies (this may take a few minutes)..."

# pip >= 24.1 rejects omegaconf 2.0.6's metadata; downgrade first.
pip install --quiet 'pip<24.1'
pip install --quiet 'omegaconf==2.0.6'
pip install --quiet 'git+https://github.com/facebookresearch/fairseq.git@v0.12.2'
pip install --quiet torchaudio librosa soundfile joblib amfm_decompy matplotlib gdown

success "fairseq stack installed."

# ── Step 6: Patch fairseq for PyTorch 2.6 ────────────────────────────────────
info "Step 6/8: Patching fairseq checkpoint loader for PyTorch 2.6+..."

python - <<'PYEOF'
import site, pathlib, sys

patched = False
for d in site.getsitepackages():
    f = pathlib.Path(d) / "fairseq/checkpoint_utils.py"
    if not f.exists():
        continue
    txt = f.read_text()
    old = 'torch.load(f, map_location=torch.device("cpu"))'
    new = 'torch.load(f, map_location=torch.device("cpu"), weights_only=False)'
    if old in txt:
        f.write_text(txt.replace(old, new))
        print(f"  Patched {f}")
        patched = True
    elif new in txt:
        print(f"  Already patched: {f}")
        patched = True

if not patched:
    print("  fairseq/checkpoint_utils.py not found — skipping patch.", file=sys.stderr)
PYEOF

success "fairseq patch applied."

# ── Step 7: Apply Apple Silicon MPS fix ──────────────────────────────────────
info "Step 7/8: Checking Apple Silicon MPS fix..."

READER="$CONTROL_VC_DIR/fairseq_feature_reader.py"
if [[ "$(uname -m)" == "arm64" ]]; then
  if [[ ! -f "$READER" ]]; then
    warn "fairseq_feature_reader.py not found at $READER — skipping MPS fix."
  elif grep -q 'mps.*is_available' "$READER" && ! grep -q 'Force CPU' "$READER"; then
    info "Applying MPS fix to $READER..."
    python - "$READER" <<'PYEOF'
import sys, pathlib, re

path = pathlib.Path(sys.argv[1])
txt = path.read_text()

old = (
    'DEVICE = torch.device("cuda" if torch.cuda.is_available()\n'
    '                      else "mps" if torch.backends.mps.is_available()\n'
    '                      else "cpu")'
)
new = (
    '# Force CPU — MPS does not support all fairseq operators\n'
    'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
)

if old in txt:
    path.write_text(txt.replace(old, new))
    print(f"  Applied MPS fix to {path}")
else:
    # Try a looser regex for slightly different whitespace
    import re
    patched = re.sub(
        r'DEVICE\s*=\s*torch\.device\("cuda"[^)]+mps[^)]+\)',
        '# Force CPU — MPS does not support all fairseq operators\n'
        'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
        txt,
    )
    if patched != txt:
        path.write_text(patched)
        print(f"  Applied MPS fix (regex) to {path}")
    else:
        print(f"  MPS block not found in expected form — inspect {path} manually.")
PYEOF
    success "MPS fix applied."
  else
    success "MPS fix already applied — skipping."
  fi
else
  success "Not Apple Silicon — MPS fix not needed."
fi

# ── Step 8: Download checkpoints ─────────────────────────────────────────────
info "Step 8/8: Checking checkpoints..."

CKPT_DIR="$CONTROL_VC_DIR/checkpoints"
HUBERT="$CKPT_DIR/hubert_base_ls960.pt"

if [[ -f "$HUBERT" ]]; then
  success "Checkpoints already present — skipping download."
else
  warn "Checkpoints not found at $CKPT_DIR"
  info "Attempting automatic download via gdown..."
  GDRIVE_FOLDER="https://drive.google.com/drive/folders/1APVHQFIb1871UhvymdK_oewWKJWrInYK"

  mkdir -p "$CKPT_DIR"
  if python -m gdown --folder "$GDRIVE_FOLDER" -O "$CKPT_DIR" 2>&1; then
    success "Checkpoints downloaded to $CKPT_DIR."
  else
    echo ""
    warn "Automatic download failed (Google Drive quota or access issue)."
    echo ""
    echo "  Please download checkpoints manually from:"
    echo "    $GDRIVE_FOLDER"
    echo ""
    echo "  Place these files in: $CKPT_DIR"
    echo "    ├── embed_f0stat2/"
    echo "    │   ├── config.json"
    echo "    │   └── g_00400000  (or g_00350000)"
    echo "    ├── 3000000-BL.ckpt"
    echo "    ├── hubert_base_ls960.pt   (~1.1 GB, required)"
    echo "    ├── km.bin                 (required)"
    echo "    └── vctk_f0_vq/"
    echo "        └── g_00400000"
    echo ""
    die "Setup incomplete — please download checkpoints and re-run scripts/setup.sh."
  fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo ""
echo "  Next steps:"
echo "    source $VENV_DIR/bin/activate"
echo "    python scripts/test_wrapper.py"
echo ""
echo "  Or, to use a custom control-vc path:"
echo "    CONTROL_VC_DIR=/path/to/control-vc python scripts/test_wrapper.py"
echo ""
