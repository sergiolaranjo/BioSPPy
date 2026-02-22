#!/usr/bin/env bash
# ===========================================================================
#  BioSPPy - Local Source Installation Script
#  Installs BioSPPy from the local source tree in development (editable) mode
# ===========================================================================

set -e  # exit on first error

# ── colours (disabled when piping to file) ──────────────────────────────────
if [ -t 1 ]; then
    BOLD='\033[1m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    RED='\033[0;31m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    BOLD=''; GREEN=''; YELLOW=''; RED=''; CYAN=''; NC=''
fi

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "\n${BOLD}${CYAN}▸ $*${NC}"; }

# ── resolve script directory (= repo root) ──────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "BioSPPy source directory: $SCRIPT_DIR"

# ── parse arguments ─────────────────────────────────────────────────────────
INSTALL_MODE="dev"    # dev | full | user
WITH_GUI=true
WITH_OPTIONAL=false
WITH_TESTS=false
VENV_DIR=""
PYTHON_CMD=""

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Install BioSPPy from local source code.

Options:
  --dev             Install in editable/development mode (default)
  --full            Install as a regular package (non-editable)
  --user            Install in user space (--user flag)
  --venv DIR        Create and use a virtual environment in DIR
  --python CMD      Python interpreter to use (default: auto-detect)
  --with-optional   Also install optional dependencies (cvxopt, wfdb, pyBioSig)
  --with-tests      Also install test dependencies and run the test suite
  --no-gui          Skip GUI dependency check (tkinter)
  --help            Show this help message

Examples:
  ./install.sh                          # editable install, current Python
  ./install.sh --venv .venv             # create venv and install there
  ./install.sh --full --with-optional   # full install with all extras
  ./install.sh --dev --with-tests       # dev install + run tests
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)           INSTALL_MODE="dev";  shift ;;
        --full)          INSTALL_MODE="full"; shift ;;
        --user)          INSTALL_MODE="user"; shift ;;
        --venv)          VENV_DIR="$2";       shift 2 ;;
        --python)        PYTHON_CMD="$2";     shift 2 ;;
        --with-optional) WITH_OPTIONAL=true;  shift ;;
        --with-tests)    WITH_TESTS=true;     shift ;;
        --no-gui)        WITH_GUI=false;      shift ;;
        --help|-h)       usage ;;
        *) error "Unknown option: $1"; usage ;;
    esac
done

# ── find Python interpreter ─────────────────────────────────────────────────
step "Detecting Python interpreter"

if [[ -n "$PYTHON_CMD" ]]; then
    PY="$PYTHON_CMD"
elif command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    error "Python not found. Install Python 3.6+ and try again."
    exit 1
fi

PY_VERSION=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PY_MAJOR=$("$PY" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PY" -c "import sys; print(sys.version_info.minor)")

info "Using: $PY (Python $PY_VERSION)"

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 6 ]]; }; then
    error "Python 3.6+ is required. Found: $PY_VERSION"
    exit 1
fi

# ── virtual environment ─────────────────────────────────────────────────────
if [[ -n "$VENV_DIR" ]]; then
    step "Setting up virtual environment: $VENV_DIR"

    if [[ -d "$VENV_DIR" ]]; then
        info "Virtual environment already exists at $VENV_DIR"
    else
        info "Creating virtual environment..."
        "$PY" -m venv "$VENV_DIR"
        info "Created: $VENV_DIR"
    fi

    # activate
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        source "$VENV_DIR/bin/activate"
    elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
        source "$VENV_DIR/Scripts/activate"
    else
        error "Cannot find activate script in $VENV_DIR"
        exit 1
    fi

    PY="python"  # inside the venv
    info "Virtual environment activated"
fi

# ── upgrade pip ─────────────────────────────────────────────────────────────
step "Upgrading pip, setuptools, wheel"
"$PY" -m pip install --upgrade pip setuptools wheel 2>&1 | tail -1

# ── install dependencies ────────────────────────────────────────────────────
step "Installing core dependencies"
"$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tail -3
info "Core dependencies installed"

if [[ "$WITH_OPTIONAL" == true ]]; then
    step "Installing optional dependencies"
    "$PY" -m pip install -r "$SCRIPT_DIR/requirements-optional.txt" 2>&1 | tail -3 || \
        warn "Some optional dependencies failed to install (non-fatal)"
fi

# ── install BioSPPy ─────────────────────────────────────────────────────────
step "Installing BioSPPy ($INSTALL_MODE mode)"

case "$INSTALL_MODE" in
    dev)
        "$PY" -m pip install -e "$SCRIPT_DIR" 2>&1 | tail -3
        info "Installed in editable (development) mode"
        info "Changes to source files take effect immediately"
        ;;
    full)
        "$PY" -m pip install "$SCRIPT_DIR" 2>&1 | tail -3
        info "Installed as a regular package"
        ;;
    user)
        "$PY" -m pip install --user -e "$SCRIPT_DIR" 2>&1 | tail -3
        info "Installed in user space (editable)"
        ;;
esac

# ── verify installation ────────────────────────────────────────────────────
step "Verifying installation"

"$PY" -c "
import biosppy
print(f'  biosppy version: {biosppy.__version__}')

# verify signal modules
from biosppy.signals import ecg, eda, emg, eeg, ppg, resp, bvp, abp, acc, pcg, hrv, tools
print('  Signal modules:  ecg, eda, emg, eeg, ppg, resp, bvp, abp, acc, pcg, hrv, tools  [OK]')

# verify feature modules
from biosppy.features import time, frequency, time_freq, cepstral, phase_space, wavelet_coherence
print('  Feature modules: time, frequency, time_freq, cepstral, phase_space, wavelet_coherence  [OK]')

# verify analysis modules
from biosppy import chaos, clustering, dimensionality_reduction, biometrics, quality
print('  Analysis modules: chaos, clustering, dim_reduction, biometrics, quality  [OK]')

# verify other modules
from biosppy import storage, plotting, stats, metrics, utils
print('  Utility modules: storage, plotting, stats, metrics, utils  [OK]')

# verify synthesizers
from biosppy.synthesizers import ecg as ecg_synth, emg as emg_synth
print('  Synthesizers:    ecg, emg  [OK]')

# verify signal-specific modules
from biosppy.signals import baroreflex, multichannel, emd
print('  Advanced:        baroreflex, multichannel, emd  [OK]')
"

if [[ $? -eq 0 ]]; then
    info "All modules verified successfully"
else
    error "Module verification failed"
    exit 1
fi

# ── check GUI (tkinter) ────────────────────────────────────────────────────
if [[ "$WITH_GUI" == true ]]; then
    step "Checking GUI dependencies (tkinter)"

    "$PY" -c "import tkinter" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        info "tkinter is available"
        "$PY" -c "
from biosppy.gui import run_gui
print('  GUI module:      biosppy.gui  [OK]')
print('  Launch with:     python -c \"from biosppy.gui import run_gui; run_gui()\"')
print('  Or:              python biosppy_gui.py')
"
    else
        warn "tkinter is not available - GUI will not work"
        warn "Install with:"
        warn "  Ubuntu/Debian: sudo apt-get install python3-tk"
        warn "  Fedora/RHEL:   sudo dnf install python3-tkinter"
        warn "  macOS:         brew install python-tk"
        warn "  Conda:         conda install tk"
        warn ""
        warn "BioSPPy library functions work fine without tkinter"
    fi
fi

# ── run tests ───────────────────────────────────────────────────────────────
if [[ "$WITH_TESTS" == true ]]; then
    step "Installing test dependencies"
    "$PY" -m pip install pytest 2>&1 | tail -1

    step "Running test suite"
    "$PY" -m pytest "$SCRIPT_DIR/tests/" -v --tb=short 2>&1 || \
        warn "Some tests failed (see output above)"

    if [[ -f "$SCRIPT_DIR/test_gui.py" ]]; then
        step "Running GUI structure tests"
        "$PY" "$SCRIPT_DIR/test_gui.py" 2>&1 || \
            warn "GUI tests had issues (see output above)"
    fi
fi

# ── summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  BioSPPy installed successfully!${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Version:    $("$PY" -c "import biosppy; print(biosppy.__version__)")"
echo "  Python:     $PY_VERSION"
echo "  Mode:       $INSTALL_MODE"
if [[ -n "$VENV_DIR" ]]; then
echo "  Venv:       $VENV_DIR"
fi
echo ""
echo "  Quick start:"
echo "    python -c \"from biosppy.signals import ecg; print(ecg.__doc__[:80])\""
echo ""
echo "  Launch GUI:"
echo "    python biosppy_gui.py"
echo ""
echo "  Run examples:"
echo "    python examples/baroreflex_example.py"
echo "    python examples/example_chaos_analysis.py"
echo ""
if [[ -n "$VENV_DIR" ]]; then
echo "  Activate venv later:"
echo "    source $VENV_DIR/bin/activate"
echo ""
fi
