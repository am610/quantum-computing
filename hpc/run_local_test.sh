#!/usr/bin/env bash
# ==============================================================================
# Quick Dry-Run & Verification Script (Runs locally in ~15-30 seconds)
# Tests both TFIM parameter sweep and GHZ scaling without requiring SLURM queue.
# ==============================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$( dirname "$DIR" )"

cd "$ROOT_DIR"

echo "======================================================================"
echo "  QC-HPC Dry-Run: Validating Simulation Pipelines"
echo "======================================================================"

# Detect Python environment
if [ -f "/pscratch/sd/a/ayanmitr/QC/env/qml_env/bin/python" ]; then
    PYTHON_EXEC="/pscratch/sd/a/ayanmitr/QC/env/qml_env/bin/python"
elif [ -d ".venv" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python3"
fi
echo "Using Python interpreter: $PYTHON_EXEC"

# 1. Run small TFIM parameter sweep (4 spins, 3x3 grid = 9 evals)
echo ""
echo ">>> [1/2] Testing TFIM parameter sweep (CPU/GPU auto-detect)..."
$PYTHON_EXEC hpc/vqe_tfim_mpi_sweep.py \
    --spins 4 \
    --n-j 3 \
    --n-h 3 \
    --shots 1024 \
    --device CPU \
    --out hpc/test_tfim_sweep.json

echo "TFIM test output:"
head -n 25 hpc/test_tfim_sweep.json

# 2. Run small GHZ scaling benchmark (8 to 14 qubits)
echo ""
echo ">>> [2/2] Testing GHZ state vector scaling (8 to 14 qubits)..."
$PYTHON_EXEC hpc/ghz_cuquantum_scaling.py \
    --min-qubits 8 \
    --max-qubits 14 \
    --shots 1024 \
    --device CPU \
    --out hpc/test_ghz_scaling.json

echo "GHZ test output:"
head -n 25 hpc/test_ghz_scaling.json

echo ""
echo "======================================================================"
echo "  ALL LOCAL DRY-RUN TESTS PASSED!"
echo "======================================================================"
