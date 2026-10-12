#!/bin/bash
# ==============================================================================
# SLURM Batch Script: NERSC Perlmutter (NVIDIA A100 SXM4 x 4)
# Quantum Simulation: Multi-GPU VQE / Parameter Sweep & GHZ Scaling
# ==============================================================================
#SBATCH -A m4341                           # Or your NERSC account/repo
#SBATCH -C gpu                             # Perlmutter GPU nodes
#SBATCH -q regular                         # Queue: regular (or debug for quick tests)
#SBATCH -N 1                               # 1 node (4 NVIDIA A100 GPUs)
#SBATCH --ntasks-per-node=4                # 1 task per GPU
#SBATCH --gpus-per-node=4                  # 4 GPUs per node
#SBATCH -c 32                              # 32 CPU cores per task
#SBATCH -t 00:30:00                        # Walltime limit
#SBATCH -J qc-perlmutter-bench
#SBATCH -o slurm_perlmutter_%j.out
#SBATCH -e slurm_perlmutter_%j.err

echo "=== Job Started: $(date) on $(hostname) ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID | NODES: $SLURM_JOB_NUM_NODES | TASKS: $SLURM_NTASKS"

# 1. Environment & MPI Settings
module load cudatoolkit
module load cray-mpich

# Activate your Python virtual environment (if created)
if [ -d "$SCRATCH/my_quantum_env" ]; then
    source $SCRATCH/my_quantum_env/bin/activate
fi

# Cray MPICH Slingshot fabric resilience flags
export MPICH_OFI_CXI_COUNTER_REPORT=0
export MPICH_SMP_SINGLE_COPY_MODE=NONE

# 2. Benchmark 1: Distributed TFIM Parameter Sweep across all 4 GPUs via mpi4py
echo ""
echo ">>> Running Benchmark 1: Distributed TFIM Parameter Sweep across 4 A100 GPUs..."
srun bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID python -u hpc/vqe_tfim_mpi_sweep.py \
    --spins 10 \
    --n-j 8 \
    --n-h 8 \
    --shots 2048 \
    --device GPU \
    --precision double \
    --out tfim_perlmutter_results_${SLURM_JOB_ID}.json'

# 3. Benchmark 2: High-Qubit State Vector Scaling via cuStateVec 4-GPU Blocking
echo ""
echo ">>> Running Benchmark 2: High-Qubit Scaling (26 to 32 qubits) with 4-GPU blocking..."
python -u hpc/ghz_cuquantum_scaling.py \
    --min-qubits 26 \
    --max-qubits 32 \
    --shots 2048 \
    --device GPU \
    --precision double \
    --blocking \
    --out ghz_perlmutter_results_${SLURM_JOB_ID}.json

echo ""
echo "=== Job Completed: $(date) ==="
