#!/bin/bash
# ==============================================================================
# SLURM Batch Script: NCSA DeltaAI (NVIDIA GH200 Grace Hopper x 4)
# Quantum Simulation: Multi-GPU VQE / Parameter Sweep & GHZ Scaling
# ==============================================================================
#SBATCH --account=<your_delta_account>      # Replace with your approved DeltaAI account
#SBATCH --partition=ghx4                   # DeltaAI GH200 partition (or ghx4-interactive)
#SBATCH --nodes=1                          # Number of nodes (4 GPUs per node)
#SBATCH --ntasks-per-node=4                # 1 MPI rank per GPU
#SBATCH --gpus-per-node=4                  # 4 GH200 GPUs per node
#SBATCH --cpus-per-task=18                 # 18 CPU cores per rank (1/4 of 72-core NUMA domain)
#SBATCH --time=00:30:00                    # Walltime limit (HH:MM:SS)
#SBATCH --job-name=qc-deltaai-bench
#SBATCH --output=slurm_deltaai_%j.out
#SBATCH --error=slurm_deltaai_%j.err

echo "=== Job Started: $(date) on $(hostname) ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID | NODES: $SLURM_JOB_NUM_NODES | TASKS: $SLURM_NTASKS"

# 1. Load DeltaAI Qiskit Aer & cuQuantum module
module purge
module load python/miniforge3_qiskit_aer/2.4.2
conda activate base

# 2. Set Cray MPICH / HPE Slingshot 11 resilience flags
# Fix MPI_Finalize buffer crashes in iterative variational loops
export MPICH_OFI_CXI_COUNTER_REPORT=0

# Prevent Cross-Memory Attach (CMA) process_vm_readv permission errors
export MPICH_SMP_SINGLE_COPY_MODE=NONE

# 3. Benchmark 1: Distributed TFIM Parameter Sweep across all 4 GPUs via mpi4py
echo ""
echo ">>> Running Benchmark 1: Distributed TFIM Parameter Sweep across 4 GPUs..."
srun bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID python -u hpc/vqe_tfim_mpi_sweep.py \
    --spins 10 \
    --n-j 8 \
    --n-h 8 \
    --shots 2048 \
    --device GPU \
    --precision double \
    --out tfim_deltaai_results_${SLURM_JOB_ID}.json'

# 4. Benchmark 2: High-Qubit State Vector Scaling via cuStateVec 4-GPU Blocking
echo ""
echo ">>> Running Benchmark 2: High-Qubit Scaling (26 to 33 qubits) with 4-GPU blocking..."
python -u hpc/ghz_cuquantum_scaling.py \
    --min-qubits 26 \
    --max-qubits 33 \
    --shots 2048 \
    --device GPU \
    --precision double \
    --blocking \
    --out ghz_deltaai_results_${SLURM_JOB_ID}.json

echo ""
echo "=== Job Completed: $(date) ==="
