# HPC Quantum Simulation Guide: GPU Scaling, MPI & Frameworks

**System Context:** NCSA DeltaAI (NVIDIA GH200 Grace Hopper, HPE Slingshot 11, Cray MPICH) & General HPC GPU Clusters (e.g., NERSC Perlmutter)  
**Source Reference:** [NCSA DeltaAI Documentation — Quantum Simulation Resources](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/software.html#quantum-simulation-resources)  
**Scope:** Architecture sizing, GPU acceleration, multi-node scaling, framework recipes (CUDA-Q, cuQuantum, PennyLane, Qiskit Aer), and cluster troubleshooting.

---

## 1. Quick Decision Matrix & Workload Selection

Choose the right framework and execution pattern based on your scientific workload:

| Workload Type | Recommended Framework & Backend | Scaling Mechanism | Key Advantage / Use Case |
|---|---|---|---|
| **Large Single Circuit (34–37+ qubits)** | **PennyLane** (`lightning.gpu`, `mpi=True`) or<br>**CUDA-Q** (`option="mgpu,fp64"`) | Distributed state vector over MPI | Splits the state vector across multiple GPUs and nodes. |
| **Intra-Node Scaling (up to ~33 qubits)** | **Qiskit Aer** (`cuStateVec_enable=True`, `blocking_enable=True`) | cuStateVec GPU blocking (single process, 4 GPUs) | Distributes memory across 4 GPUs **without needing MPI**. |
| **Parameter Sweeps / Variational (VQE / QAOA)** | **Qiskit Aer** + **`mpi4py`** worker pool | Embarrassingly parallel (1 independent GPU simulator per rank) | Bypasses MPI communication bottlenecks; robust and linear scaling. |
| **Circuit Batching (within single node)** | **CUDA-Q** (`option="mqpu"`) | Native circuit dispatch across GPUs | One process evaluates circuit batches concurrently across local GPUs. |
| **Large Low-Entanglement Circuits** | **PennyLane** (`lightning.tensor`) or<br>**CUDA-Q** (`tensornet`) | Tensor networks (cuTensorNet) | Memory scales with entanglement entropy rather than $2^n$. |

---

## 2. State Vector Memory & GPU Sizing Rules

A state vector for $n$ qubits requires $2^n$ complex amplitudes.
* **Double precision (`complex128`, default):** $\text{Memory} = 2^n \times 16\text{ bytes}$
* **Single precision (`complex64`):** $\text{Memory} = 2^n \times 8\text{ bytes}$

Each NVIDIA GH200 GPU features 120 GB HBM3 (~95–97 GB usable after driver/runtime allocation). For NVIDIA A100 (80 GB) or H100 (80 GB), usable memory is ~70–75 GB.

### Minimum GPU Requirements (complex128, 4 GPUs per node)

| Qubit Count ($n$) | State Vector Size | Minimum GPUs Needed | Topology |
|---|---|---|---|
| **30** | 16 GiB | 1 GPU | Single GPU |
| **32** | 64 GiB | 1 GPU (GH200) / 2 GPUs (A100/H100) | Single GPU / 1 node |
| **33** | 128 GiB | 2 GPUs | 1 node |
| **34** | 256 GiB | 4 GPUs | 1 node (all 4 GPUs) |
| **35** | 512 GiB | 8 GPUs | 2 nodes |
| **36** | 1 TiB | 16 GPUs | 4 nodes |
| **37** | 2 TiB | 32 GPUs | 8 nodes |

> [!TIP]
> **Half-Precision Headroom (`complex64`):**
> Switching to single precision cuts state vector memory by **50%**, allowing a 33-qubit circuit on a single GH200 GPU, or 35 qubits on a single 4-GPU node.
> - **PennyLane:** Pass `c_dtype=np.complex64` to `qml.device()`.
> - **Qiskit Aer:** Set `precision='single'` in `AerSimulator()`.
> - **CUDA-Q:** Pass `option="mgpu,fp32"`.
> 
> *Note:* Verify critical numerical algorithms (such as expectation value convergence in deep circuits) in `complex128` before switching to `complex64`.

---

## 3. NVIDIA CUDA Quantum (CUDA-Q)

CUDA-Q is NVIDIA's framework for hybrid quantum-classical computing with native multi-GPU and MPI execution.

### Environment Setup
```bash
module load python/cuda_quantum/0.14.0
conda activate base
python -c "import cudaq; print(cudaq.__version__)"
```

### Available Targets
* `nvidia`: Single-GPU state vector simulation (default).
* `nvidia` with `option="mgpu,fp64"`: Multi-GPU distributed state vector via MPI.
* `nvidia` with `option="mqpu"`: Circuit batching across GPUs without MPI.
* `tensornet`: Tensor network simulation for large structured circuits.

---

### Example: Single-GPU Circuit
```python
import cudaq

cudaq.set_target("nvidia")

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

result = cudaq.sample(bell, shots_count=1000)
print(result)
```

**Single-GPU SLURM Script (`cudaq_single.slurm`):**
```bash
#!/bin/bash
#SBATCH --account=<account_name>
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --job-name=cudaq-single

module load python/cuda_quantum/0.14.0
conda activate base

python -u my_circuit.py
```

---

### Example: Multi-GPU Distributed State Vector (4 GPUs, 34+ Qubits)
```python
# multi_gpu_cudaq.py
import cudaq

cudaq.mpi.initialize()
cudaq.set_target("nvidia", option="mgpu,fp64")

N = 34  # ~256 GiB (complex128) -> ~64 GiB per GPU on 4 GPUs

@cudaq.kernel
def ghz(n: int):
    qubits = cudaq.qvector(n)
    h(qubits[0])
    for i in range(1, n):
        cx(qubits[0], qubits[i])

result = cudaq.sample(ghz, N, shots_count=1000)

if cudaq.mpi.rank() == 0:
    print(f"GHZ({N}) distributed across {cudaq.mpi.num_ranks()} GPUs")

cudaq.mpi.finalize()
```

**Multi-GPU SLURM Script (`cudaq_mgpu.slurm`):**
```bash
#!/bin/bash
#SBATCH --account=<account_name>
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --job-name=cudaq-mgpu

module load python/cuda_quantum/0.14.0
conda activate base

srun python -u multi_gpu_cudaq.py
```

> [!WARNING]
> **CRITICAL CUDA-Q MPI BINDING RULE:**  
> Do **NOT** set `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` before launching CUDA-Q multi-node jobs.  
> The CUDA-Q `nvidia-mgpu` plugin internally enumerates all GPUs on each node and binds one GPU per rank. Manual pre-pinning prevents the plugin from selecting devices, causing segmentation faults on all ranks at multi-node scale.

---

## 4. PennyLane (Lightning Backends)

PennyLane provides auto-differentiation and source-compiled Lightning backends linked to Cray MPICH and cuQuantum.

### Environment Setup
```bash
module load python/pennylane/0.44
conda activate pennylane-0.44
python -c "import pennylane as qml; print(qml.__version__)"
```

### Lightning Backends
* `lightning.gpu`: GPU state vector via cuStateVec (MPI-enabled for distributed multi-GPU).
* `lightning.tensor`: GPU tensor network via cuTensorNet.
* `lightning.kokkos`: Kokkos accelerated backend (MPI-enabled).
* `lightning.qubit`: Optimized CPU state vector.

---

### Example: Single-GPU Circuit
```python
import pennylane as qml

dev = qml.device("lightning.gpu", wires=2)

@qml.qnode(dev)
def bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

probs = bell()
print(f"Probabilities: {probs}")
```

---

### Example: Distributed Multi-GPU (4 GPUs, 34+ Qubits)
```python
# multi_gpu_pennylane.py
import pennylane as qml
import numpy as np

N = 34  # ~256 GiB (complex128 default)
# Note: pass c_dtype=np.complex64 to halve memory consumption if needed
dev = qml.device("lightning.gpu", wires=N, mpi=True)

@qml.qnode(dev)
def ghz():
    qml.Hadamard(wires=0)
    for i in range(1, N):
        qml.CNOT(wires=[0, i])
    return qml.probs(wires=range(min(N, 5)))

probs = ghz()
print(f"GHZ({N}) top probabilities: {probs[:3]}")
```

**Multi-GPU SLURM Script (`pennylane_mgpu.slurm`):**
```bash
#!/bin/bash
#SBATCH --account=<account_name>
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --job-name=pl-mgpu

module load python/pennylane/0.44
conda activate pennylane-0.44

srun python -u multi_gpu_pennylane.py
```

---

## 5. Qiskit Aer (GPU Acceleration & Scaling)

Qiskit Aer provides cuStateVec-accelerated state vector simulation for Qiskit workflows.

### Environment Setup
```bash
module load python/miniforge3_qiskit_aer/2.4.2
conda activate base
python -c "import qiskit_aer; print(qiskit_aer.__version__)"
```

---

### Pattern A: Intra-Node 4-GPU cuStateVec Blocking (Up to ~33 Qubits, No MPI)
This distributes an $N$-qubit state vector across all 4 GPUs on a single node without requiring MPI setup.

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Rule of thumb: blocking_qubits = N - 2
# Leaves 2 qubits of state per shard so all 4 shards fit across the 4 GPUs with headroom
N = 33
sim = AerSimulator(
    method='statevector',
    device='GPU',
    cuStateVec_enable=True,
    blocking_enable=True,
    blocking_qubits=N - 2
)

qc = QuantumCircuit(N)
qc.h(0)
for i in range(1, N):
    qc.cx(0, i)
qc.measure_all()

result = sim.run(qc, shots=1000).result()
print("Simulation complete.")
```

*Performance Benchmark:* A 33-qubit state vector at `complex128` (128 GiB total) finishes in ~5 s; a 30-qubit circuit completes in ~1.1 s.

---

### Pattern B: Multi-Node Parameter Sweeps (`mpi4py` Worker Pool)

> [!CAUTION]
> **Aer Multi-Node GPU Bug:**  
> In Qiskit Aer 0.17.2, native MPI-distributed state vector execution segfaults on GPUs when `MPI world size >= 2`.  
> For true distributed state vectors across nodes, use **PennyLane `lightning.gpu` with `mpi=True`**.  
> For parameter sweeps, QAOA, or VQE grids across nodes, use the embarrassingly parallel **`mpi4py` worker pool** below:

```python
# mpi_parameter_sweep.py
from mpi4py import MPI
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

# Parameter grid
grid = [(g, b) for g in np.linspace(0.1, 1.5, 10)
               for b in np.linspace(0.1, 1.5, 10)]

# Interleaved round-robin partition: rank claims grid[i] where i % nranks == rank
my_indices = list(range(rank, len(grid), nranks))

sim = AerSimulator(method="statevector", device="GPU", cuStateVec_enable=True)

local_results = []
for idx in my_indices:
    gamma, beta = grid[idx]
    qc = QuantumCircuit(8)
    qc.h(range(8))
    # ... apply circuit operations with gamma and beta ...
    qc.measure_all()
    res = sim.run(qc, shots=2048).result()
    local_results.append((idx, gamma, beta, res.get_counts()))

# Gather all chunks back to rank 0
all_results = comm.gather(local_results, root=0)

if rank == 0:
    flat = [item for chunk in all_results for item in chunk]
    print(f"Aggregated {len(flat)} evaluation points across {nranks} GPU ranks.")
```

**SLURM Script for `mpi4py` Parameter Sweep (`aer_mpi_sweep.slurm`):**
```bash
#!/bin/bash
#SBATCH --account=<account_name>
#SBATCH --partition=ghx4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18      # 1/4 of a 72-core NUMA per rank
#SBATCH --time=00:15:00
#SBATCH --job-name=aer-mpi-sweep

module load python/miniforge3_qiskit_aer/2.4.2
conda activate base

srun bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID python -u mpi_parameter_sweep.py'
```

---

## 6. NVIDIA cuQuantum SDK Import Changes (v26.01.0+)

When working with cuQuantum directly:
```bash
module load python/miniforge3_cuquantum/26.01.0
conda activate base
```

Import path changes in v26.01.0+:
* `import custatevec` $\rightarrow$ `from cuquantum.bindings import custatevec`
* `import cutensornet` $\rightarrow$ `from cuquantum.bindings import cutensornet`
* `from cuquantum import CircuitToEinsum` $\rightarrow$ `from cuquantum.tensornet import CircuitToEinsum`

---

## 7. HPC MPI & Network Fabric Troubleshooting

Two environment variables resolve known failure modes on Cray MPICH / HPE Slingshot 11 (CXI) networks:

### 1. `MPI_Finalize` Crashes in Optimization Loops
* **Symptom:** MPI segfaults during `MPI_Finalize` after many MPI communicators are dynamically created (e.g. inside iterative VQE or CUDA-Q `nvidia-mgpu` loops).
* **Cause:** Mismatched CXI counter buffer sizing.
* **Fix:**
  ```bash
  export MPICH_OFI_CXI_COUNTER_REPORT=0
  ```

### 2. Intra-Node Cross-Memory Attach Permission Failure
* **Symptom:** Job fails with error:
  `process_vm_readv: Operation not permitted`
* **Cause:** Cray MPICH single-copy Cross-Memory Attach (CMA) optimization failing Linux kernel ptrace / process permission checks.
* **Fix:** Fall back to two-copy intra-node transfers:
  ```bash
  export MPICH_SMP_SINGLE_COPY_MODE=NONE
  ```
  *(Performance impact is negligible since inter-node bandwidth is the primary bottleneck).*

---

## 8. HPC Python Best Practices: The `venv` Overlay Pattern

On HPC clusters, system modules typically set `PYTHONNOUSERSITE=1` to prevent stale packages in `~/.local` from shadowing optimized system libraries. Consequently, `pip install --user <pkg>` appears to succeed but **cannot be imported**.

### The Recommended Solution: Virtual Environment Overlay
Create a virtual environment with `--system-site-packages` layered on top of the cluster's base module. It inherits the high-performance CUDA/MPI stacks while allowing custom package installations:

```bash
# 1. Load the target module and activate conda base
module load python/miniforge3_pytorch/2.11.0    # or python/miniforge3_qiskit_aer/2.4.2
conda activate base

# 2. Create the venv in high-speed scratch or work storage
python -m venv --system-site-packages /pscratch/sd/a/<username>/my_quantum_env

# 3. Activate the venv and install needed packages
source /pscratch/sd/a/<username>/my_quantum_env/bin/activate
pip install humanize lz4

# 4. Overriding a base package version if required
pip install --ignore-installed tabulate==0.8.10
```

### Registering with Open OnDemand / JupyterLab
To make this environment available as a Jupyter kernel in Open OnDemand:
```bash
source /pscratch/sd/a/<username>/my_quantum_env/bin/activate
setup-kernel
```
The kernel will appear automatically in JupyterLab as `Python (<module>-<venv_name>)`.
