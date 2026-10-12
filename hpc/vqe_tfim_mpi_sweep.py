#!/usr/bin/env python3
"""
VQE / Parameter Sweep for the Transverse-Field Ising Model (TFIM) on Multi-GPU HPC.

Physics:
    H(J, h) = -J * sum_{i} Z_i Z_{i+1} - h * sum_{i} X_i
    Sweeps coupling J and transverse field h across a 2D parameter grid.
    Demonstrates quantum phase transition from ferromagnet (large J) to paramagnet (large h).

HPC Pattern:
    Embarrassingly parallel multi-node parameter sweep via mpi4py + Qiskit Aer GPU (cuStateVec).
    Avoids multi-node MPI statevector segfaults in Aer 0.17+ by assigning full circuits
    to individual GPU ranks and gathering results at rank 0 via comm.gather.
"""

import argparse
import json
import os
import sys
import time
import numpy as np

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


def build_tfim_hamiltonian(n_spins: int, j_val: float, h_val: float) -> SparsePauliOp:
    """Construct the TFIM Hamiltonian for n_spins with open boundary conditions."""
    pauli_list = []
    # ZZ interaction terms: -J * Z_i Z_{i+1}
    for i in range(n_spins - 1):
        label = ["I"] * n_spins
        label[i] = "Z"
        label[i + 1] = "Z"
        pauli_list.append(("".join(label), -float(j_val)))

    # Transverse field terms: -h * X_i
    for i in range(n_spins):
        label = ["I"] * n_spins
        label[i] = "X"
        pauli_list.append(("".join(label), -float(h_val)))

    return SparsePauliOp.from_list(pauli_list)


def build_ansatz(n_spins: int, depth: int = 2):
    """
    Construct a hardware-efficient parameterized ansatz.
    Alternating Ry single-qubit rotations and entangling CNOT layers.
    """
    qc = QuantumCircuit(n_spins)
    params = []
    param_idx = 0

    for d in range(depth):
        for q in range(n_spins):
            p = Parameter(f"theta_{param_idx}")
            params.append(p)
            qc.ry(p, q)
            param_idx += 1

        for q in range(n_spins - 1):
            qc.cx(q, q + 1)

    # Final rotation layer
    for q in range(n_spins):
        p = Parameter(f"theta_{param_idx}")
        params.append(p)
        qc.ry(p, q)
        param_idx += 1

    return qc, params


def compute_exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """Compute exact ground state energy via sparse/dense diagonalization (ground truth)."""
    matrix = hamiltonian.to_matrix()
    eigvals = np.linalg.eigvalsh(matrix)
    return float(np.min(eigvals))


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU MPI Parameter Sweep for TFIM")
    parser.add_argument("--spins", type=int, default=8, help="Number of spins/qubits (default: 8)")
    parser.add_argument("--depth", type=int, default=2, help="Ansatz circuit depth (default: 2)")
    parser.add_argument("--n-j", type=int, default=8, help="Number of J grid points (default: 8)")
    parser.add_argument("--n-h", type=int, default=8, help="Number of h grid points (default: 8)")
    parser.add_argument("--j-min", type=float, default=0.2, help="Min J value (default: 0.2)")
    parser.add_argument("--j-max", type=float, default=1.8, help="Max J value (default: 1.8)")
    parser.add_argument("--h-min", type=float, default=0.2, help="Min h value (default: 0.2)")
    parser.add_argument("--h-max", type=float, default=1.8, help="Max h value (default: 1.8)")
    parser.add_argument("--shots", type=int, default=4096, help="Measurement shots per evaluation (default: 4096)")
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"], help="Aer device (default: GPU)")
    parser.add_argument("--precision", type=str, default="double", choices=["single", "double"], help="Precision")
    parser.add_argument("--out", type=str, default="tfim_sweep_results.json", help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()

    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nranks = comm.Get_size()
    else:
        comm = None
        rank = 0
        nranks = 1

    if rank == 0:
        print("=" * 72)
        print("  HPC Multi-GPU VQE / Parameter Sweep: Transverse-Field Ising Model")
        print(f"  MPI Ranks: {nranks} | Spins (Qubits): {args.spins} | Device: {args.device}")
        print(f"  Grid Size: {args.n_j} x {args.n_h} = {args.n_j * args.n_h} evaluations")
        print(f"  Precision: {args.precision} | Shots: {args.shots}")
        print("=" * 72, flush=True)

    # Initialize AerSimulator for this rank's assigned GPU
    sim_kwargs = {
        "method": "statevector",
        "device": args.device,
        "precision": args.precision,
    }
    if args.device == "GPU":
        sim_kwargs["cuStateVec_enable"] = True

    try:
        sim = AerSimulator(**sim_kwargs)
        # Verify simulator is responsive
        probe_qc = QuantumCircuit(1)
        probe_qc.h(0)
        probe_qc.measure_all()
        sim.run(probe_qc, shots=10).result()
    except Exception as e:
        if args.device == "GPU":
            if rank == 0:
                print(f"[Rank {rank}] GPU initialization failed ({e}). Falling back to CPU.")
            sim_kwargs["device"] = "CPU"
            sim_kwargs.pop("cuStateVec_enable", None)
            sim = AerSimulator(**sim_kwargs)
        else:
            raise e

    # Build 2D parameter grid (J, h)
    j_vals = np.linspace(args.j_min, args.j_max, args.n_j)
    h_vals = np.linspace(args.h_min, args.h_max, args.n_h)
    grid = [(j, h) for j in j_vals for h in h_vals]
    n_points = len(grid)

    # Interleaved round-robin partition across MPI ranks
    my_indices = [idx for idx in range(n_points) if idx % nranks == rank]
    ansatz, params = build_ansatz(args.spins, depth=args.depth)

    # Random fixed ansatz angles for benchmark evaluation
    rng = np.random.default_rng(42)
    fixed_angles = rng.uniform(0, 2 * np.pi, len(params))
    bound_circuit = ansatz.assign_parameters(dict(zip(params, fixed_angles)))

    t0_rank = time.perf_counter()
    local_records = []

    for idx in my_indices:
        j_val, h_val = grid[idx]
        t_eval_start = time.perf_counter()

        hamiltonian = build_tfim_hamiltonian(args.spins, j_val, h_val)
        exact_e0 = compute_exact_ground_energy(hamiltonian) if args.spins <= 12 else None

        # Measure expectation value of each Pauli component
        estimated_energy = 0.0
        for pauli_str, coeff in hamiltonian.to_list():
            if abs(coeff) < 1e-12:
                continue
            diag_qc = bound_circuit.copy()
            # Rotate basis to diagonalize Pauli operator
            for q_idx, p_char in enumerate(reversed(pauli_str)):
                if p_char == "X":
                    diag_qc.h(q_idx)
                elif p_char == "Y":
                    diag_qc.sdg(q_idx)
                    diag_qc.h(q_idx)
            diag_qc.measure_all()

            compiled_qc = transpile(diag_qc, sim, optimization_level=1)
            counts = sim.run(compiled_qc, shots=args.shots).result().get_counts()

            # Expectation value: +1 for even parity, -1 for odd parity
            total_shots = sum(counts.values())
            parity_sum = 0
            for bitstr, cnt in counts.items():
                measured_qubits_parity = sum(int(b) for b in bitstr) % 2
                parity_sum += (-1) ** measured_qubits_parity * cnt
            exp_val = parity_sum / total_shots
            estimated_energy += coeff.real * exp_val

        dt_eval = time.perf_counter() - t_eval_start
        local_records.append({
            "idx": idx,
            "rank": rank,
            "j": round(float(j_val), 4),
            "h": round(float(h_val), 4),
            "vqe_energy": round(float(estimated_energy), 6),
            "exact_e0": round(float(exact_e0), 6) if exact_e0 is not None else None,
            "eval_time_sec": round(dt_eval, 4),
        })

    t_rank_total = time.perf_counter() - t0_rank

    if MPI_AVAILABLE and nranks > 1:
        all_gathered = comm.gather(local_records, root=0)
    else:
        all_gathered = [local_records]

    if rank == 0:
        flattened = [item for sublist in all_gathered for item in sublist]
        flattened.sort(key=lambda x: x["idx"])

        total_walltime = max(t_rank_total for sublist in all_gathered)
        throughput = len(flattened) / total_walltime if total_walltime > 0 else 0

        summary = {
            "meta": {
                "spins": args.spins,
                "grid_points": n_points,
                "nranks": nranks,
                "device": sim_kwargs["device"],
                "precision": args.precision,
                "shots": args.shots,
                "total_walltime_sec": round(total_walltime, 3),
                "throughput_evals_per_sec": round(throughput, 2),
            },
            "results": flattened,
        }

        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 72)
        print(f"  SWEEP COMPLETE in {total_walltime:.2f} seconds ({throughput:.2f} evals/sec)")
        print(f"  Total Evaluations: {len(flattened)} / {n_points}")
        print(f"  Results saved to:  {os.path.abspath(args.out)}")
        print("=" * 72)


if __name__ == "__main__":
    main()
