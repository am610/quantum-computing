#!/usr/bin/env python3
"""
High-Qubit State Vector Scaling Benchmark (GHZ States up to 34+ Qubits).

Physics:
    GHZ(N) = (|00...0> + |11...1>) / sqrt(2)
    Standard multipartite entanglement benchmark across quantum hardware and HPC simulators.

HPC Scaling Patterns:
    1. Single-GPU cuStateVec statevector simulation (up to ~32-33 qubits on GH200 / A100).
    2. Intra-node 4-GPU cuStateVec blocking (blocking_enable=True, blocking_qubits=N-2)
       allowing state vectors up to 34 qubits (256 GiB) without MPI.
    3. Half-precision testing: complex128 vs complex64 (saving 50% memory).
"""

import argparse
import json
import os
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def build_ghz_circuit(n_qubits: int) -> QuantumCircuit:
    """Build an N-qubit GHZ entanglement circuit."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def compute_statevector_memory_bytes(n_qubits: int, precision: str = "double") -> float:
    """Calculate theoretical memory required to store state vector in GiB."""
    bytes_per_amp = 16 if precision == "double" else 8
    total_bytes = (2 ** n_qubits) * bytes_per_amp
    return total_bytes / (1024 ** 3)


def parse_args():
    parser = argparse.ArgumentParser(description="High-Qubit State Vector Scaling Benchmark")
    parser.add_argument("--min-qubits", type=int, default=24, help="Min qubit count (default: 24)")
    parser.add_argument("--max-qubits", type=int, default=32, help="Max qubit count (default: 32)")
    parser.add_argument("--shots", type=int, default=2048, help="Measurement shots (default: 2048)")
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"], help="Device")
    parser.add_argument("--precision", type=str, default="double", choices=["single", "double"], help="Precision")
    parser.add_argument("--blocking", action="store_true", help="Enable cuStateVec 4-GPU intra-node blocking")
    parser.add_argument("--out", type=str, default="ghz_scaling_results.json", help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print("  High-Qubit State Vector Scaling Benchmark: GHZ Entanglement")
    print(f"  Qubit Range: {args.min_qubits} to {args.max_qubits}")
    print(f"  Device: {args.device} | Precision: {args.precision} | Blocking: {args.blocking}")
    print(f"  Shots: {args.shots}")
    print("=" * 72)

    results = []

    for n in range(args.min_qubits, args.max_qubits + 1):
        mem_gib = compute_statevector_memory_bytes(n, args.precision)
        print(f"\n[Running N={n} qubits | Theoretical Memory: {mem_gib:.3f} GiB]")

        sim_kwargs = {
            "method": "statevector",
            "device": args.device,
            "precision": args.precision,
        }

        if args.device == "GPU":
            sim_kwargs["cuStateVec_enable"] = True
            if args.blocking and n >= 4:
                # Rule of thumb: blocking_qubits = N - 2
                sim_kwargs["blocking_enable"] = True
                sim_kwargs["blocking_qubits"] = n - 2

        try:
            sim = AerSimulator(**sim_kwargs)
            qc = build_ghz_circuit(n)

            t0_compile = time.perf_counter()
            compiled_qc = transpile(qc, sim, optimization_level=1)
            t_compile = time.perf_counter() - t0_compile

            t0_exec = time.perf_counter()
            job = sim.run(compiled_qc, shots=args.shots)
            sim_result = job.result()
            t_exec = time.perf_counter() - t0_exec

            counts = sim_result.get_counts()
            all_zeros = "0" * n
            all_ones = "1" * n
            fidelity = (counts.get(all_zeros, 0) + counts.get(all_ones, 0)) / args.shots

            record = {
                "qubits": n,
                "memory_gib": round(mem_gib, 4),
                "compile_time_sec": round(t_compile, 4),
                "sim_time_sec": round(t_exec, 4),
                "ghz_fidelity": round(fidelity, 4),
                "status": "SUCCESS",
            }
            print(f"  => Complete! Sim Time: {t_exec:.3f}s | Compile Time: {t_compile:.3f}s | Fidelity: {fidelity:.4f}")
            results.append(record)

        except Exception as e:
            print(f"  => Failed at N={n}: {e}")
            results.append({
                "qubits": n,
                "memory_gib": round(mem_gib, 4),
                "status": "FAILED",
                "error": str(e),
            })
            # If out of memory or failed, break loop
            break

    with open(args.out, "w") as f:
        json.dump({"meta": vars(args), "benchmarks": results}, f, indent=2)

    print("\n" + "=" * 72)
    print(f"  Scaling Benchmark finished. Results saved to: {os.path.abspath(args.out)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
