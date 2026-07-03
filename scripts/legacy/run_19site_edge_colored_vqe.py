from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    bipartite_entropy_numpy,
    build_heisenberg_ansatz,
    fixed_sector_fidelity,
    heisenberg_hamiltonian,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
    max_abs_magnetization_numpy,
    ordered_group_names,
    validate_bond_groups_are_matchings,
)


NUM_SITES = 19
REFERENCE_ENERGY = None
PAULI_SCALE = 1.0


def qiskit_state(circuit, parameters, values: np.ndarray) -> Statevector:
    if len(parameters) == 0:
        return Statevector.from_instruction(circuit)
    bound = circuit.assign_parameters(dict(zip(parameters, values)), inplace=False)
    return Statevector.from_instruction(bound)


def qiskit_energy(circuit, parameters, hamiltonian, values: np.ndarray) -> float:
    state = qiskit_state(circuit, parameters, values)
    return float(np.real(state.expectation_value(hamiltonian)))


def optimize_edge_colored_depth(
    depth: int,
    bonds,
    groups,
    hamiltonian,
    starts: list[np.ndarray],
    maxfev: int,
):
    circuit, parameters = build_heisenberg_ansatz(
        NUM_SITES,
        bonds,
        layers=depth,
        parameterization="grouped",
        bond_groups=groups,
    )
    evaluations = 0
    history: list[float] = []

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        energy = qiskit_energy(circuit, parameters, hamiltonian, values)
        history.append(energy)
        return energy

    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            options={
                "maxfev": maxfev,
                "xtol": 1e-4,
                "ftol": 1e-7,
                "disp": False,
            },
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result

    if best is None:
        raise RuntimeError("Optimizer did not produce a result")

    values = np.asarray(best.x, dtype=float)
    state = qiskit_state(circuit, parameters, values)
    return {
        "result": best,
        "circuit": circuit,
        "parameters": parameters,
        "values": values,
        "state": state,
        "evaluations": evaluations,
        "history": history,
    }


def starts_for_depth(
    depth: int,
    num_colors: int,
    previous_best: np.ndarray | None,
    rng: np.random.Generator,
    restarts: int,
) -> list[np.ndarray]:
    width = depth * num_colors
    starts = [np.zeros(width), np.full(width, np.pi / 2)]
    if previous_best is not None:
        starts.append(np.concatenate([previous_best, np.zeros(num_colors)]))
        starts.append(np.concatenate([previous_best, previous_best[-num_colors:]]))
    for _ in range(restarts):
        starts.append(rng.uniform(-0.25, 0.25, width))
    return starts


def main() -> None:
    global REFERENCE_ENERGY
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--maxfev", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    start_time = time.time()
    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    group_names = ordered_group_names(groups, len(bonds))
    print(f"edge-color matching check: {validate_bond_groups_are_matchings(bonds, groups)}")

    hamiltonian = heisenberg_hamiltonian(NUM_SITES, bonds, pauli_scale=PAULI_SCALE)
    exact = load_sector_exact_result(PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz")
    REFERENCE_ENERGY = exact.energy
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    previous_best = None

    p0_circuit, p0_params = build_heisenberg_ansatz(NUM_SITES, bonds, layers=0)
    p0_values = np.zeros(0)
    p0_state = qiskit_state(p0_circuit, p0_params, p0_values)
    p0_energy = qiskit_energy(p0_circuit, p0_params, hamiltonian, p0_values)
    rows.append(
        {
            "ansatz": "dimer_baseline",
            "depth": 0,
            "parameters": 0,
            "energy": p0_energy,
            "error_vs_reference": p0_energy - REFERENCE_ENERGY,
            "fidelity": fixed_sector_fidelity(p0_state.data, exact),
            "entropy": bipartite_entropy_numpy(p0_state.data),
            "max_magnetization": max_abs_magnetization_numpy(p0_state.data, NUM_SITES),
            "evaluations": 1,
            "success": True,
            "message": "baseline",
            "best_parameters": "",
            "engine": "qiskit.Statevector+scipy.Powell",
        }
    )
    print(f"dimer_baseline p=0: E={p0_energy:.8f}")

    for depth in range(1, args.max_layers + 1):
        starts = starts_for_depth(
            depth,
            len(group_names),
            previous_best,
            rng,
            restarts=args.restarts,
        )
        opt = optimize_edge_colored_depth(
            depth,
            bonds,
            groups,
            hamiltonian,
            starts,
            maxfev=args.maxfev,
        )
        result = opt["result"]
        values = opt["values"]
        state_data = np.asarray(opt["state"].data)
        energy = float(result.fun)
        previous_best = values
        rows.append(
            {
                "ansatz": "edge_colored_hva_optimized",
                "depth": depth,
                "parameters": len(values),
                "energy": energy,
                "error_vs_reference": energy - REFERENCE_ENERGY,
                "fidelity": fixed_sector_fidelity(state_data, exact),
                "entropy": bipartite_entropy_numpy(state_data),
                "max_magnetization": max_abs_magnetization_numpy(state_data, NUM_SITES),
                "evaluations": opt["evaluations"],
                "success": bool(result.success),
                "message": str(result.message),
                "best_parameters": " ".join(f"{value:.10g}" for value in values),
                "engine": "qiskit.Statevector+scipy.Powell",
            }
        )
        print(
            f"edge_colored_hva_optimized p={depth}: E={energy:.8f}, "
            f"err={energy - REFERENCE_ENERGY:.8f}, evals={opt['evaluations']}"
        )

    output_path = PROJECT_ROOT / "results" / "19site_edge_colored_vqe.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"elapsed={time.time() - start_time:.2f}s")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
