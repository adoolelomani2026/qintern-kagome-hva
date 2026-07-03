from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    bipartite_entropy_numpy,
    build_heisenberg_ansatz,
    exact_ground_state_fixed_sz,
    fixed_sector_fidelity,
    heisenberg_hamiltonian,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
    max_abs_magnetization_numpy,
    ordered_group_names,
    save_sector_exact_result,
    validate_bond_groups_are_matchings,
)


NUM_SITES = 19
N_DOWN = 9
PAULI_SCALE = 1.0
REFERENCE_ENERGY = None


def qiskit_expectations(circuit, hamiltonian, parameter_values=None) -> np.ndarray:
    estimator = StatevectorEstimator()
    if parameter_values is None:
        job = estimator.run([(circuit, hamiltonian)])
    else:
        job = estimator.run([(circuit, hamiltonian, np.asarray(parameter_values))])
    return np.atleast_1d(job.result()[0].data.evs)


def shared_candidates(layers: int) -> np.ndarray:
    return np.array(
        [
            np.zeros(layers),
            np.full(layers, np.pi / 8),
            np.full(layers, np.pi / 4),
            np.full(layers, np.pi / 2),
        ]
    )


def edge_color_candidates(layers: int, num_colors: int) -> np.ndarray:
    width = layers * num_colors
    candidates = [
        np.zeros(width),
        np.full(width, np.pi / 8),
        np.full(width, np.pi / 2),
    ]
    for color in range(num_colors):
        values = np.zeros((layers, num_colors))
        values[:, color] = np.pi / 8
        candidates.append(values.reshape(-1))
    return np.array(candidates)


def bond_candidates(num_bonds: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    return np.array(
        [
            np.zeros(num_bonds),
            np.full(num_bonds, np.pi / 8),
            rng.uniform(-0.2, 0.2, num_bonds),
        ]
    )


def ry_edge_candidates(layers: int, num_colors: int) -> np.ndarray:
    width = NUM_SITES + layers * num_colors
    zero = np.zeros(width)
    weak_ry = np.zeros(width)
    weak_ry[:NUM_SITES] = 0.05 * np.array([1 if i % 2 == 0 else -1 for i in range(NUM_SITES)])
    weak_ry[NUM_SITES:] = np.pi / 8
    return np.array([zero, weak_ry])


def bind_state(circuit, parameters, values: np.ndarray) -> Statevector:
    if len(parameters) == 0:
        return Statevector.from_instruction(circuit)
    bound = circuit.assign_parameters(dict(zip(parameters, values)), inplace=False)
    return Statevector.from_instruction(bound)


def evaluate_family(
    rows: list[dict],
    label: str,
    layers: int,
    parameterization: str,
    circuit,
    parameters,
    hamiltonian,
    values: np.ndarray,
    exact,
) -> None:
    if len(parameters) == 0:
        energies = qiskit_expectations(circuit, hamiltonian)
        values = np.zeros((1, 0))
    else:
        energies = qiskit_expectations(circuit, hamiltonian, values)

    best_index = int(np.argmin(energies))
    best_values = np.asarray(values[best_index]).ravel()
    best_energy = float(energies[best_index])
    state = bind_state(circuit, parameters, best_values)
    state_data = np.asarray(state.data)

    rows.append(
        {
            "ansatz": label,
            "depth": layers,
            "parameters": len(parameters),
            "trials": len(energies),
            "energy": best_energy,
            "error_vs_reference": best_energy - REFERENCE_ENERGY,
            "fidelity": "" if exact is None else fixed_sector_fidelity(state_data, exact),
            "entropy": bipartite_entropy_numpy(state_data),
            "max_magnetization": max_abs_magnetization_numpy(state_data, NUM_SITES),
            "best_parameters": " ".join(f"{value:.10g}" for value in best_values),
            "engine": "qiskit.StatevectorEstimator",
        }
    )
    print(
        f"{label:28s} p={layers}: E={best_energy:.8f}, "
        f"err={best_energy - REFERENCE_ENERGY:.8f}, trials={len(energies)}"
    )


def load_or_compute_exact(cache_path: Path, bonds, skip_exact: bool):
    if skip_exact:
        return None
    if cache_path.exists():
        exact = load_sector_exact_result(cache_path)
        print(f"Loaded fixed-Sz exact state from {cache_path}")
        return exact

    print("Computing fixed-Sz exact state. This can take a few minutes.")
    exact = exact_ground_state_fixed_sz(
        NUM_SITES,
        bonds,
        n_down=N_DOWN,
        pauli_scale=PAULI_SCALE,
    )
    save_sector_exact_result(cache_path, exact)
    print(f"Exact fixed-Sz energy: {exact.energy:.8f}")
    print(f"Saved {cache_path}")
    return exact


def main() -> None:
    global REFERENCE_ENERGY
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--skip-exact", action="store_true")
    args = parser.parse_args()

    start = time.time()
    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    group_names = ordered_group_names(groups, len(bonds))
    print(f"edge-color matching check: {validate_bond_groups_are_matchings(bonds, groups)}")

    hamiltonian = heisenberg_hamiltonian(NUM_SITES, bonds, pauli_scale=PAULI_SCALE)
    exact_cache = PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz"
    exact_cache.parent.mkdir(exist_ok=True)
    exact = load_or_compute_exact(exact_cache, bonds, skip_exact=args.skip_exact)
    REFERENCE_ENERGY = exact.energy if exact is not None else -29.146168109350135

    rows: list[dict] = []
    p0_circuit, p0_params = build_heisenberg_ansatz(NUM_SITES, bonds, layers=0)
    evaluate_family(
        rows,
        "dimer_baseline",
        0,
        "shared",
        p0_circuit,
        p0_params,
        hamiltonian,
        np.zeros((1, 0)),
        exact,
    )

    for layers in range(1, args.max_layers + 1):
        shared_circuit, shared_params = build_heisenberg_ansatz(
            NUM_SITES,
            bonds,
            layers=layers,
            parameterization="shared",
        )
        evaluate_family(
            rows,
            "shared_hva",
            layers,
            "shared",
            shared_circuit,
            shared_params,
            hamiltonian,
            shared_candidates(layers),
            exact,
        )

        edge_circuit, edge_params = build_heisenberg_ansatz(
            NUM_SITES,
            bonds,
            layers=layers,
            parameterization="grouped",
            bond_groups=groups,
        )
        evaluate_family(
            rows,
            "edge_colored_hva",
            layers,
            "grouped",
            edge_circuit,
            edge_params,
            hamiltonian,
            edge_color_candidates(layers, len(group_names)),
            exact,
        )

    bond_circuit, bond_params = build_heisenberg_ansatz(
        NUM_SITES,
        bonds,
        layers=1,
        parameterization="bond",
    )
    evaluate_family(
        rows,
        "bond_dependent_p1_probe",
        1,
        "bond",
        bond_circuit,
        bond_params,
        hamiltonian,
        bond_candidates(len(bonds)),
        exact,
    )

    ry_circuit, ry_params = build_heisenberg_ansatz(
        NUM_SITES,
        bonds,
        layers=1,
        parameterization="grouped",
        bond_groups=groups,
        ry_layer=True,
    )
    evaluate_family(
        rows,
        "edge_color_plus_weak_ry",
        1,
        "grouped+ry",
        ry_circuit,
        ry_params,
        hamiltonian,
        ry_edge_candidates(1, len(group_names)),
        exact,
    )

    output_path = PROJECT_ROOT / "results" / "19site_edge_colored_candidate_scan.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"elapsed={time.time() - start:.2f}s")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
