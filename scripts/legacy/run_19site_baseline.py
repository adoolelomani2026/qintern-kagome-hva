from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from qiskit.primitives import StatevectorEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    build_heisenberg_ansatz,
    heisenberg_hamiltonian,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
)


REFERENCE_ENERGY = None
PAULI_SCALE = 1.0
NUM_SITES = 19


def qiskit_expectations(circuit, hamiltonian, parameter_values=None) -> np.ndarray:
    """Evaluate energies with Qiskit's StatevectorEstimator."""

    estimator = StatevectorEstimator()
    if parameter_values is None:
        job = estimator.run([(circuit, hamiltonian)])
    else:
        job = estimator.run([(circuit, hamiltonian, np.asarray(parameter_values))])
    return np.atleast_1d(job.result()[0].data.evs)


def add_best_row(rows, label, layers, parameterization, values, energies) -> None:
    best_index = int(np.argmin(energies))
    best_energy = float(energies[best_index])
    best_parameters = np.asarray(values[best_index]).ravel() if len(values) else np.array([])
    rows.append(
        {
            "label": label,
            "layers": layers,
            "parameterization": parameterization,
            "energy": best_energy,
            "reference_energy": REFERENCE_ENERGY,
            "error": best_energy - REFERENCE_ENERGY,
            "evaluations": len(energies),
            "parameters": " ".join(f"{value:.10g}" for value in best_parameters),
            "engine": "qiskit.StatevectorEstimator",
        }
    )


def main() -> None:
    global REFERENCE_ENERGY
    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    REFERENCE_ENERGY = load_sector_exact_result(
        PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz"
    ).energy
    hamiltonian = heisenberg_hamiltonian(NUM_SITES, bonds, pauli_scale=PAULI_SCALE)

    rows = []
    start = time.time()

    p0_circuit, _ = build_heisenberg_ansatz(
        NUM_SITES,
        bonds,
        layers=0,
        parameterization="shared",
    )
    p0_energy = float(qiskit_expectations(p0_circuit, hamiltonian)[0])
    rows.append(
        {
            "label": "p0_dimer",
            "layers": 0,
            "parameterization": "shared",
            "energy": p0_energy,
            "reference_energy": REFERENCE_ENERGY,
            "error": p0_energy - REFERENCE_ENERGY,
            "evaluations": 1,
            "parameters": "",
            "engine": "qiskit.StatevectorEstimator",
        }
    )
    print(f"p0_dimer: E={p0_energy:.8f}, error={p0_energy - REFERENCE_ENERGY:.8f}")

    shared_circuit, _ = build_heisenberg_ansatz(
        NUM_SITES,
        bonds,
        layers=1,
        parameterization="shared",
    )
    shared_values = np.array([[theta] for theta in [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]])
    shared_energies = qiskit_expectations(shared_circuit, hamiltonian, shared_values)
    add_best_row(rows, "p1_shared_angle_scan", 1, "shared", shared_values, shared_energies)
    print("p1_shared_angle_scan:")
    for values, energy in zip(shared_values, shared_energies):
        print(f"  theta={values[0]:.6f}: E={energy:.8f}")

    grouped_circuit, _ = build_heisenberg_ansatz(
        NUM_SITES,
        bonds,
        layers=1,
        parameterization="grouped",
        bond_groups=groups,
    )
    grouped_values = [
        np.zeros(4),
        np.full(4, np.pi / 8),
        np.full(4, np.pi / 2),
    ]
    for color in range(4):
        values = np.zeros(4)
        values[color] = np.pi / 8
        grouped_values.append(values)
    grouped_values = np.array(grouped_values)
    grouped_energies = qiskit_expectations(grouped_circuit, hamiltonian, grouped_values)
    add_best_row(
        rows,
        "p1_edge_color_scan",
        1,
        "grouped",
        grouped_values,
        grouped_energies,
    )
    print("p1_edge_color_scan:")
    for values, energy in zip(grouped_values, grouped_energies):
        print(
            "  color0={:.6f}, color1={:.6f}, color2={:.6f}, color3={:.6f}: "
            "E={:.8f}".format(*values, energy)
        )

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "19site_qiskit_baseline.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"elapsed={time.time() - start:.2f}s")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
