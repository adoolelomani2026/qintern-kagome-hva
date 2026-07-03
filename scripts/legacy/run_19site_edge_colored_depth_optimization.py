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
    bond_correlation_vector_numpy,
    bond_delocalization_metrics,
    build_heisenberg_ansatz,
    fixed_sector_fidelity,
    heisenberg_hamiltonian,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
    matching_from_bonds,
    max_abs_magnetization_numpy,
    ordered_group_names,
    random_dimer_coverings,
    rvb_state_from_coverings,
    singlet_product_state_from_pairs,
    validate_bond_groups_are_matchings,
)


NUM_SITES = 19
REFERENCE_ENERGY = None
PAULI_SCALE = 1.0
FALLBACK_TOL = 1e-10


def build_initial_state(mode: str, bonds, coverings: int, seed: int) -> tuple[np.ndarray, str]:
    if mode == "static":
        pairs = matching_from_bonds(bonds)
        return singlet_product_state_from_pairs(NUM_SITES, pairs), "static_dimer"
    if mode == "rvb":
        dimer_coverings = random_dimer_coverings(
            NUM_SITES,
            bonds,
            count=coverings,
            seed=seed,
            require_maximal=True,
        )
        return rvb_state_from_coverings(NUM_SITES, dimer_coverings), f"rvb_{len(dimer_coverings)}"
    raise ValueError("mode must be 'static' or 'rvb'")


def build_weighted_rvb_state(path: Path) -> tuple[np.ndarray, str]:
    data = np.load(path)
    sector_state = np.asarray(data["sector_state"])
    basis = np.asarray(data["basis"])
    full_state = np.zeros(2**NUM_SITES, dtype=complex)
    full_state[basis] = sector_state
    label = f"weighted_rvb_{int(data['coverings'])}"
    return full_state / np.linalg.norm(full_state), label


def qiskit_state(initial_state: np.ndarray, circuit, parameters, values: np.ndarray) -> Statevector:
    state = Statevector(initial_state)
    if len(parameters) == 0:
        return state
    bound = circuit.assign_parameters(dict(zip(parameters, values)), inplace=False)
    return state.evolve(bound)


def state_metrics(state_data: np.ndarray, bonds, exact) -> dict[str, float]:
    correlations = bond_correlation_vector_numpy(
        state_data,
        NUM_SITES,
        bonds,
        pauli_scale=PAULI_SCALE,
    )
    metrics = bond_delocalization_metrics(correlations)
    metrics.update(
        {
            "fidelity": fixed_sector_fidelity(state_data, exact),
            "entropy": bipartite_entropy_numpy(state_data),
            "max_magnetization": max_abs_magnetization_numpy(state_data, NUM_SITES),
        }
    )
    return metrics


def non_identity_random_start(
    rng: np.random.Generator,
    width: int,
    scale: float,
) -> np.ndarray:
    signs = rng.choice([-1.0, 1.0], size=width)
    magnitudes = rng.uniform(0.05, scale, size=width)
    return signs * magnitudes


def starts_for_depth(
    depth: int,
    num_colors: int,
    previous_best: np.ndarray | None,
    rng: np.random.Generator,
    restarts: int,
    scale: float,
) -> list[np.ndarray]:
    width = depth * num_colors
    starts: list[np.ndarray] = []

    # Avoid identity-dominated seeds.  These two starts are intentionally away
    # from 0 and pi/2, unlike the coarse scan.
    starts.append(np.full(width, np.pi / 8))
    starts.append(non_identity_random_start(rng, width, scale))

    if previous_best is not None:
        starts.append(
            np.concatenate(
                [
                    previous_best,
                    non_identity_random_start(rng, num_colors, scale),
                ]
            )
        )

    while len(starts) < restarts + 2:
        starts.append(non_identity_random_start(rng, width, scale))
    return starts


def optimize_depth(
    depth: int,
    initial_state: np.ndarray,
    bonds,
    groups,
    hamiltonian,
    starts: list[np.ndarray],
    maxfev: int,
    method: str,
):
    circuit, parameters = build_heisenberg_ansatz(
        NUM_SITES,
        bonds,
        layers=depth,
        parameterization="grouped",
        bond_groups=groups,
        initial_state="zero",
    )
    history: list[dict] = []
    evaluations = 0

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        state = qiskit_state(initial_state, circuit, parameters, values)
        energy = float(np.real(state.expectation_value(hamiltonian)))
        history.append({"depth": depth, "evaluation": evaluations, "energy": energy})
        return energy

    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method=method,
            options={"maxfev": maxfev, "maxiter": maxfev, "disp": False},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result

    if best is None:
        raise RuntimeError("No optimizer result")
    values = np.asarray(best.x, dtype=float)
    state = qiskit_state(initial_state, circuit, parameters, values)
    return best, values, np.asarray(state.data), evaluations, history, circuit, parameters


def main() -> None:
    global REFERENCE_ENERGY
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--maxfev", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--init", choices=["static", "rvb", "weighted"], default="rvb")
    parser.add_argument("--coverings", type=int, default=16)
    parser.add_argument(
        "--weighted-state",
        default=str(PROJECT_ROOT / "results" / "19site_weighted_rvb_state_n54.npz"),
    )
    parser.add_argument("--method", choices=["Powell", "Nelder-Mead", "COBYLA"], default="Powell")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--start-scale", type=float, default=0.35)
    args = parser.parse_args()

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    group_names = ordered_group_names(groups, len(bonds))
    print(f"edge-color matching check: {validate_bond_groups_are_matchings(bonds, groups)}")

    exact = load_sector_exact_result(PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz")
    REFERENCE_ENERGY = exact.energy
    hamiltonian = heisenberg_hamiltonian(NUM_SITES, bonds, pauli_scale=PAULI_SCALE)
    if args.init == "weighted":
        initial_state, init_label = build_weighted_rvb_state(Path(args.weighted_state))
    else:
        initial_state, init_label = build_initial_state(args.init, bonds, args.coverings, args.seed)

    rows: list[dict] = []
    history_rows: list[dict] = []

    p0_energy = float(np.real(Statevector(initial_state).expectation_value(hamiltonian)))
    p0_metrics = state_metrics(initial_state, bonds, exact)
    rows.append(
        {
            "initialization": init_label,
            "ansatz": "initial_state",
            "depth": 0,
            "parameters": 0,
            "energy": p0_energy,
            "error_vs_reference": p0_energy - REFERENCE_ENERGY,
            "evaluations": 1,
            "success": True,
            "message": "initial state",
            "best_parameters": "",
            **p0_metrics,
        }
    )
    print(f"{init_label} p=0: E={p0_energy:.8f}, F={p0_metrics['fidelity']:.6f}")

    previous_best = None
    for depth in range(1, args.max_layers + 1):
        starts = starts_for_depth(
            depth,
            len(group_names),
            previous_best,
            rng,
            restarts=args.restarts,
            scale=args.start_scale,
        )
        previous_for_safety = previous_best
        result, values, state_data, evaluations, history, circuit, parameters = optimize_depth(
            depth,
            initial_state,
            bonds,
            groups,
            hamiltonian,
            starts,
            maxfev=args.maxfev,
            method=args.method,
        )
        safety_values = [np.zeros(depth * len(group_names))]
        if previous_for_safety is not None:
            safety_values.append(
                np.concatenate([previous_for_safety, np.zeros(len(group_names))])
            )
        safety_message = ""
        for candidate in safety_values:
            candidate_state = qiskit_state(initial_state, circuit, parameters, candidate)
            candidate_energy = float(np.real(candidate_state.expectation_value(hamiltonian)))
            evaluations += 1
            if candidate_energy < float(result.fun) - FALLBACK_TOL:
                values = candidate
                state_data = np.asarray(candidate_state.data)
                result.fun = candidate_energy
                safety_message = " fallback_candidate_selected"
        previous_best = values
        metrics = state_metrics(state_data, bonds, exact)
        energy = float(result.fun)
        rows.append(
            {
                "initialization": init_label,
                "ansatz": "edge_colored_hva_optimized",
                "depth": depth,
                "parameters": len(values),
                "energy": energy,
                "error_vs_reference": energy - REFERENCE_ENERGY,
                "evaluations": evaluations,
                "success": bool(result.success),
                "message": str(result.message) + safety_message,
                "best_parameters": " ".join(f"{value:.10g}" for value in values),
                **metrics,
            }
        )
        history_rows.extend(history)
        print(
            f"{init_label} p={depth}: E={energy:.8f}, "
            f"err={energy - REFERENCE_ENERGY:.8f}, F={metrics['fidelity']:.6f}, "
            f"deloc={metrics['af_weight_participation']:.3f}, evals={evaluations}"
        )

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"19site_edge_colored_depth_optimization_{args.init}.csv"
    history_path = output_dir / f"19site_edge_colored_depth_optimization_{args.init}_history.csv"

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["depth", "evaluation", "energy"])
        writer.writeheader()
        writer.writerows(history_rows)

    print(f"elapsed={time.time() - start_time:.2f}s")
    print(f"Wrote {output_path}")
    print(f"Wrote {history_path}")


if __name__ == "__main__":
    main()
