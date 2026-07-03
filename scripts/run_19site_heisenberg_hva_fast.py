from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    bipartite_entropy_numpy,
    bond_correlation_vector_numpy,
    bond_delocalization_metrics,
    build_sector_bond_index_cache,
    embed_sector_state,
    fixed_sector_fidelity,
    heisenberg_sector_energy_cached_numpy,
    heisenberg_sector_state_from_initial_numpy,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
    matching_from_bonds,
    max_abs_magnetization_numpy,
    ordered_group_names,
    random_dimer_coverings,
    rvb_state_from_coverings,
    sector_state_from_full_state,
    singlet_product_state_from_pairs,
    validate_bond_groups_are_matchings,
)


NUM_SITES = 19
PAULI_SCALE = 1.0
FALLBACK_TOL = 1e-10


def load_initial_state(args, bonds) -> tuple[np.ndarray, str]:
    if args.init == "static":
        pairs = matching_from_bonds(bonds)
        state = singlet_product_state_from_pairs(NUM_SITES, pairs)
        return state, "static_dimer"
    if args.init == "rvb":
        coverings = random_dimer_coverings(
            NUM_SITES,
            bonds,
            count=args.coverings,
            seed=args.seed,
            require_maximal=True,
        )
        state = rvb_state_from_coverings(NUM_SITES, coverings)
        return state, f"rvb_{len(coverings)}"
    if args.init == "weighted":
        data = np.load(args.weighted_state)
        state = np.zeros(2**NUM_SITES, dtype=complex)
        state[np.asarray(data["basis"])] = np.asarray(data["sector_state"])
        return state / np.linalg.norm(state), f"weighted_rvb_{int(data['coverings'])}"
    raise ValueError("Unknown initialization")


def metrics_for_state(state: np.ndarray, bonds, exact) -> dict[str, float]:
    correlations = bond_correlation_vector_numpy(
        state,
        NUM_SITES,
        bonds,
        pauli_scale=PAULI_SCALE,
    )
    metrics = bond_delocalization_metrics(correlations)
    metrics.update(
        {
            "fidelity": fixed_sector_fidelity(state, exact),
            "entropy": bipartite_entropy_numpy(state),
            "max_magnetization": max_abs_magnetization_numpy(state, NUM_SITES),
        }
    )
    return metrics


def optimizer_options(method: str, maxiter: int) -> dict:
    if method == "COBYLA":
        return {"maxiter": maxiter, "rhobeg": 0.08, "tol": 1e-5, "disp": False}
    if method == "Nelder-Mead":
        return {"maxfev": maxiter, "maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-8, "disp": False}
    if method == "Powell":
        return {"maxfev": maxiter, "maxiter": maxiter, "xtol": 1e-5, "ftol": 1e-8, "disp": False}
    if method == "L-BFGS-B":
        return {"maxiter": maxiter, "maxfun": maxiter, "ftol": 1e-10, "gtol": 1e-6, "disp": False}
    raise ValueError(f"Unsupported method: {method}")


def non_identity_start(rng: np.random.Generator, width: int, scale: float) -> np.ndarray:
    signs = rng.choice([-1.0, 1.0], size=width)
    magnitudes = rng.uniform(0.02, scale, size=width)
    return signs * magnitudes


def starts_for_depth(
    depth: int,
    colors: int,
    rng: np.random.Generator,
    restarts: int,
    scale: float,
    previous: np.ndarray | None,
) -> list[dict[str, object]]:
    width = depth * colors
    raw_starts: list[tuple[str, np.ndarray]] = [
        ("small_angle_0.02", np.full(width, 0.02)),
        ("small_angle_0.03", np.full(width, 0.03)),
        ("small_angle_0.04", np.full(width, 0.04)),
        ("small_angle_0.05", np.full(width, 0.05)),
        ("small_angle_-0.02", np.full(width, -0.02)),
        ("small_angle_-0.03", np.full(width, -0.03)),
        ("small_angle_-0.04", np.full(width, -0.04)),
        ("alternating_small_angle", np.array([0.05 if i % 2 == 0 else -0.05 for i in range(width)])),
    ]
    if previous is not None:
        raw_starts.append(("warm_previous_plus_0.03", np.concatenate([previous, np.full(colors, 0.03)])))
        raw_starts.append(
            (
                "warm_previous_plus_random",
                np.concatenate([previous, non_identity_start(rng, colors, scale)]),
            )
        )
    for restart_index in range(restarts):
        raw_starts.append((f"random_restart_{restart_index}", non_identity_start(rng, width, scale)))
    return [
        {"start_id": index, "start_kind": kind, "values": values}
        for index, (kind, values) in enumerate(raw_starts)
    ]


def optimize_depth(
    depth: int,
    initial_state: np.ndarray,
    bond_caches,
    bonds,
    groups,
    starts: list[dict[str, object]],
    method: str,
    maxiter: int,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int, bool, str, list[dict]]:
    history: list[dict] = []
    evaluations = 0
    best_seen_energy = np.inf
    best_seen_values: np.ndarray | None = None
    active_start_id = -1
    active_start_kind = "unassigned"

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations, best_seen_energy, best_seen_values
        evaluations += 1
        active_values = np.asarray(values, dtype=float)
        state = heisenberg_sector_state_from_initial_numpy(
            initial_state,
            bond_caches,
            depth,
            active_values,
            parameterization="grouped",
            bond_groups=groups,
        )
        energy = heisenberg_sector_energy_cached_numpy(
            state,
            bond_caches,
            pauli_scale=PAULI_SCALE,
        )
        is_best_so_far = energy <= best_seen_energy + 1e-12
        if energy < best_seen_energy:
            best_seen_energy = float(energy)
            best_seen_values = active_values.copy()
        history.append(
            {
                "depth": depth,
                "start_id": active_start_id,
                "start_kind": active_start_kind,
                "evaluation": evaluations,
                "energy": energy,
                "best_so_far_energy": best_seen_energy,
                "is_best_so_far": is_best_so_far,
                "optimizer_method": method,
                "parameter_vector": " ".join(f"{value:.12g}" for value in active_values),
            }
        )
        return energy

    best = None
    for start in starts:
        active_start_id = int(start["start_id"])
        active_start_kind = str(start["start_kind"])
        result = minimize(
            objective,
            np.asarray(start["values"], dtype=float),
            method=method,
            options=optimizer_options(method, maxiter),
            bounds=bounds if method in {"L-BFGS-B", "Powell"} else None,
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result

    if best is None:
        raise RuntimeError("Optimizer did not produce a result")

    best_values = best_seen_values if best_seen_values is not None else np.asarray(best.x, dtype=float)
    best_state = heisenberg_sector_state_from_initial_numpy(
        initial_state,
        bond_caches,
        depth,
        best_values,
        parameterization="grouped",
        bond_groups=groups,
    )
    best_energy = heisenberg_sector_energy_cached_numpy(best_state, bond_caches, pauli_scale=PAULI_SCALE)
    return (
        best_values,
        best_state,
        best_energy,
        evaluations,
        bool(best.success),
        str(best.message),
        history,
    )


def apply_identity_safety(
    depth: int,
    colors: int,
    initial_state: np.ndarray,
    bond_caches,
    groups,
    candidates: list[np.ndarray],
    current_values: np.ndarray,
    current_state: np.ndarray,
    current_energy: float,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    message = ""
    for candidate in candidates:
        if len(candidate) != depth * colors:
            continue
        state = heisenberg_sector_state_from_initial_numpy(
            initial_state,
            bond_caches,
            depth,
            candidate,
            parameterization="grouped",
            bond_groups=groups,
        )
        energy = heisenberg_sector_energy_cached_numpy(state, bond_caches, pauli_scale=PAULI_SCALE)
        if energy < current_energy - FALLBACK_TOL:
            current_values = candidate
            current_state = state
            current_energy = energy
            message = " fallback_candidate_selected"
    return current_values, current_state, current_energy, message


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "unavailable"
    try:
        import qiskit

        versions["qiskit"] = qiskit.__version__
    except Exception:
        versions["qiskit"] = "unavailable"
    return versions


def thread_environment() -> dict[str, str | None]:
    return {
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def default_tag(args) -> str:
    method = args.method.lower().replace("-", "")
    return (
        f"p{args.max_layers}_{method}_mi{args.maxiter}_r{args.restarts}_"
        f"seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def relative_to_project(path: str | Path) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve()
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except (OSError, ValueError):
        return str(path).replace("\\", "/")


def write_metadata(path: Path, args, exact, output_path: Path, history_path: Path, elapsed: float) -> None:
    metadata = {
        "command": " ".join([Path(sys.executable).name, *sys.argv]).replace("\\", "/"),
        "seed": args.seed,
        "method": args.method,
        "maxiter": args.maxiter,
        "base_deterministic_starts": 8,
        "warm_start_candidates_per_depth_after_p1": 2,
        "random_restarts": args.restarts,
        "start_strategy": "small-angle deterministic + warm starts + optional random restarts",
        "start_scale": args.start_scale,
        "bounded": args.bounded,
        "bound": args.bound if args.bounded else None,
        "max_layers": args.max_layers,
        "initialization": args.init,
        "weighted_state": relative_to_project(args.weighted_state) if args.init == "weighted" else None,
        "reference_energy": exact.energy,
        "pauli_scale": PAULI_SCALE,
        "hamiltonian_convention": "unscaled Pauli: sum_<ij>(XX+YY+ZZ)",
        "bond_file": "data/19site_bonds.csv",
        "output_csv": relative_to_project(output_path),
        "history_csv": relative_to_project(history_path),
        "optimizer_backend": "fixed-Sz sector cached NumPy Heisenberg kernels",
        "elapsed_seconds": elapsed,
        "blas_threads": thread_environment(),
        "versions": package_versions(),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", choices=["static", "rvb", "weighted"], default="weighted")
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--method", choices=["COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B"], default="Nelder-Mead")
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--start-scale", type=float, default=0.18)
    parser.add_argument("--coverings", type=int, default=16)
    parser.add_argument("--tag", default="", help="Optional suffix for result filenames.")
    parser.add_argument("--auto-tag", action="store_true", help="Append method/budget/date tag to avoid overwriting results.")
    parser.add_argument("--bounded", action="store_true", help="Use theta bounds when the optimizer supports bounds.")
    parser.add_argument("--bound", type=float, default=0.15)
    parser.add_argument(
        "--weighted-state",
        default=str(PROJECT_ROOT / "results" / "19site_weighted_rvb_state_n54.npz"),
    )
    args = parser.parse_args()

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    colors = len(ordered_group_names(groups, len(bonds)))
    print(f"edge-color matching check: {validate_bond_groups_are_matchings(bonds, groups)}")

    exact = load_sector_exact_result(PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz")
    reference_energy = exact.energy
    initial_state_full, init_label = load_initial_state(args, bonds)
    initial_state = sector_state_from_full_state(initial_state_full, exact.basis)
    sector_norm = float(np.linalg.norm(initial_state))
    if sector_norm == 0.0:
        raise RuntimeError("Initial state has zero weight in the target fixed-Sz sector")
    initial_state = initial_state / sector_norm
    initial_state_full = embed_sector_state(initial_state, exact.basis, NUM_SITES)
    bond_caches = build_sector_bond_index_cache(exact.basis, bonds)

    rows: list[dict] = []
    history_rows: list[dict] = []

    init_energy = heisenberg_sector_energy_cached_numpy(initial_state, bond_caches, pauli_scale=PAULI_SCALE)
    init_metrics = metrics_for_state(initial_state_full, bonds, exact)
    rows.append(
        {
            "initialization": init_label,
            "depth": 0,
            "parameters": 0,
            "energy": init_energy,
            "error_vs_reference": init_energy - reference_energy,
            "energy_improvement_vs_initial": 0.0,
            "energy_improvement_vs_previous_depth": 0.0,
            "evaluations": 1,
            "optimizer_converged": True,
            "energy_improved": False,
            "energy_improved_vs_initial": False,
            "energy_improved_vs_previous": False,
            "message": "initial state",
            "best_parameters": "",
            **init_metrics,
        }
    )
    print(
        f"{init_label} p=0: E={init_energy:.8f}, "
        f"err={init_energy - reference_energy:.8f}, F={init_metrics['fidelity']:.6f}"
    )

    previous_best = None
    previous_depth_energy = init_energy
    previous_candidates: list[np.ndarray] = []
    for depth in range(1, args.max_layers + 1):
        starts = starts_for_depth(
            depth,
            colors,
            rng,
            restarts=args.restarts,
            scale=args.start_scale,
            previous=previous_best,
        )
        values, state, energy, evaluations, success, message, history = optimize_depth(
            depth,
            initial_state,
            bond_caches,
            bonds,
            groups,
            starts,
            method=args.method,
            maxiter=args.maxiter,
            bounds=[(-args.bound, args.bound)] * (depth * colors) if args.bounded else None,
        )
        safety_candidates = [np.zeros(depth * colors)]
        safety_candidates.extend(previous_candidates)
        values, state, energy, safety_msg = apply_identity_safety(
            depth,
            colors,
            initial_state,
            bond_caches,
            groups,
            safety_candidates,
            values,
            state,
            energy,
        )
        message += safety_msg
        full_state = embed_sector_state(state, exact.basis, NUM_SITES)
        metrics = metrics_for_state(full_state, bonds, exact)
        previous_best = values
        previous_candidates = [
            np.concatenate([values, np.zeros(colors)]),
            np.concatenate([values, np.full(colors, 0.03)]),
        ]
        rows.append(
            {
                "initialization": init_label,
                "depth": depth,
                "parameters": len(values),
                "energy": energy,
                "error_vs_reference": energy - reference_energy,
                "energy_improvement_vs_initial": init_energy - energy,
                "energy_improvement_vs_previous_depth": previous_depth_energy - energy,
                "evaluations": evaluations,
                "optimizer_converged": success,
                "energy_improved": energy < init_energy - FALLBACK_TOL,
                "energy_improved_vs_initial": energy < init_energy - FALLBACK_TOL,
                "energy_improved_vs_previous": energy < previous_depth_energy - FALLBACK_TOL,
                "message": message,
                "best_parameters": " ".join(f"{value:.10g}" for value in values),
                **metrics,
            }
        )
        for item in history:
            item["initialization"] = init_label
            history_rows.append(item)
        print(
            f"{init_label} p={depth}: E={energy:.8f}, "
            f"err={energy - reference_energy:.8f}, "
            f"dE={init_energy - energy:.8f}, F={metrics['fidelity']:.6f}, "
            f"deloc={metrics['af_weight_participation']:.3f}, evals={evaluations}"
        )
        previous_depth_energy = energy

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    active_tag = args.tag or (default_tag(args) if args.auto_tag else "")
    tag = f"_{active_tag}" if active_tag else ""
    output_path = output_dir / f"19site_heisenberg_hva_fast_{args.init}{tag}.csv"
    history_path = output_dir / f"19site_heisenberg_hva_fast_{args.init}{tag}_history.csv"

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "initialization",
                "depth",
                "start_id",
                "start_kind",
                "evaluation",
                "energy",
                "best_so_far_energy",
                "is_best_so_far",
                "optimizer_method",
                "parameter_vector",
            ],
        )
        writer.writeheader()
        writer.writerows(history_rows)

    elapsed = time.time() - start_time
    metadata_path = output_path.with_suffix(".metadata.json")
    write_metadata(metadata_path, args, exact, output_path, history_path, elapsed)
    print(f"elapsed={elapsed:.2f}s")
    print(f"Wrote {output_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
