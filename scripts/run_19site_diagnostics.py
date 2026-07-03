from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    bipartite_entropy_numpy,
    bond_delocalization_metrics,
    build_bond_index_cache,
    build_sector_bond_index_cache,
    embed_sector_state,
    entropy_profile_numpy,
    exponential_decay_length_from_profile,
    enumerate_maximum_dimer_coverings,
    fixed_sector_fidelity,
    graph_distance_matrix,
    heisenberg_energy_cached_numpy,
    heisenberg_sector_state_from_initial_numpy,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
    matching_from_bonds,
    rvb_state_from_coverings,
    sector_state_from_full_state,
    singlet_product_state_from_pairs,
    spin_correlation_distance_profile,
    spin_z_expectations_numpy,
)


NUM_SITES = 19
PAULI_SCALE = 1.0


def parse_parameters(text: str) -> np.ndarray:
    if not text or not str(text).strip():
        return np.array([], dtype=float)
    return np.array([float(piece) for piece in str(text).split()], dtype=float)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def default_hva_csv() -> Path | None:
    matches = [
        path
        for path in sorted((PROJECT_ROOT / "results").glob("19site_heisenberg_hva_fast_weighted*.csv"))
        if "history" not in path.name and "smoke" not in path.name
    ]
    scored: list[tuple[float, int, float, Path]] = []
    for path in matches:
        try:
            rows = read_csv_dicts(path)
            depths = [int(row["depth"]) for row in rows if row.get("depth", "").strip()]
            energies = [float(row["energy"]) for row in rows if row.get("energy", "").strip()]
        except (OSError, KeyError, ValueError):
            continue
        if depths and energies:
            scored.append((min(energies), -max(depths), -path.stat().st_mtime, path))
    return min(scored, key=lambda item: item[0])[3] if scored else None


def load_weighted_state(path: Path) -> np.ndarray:
    data = np.load(path)
    state = np.zeros(2**NUM_SITES, dtype=complex)
    state[np.asarray(data["basis"])] = np.asarray(data["sector_state"])
    return state / np.linalg.norm(state)


def slug(text: str) -> str:
    return (
        text.lower()
        .replace("+", "plus")
        .replace("=", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = ["site", *[f"site_{index}" for index in range(matrix.shape[1])]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for site, row in enumerate(matrix):
            writer.writerow(
                {"site": site, **{f"site_{index}": float(value) for index, value in enumerate(row)}}
            )


def cached_bond_correlations(state: np.ndarray, bond_caches) -> np.ndarray:
    return np.array(
        [
            heisenberg_energy_cached_numpy(
                state,
                [cache],
                pauli_scale=PAULI_SCALE,
            )
            for cache in bond_caches
        ]
    )


def build_states(bonds, groups, exact) -> dict[str, np.ndarray]:
    exact_full = embed_sector_state(exact.state, exact.basis, NUM_SITES)
    weighted = load_weighted_state(PROJECT_ROOT / "results" / "19site_weighted_rvb_state_n54.npz")
    states: dict[str, np.ndarray] = {
        "exact": exact_full,
        "static_dimer": singlet_product_state_from_pairs(NUM_SITES, matching_from_bonds(bonds)),
        "equal_rvb_54": rvb_state_from_coverings(
            NUM_SITES,
            enumerate_maximum_dimer_coverings(NUM_SITES, bonds),
        ),
        "weighted_rvb_54": weighted,
    }

    hva_csv = default_hva_csv()
    if hva_csv is not None:
        weighted_sector = sector_state_from_full_state(weighted, exact.basis)
        caches = build_sector_bond_index_cache(exact.basis, bonds)
        for row in read_csv_dicts(hva_csv):
            depth = int(row["depth"])
            if depth <= 0:
                continue
            parameters = parse_parameters(row.get("best_parameters", ""))
            if len(parameters) == 0:
                continue
            sector_state = heisenberg_sector_state_from_initial_numpy(
                weighted_sector,
                caches,
                depth,
                parameters,
                parameterization="grouped",
                bond_groups=groups,
            )
            states[f"weighted_hva_p{depth}"] = embed_sector_state(
                sector_state,
                exact.basis,
                NUM_SITES,
            )
    return states


def entropy_cuts(mode: str) -> list[int] | None:
    if mode == "all":
        return None
    if mode == "selected":
        return [1, 4, 7, 9, 12, 15, 18]
    raise ValueError(f"Unknown entropy cut mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entropy-cuts",
        choices=["selected", "all"],
        default="selected",
        help="Use selected publication cuts by default; choose all for full 1..18 profiles.",
    )
    args = parser.parse_args()

    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    exact = load_sector_exact_result(PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz")
    states = build_states(bonds, groups, exact)
    active_cuts = entropy_cuts(args.entropy_cuts)
    bond_caches = build_bond_index_cache(NUM_SITES, bonds)
    distance_matrix = graph_distance_matrix(NUM_SITES, bonds)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    bond_rows = []
    site_rows = []
    entropy_rows = []
    paper_rows_by_state = {}
    for state_name, state in states.items():
        correlations = cached_bond_correlations(state, bond_caches)
        paper_rows_by_state[state_name] = {
            "state": state_name,
            **bond_delocalization_metrics(correlations),
        }
        for (i, j), value in zip(bonds, correlations):
            bond_rows.append(
                {
                    "state": state_name,
                    "i": i,
                    "j": j,
                    "correlation_unscaled_pauli": float(value),
                }
            )
        sz = spin_z_expectations_numpy(state, NUM_SITES)
        for site, value in enumerate(sz):
            site_rows.append({"state": state_name, "site": site, "sz": float(value)})
        cut_labels = list(range(1, NUM_SITES)) if active_cuts is None else active_cuts
        for cut, entropy in zip(cut_labels, entropy_profile_numpy(state, cuts=active_cuts)):
            entropy_rows.append({"state": state_name, "cut": cut, "entropy": float(entropy)})

    matrices = {
        state_name: np.eye(NUM_SITES, dtype=float) * (3.0 * PAULI_SCALE)
        for state_name in states
    }
    for i in range(NUM_SITES):
        for j in range(i + 1, NUM_SITES):
            cache = build_bond_index_cache(NUM_SITES, [(i, j)])
            for state_name, state in states.items():
                value = heisenberg_energy_cached_numpy(state, cache, pauli_scale=PAULI_SCALE)
                matrices[state_name][i, j] = value
                matrices[state_name][j, i] = value
    distance_rows = []
    for state_name, matrix in matrices.items():
        write_matrix_csv(results_dir / f"19site_spin_correlation_{slug(state_name)}.csv", matrix)
        profile = spin_correlation_distance_profile(matrix, distance_matrix)
        paper_rows_by_state[state_name]["spin_graph_corr_length"] = exponential_decay_length_from_profile(
            profile
        )
        for row in profile:
            distance_rows.append({"state": state_name, **row})

    with (results_dir / "19site_bond_correlations_by_state.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bond_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bond_rows)

    with (results_dir / "19site_site_magnetization.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(site_rows[0].keys()))
        writer.writeheader()
        writer.writerows(site_rows)

    with (results_dir / "19site_entropy_profiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entropy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(entropy_rows)

    paper_rows = list(paper_rows_by_state.values())
    with (results_dir / "19site_paper_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paper_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paper_rows)

    with (results_dir / "19site_spin_distance_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(distance_rows)

    exact_by_bond = {
        (row["i"], row["j"]): row["correlation_unscaled_pauli"]
        for row in bond_rows
        if row["state"] == "exact"
    }
    static_by_bond = {
        (row["i"], row["j"]): row["correlation_unscaled_pauli"]
        for row in bond_rows
        if row["state"] == "static_dimer"
    }
    legacy_rows = [
        {
            "i": i,
            "j": j,
            "exact_correlation": exact_by_bond[(i, j)],
            "dimer_correlation": static_by_bond[(i, j)],
            "difference": static_by_bond[(i, j)] - exact_by_bond[(i, j)],
        }
        for i, j in bonds
    ]
    with (results_dir / "19site_bond_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(legacy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(legacy_rows)

    print(f"states: {', '.join(states)}")
    for state_name, state in states.items():
        correlations = cached_bond_correlations(state, bond_caches)
        print(
            f"{state_name}: E={float(np.sum(correlations)):.12f}, "
            f"F={fixed_sector_fidelity(state, exact):.12f}, "
            f"S_mid={bipartite_entropy_numpy(state):.6f}, "
            f"bond range={correlations.min():.6f} to {correlations.max():.6f}, "
            f"spin_xi_graph={paper_rows_by_state[state_name]['spin_graph_corr_length']:.3f}"
        )
    print(f"Wrote diagnostics to {results_dir}")


if __name__ == "__main__":
    main()
