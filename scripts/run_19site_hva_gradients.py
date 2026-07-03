from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    build_sector_bond_index_cache,
    heisenberg_sector_energy_cached_numpy,
    heisenberg_sector_state_from_initial_numpy,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
    ordered_group_names,
)


PAULI_SCALE = 1.0


def load_weighted_sector_state(path: Path) -> np.ndarray:
    data = np.load(path)
    state = np.asarray(data["sector_state"], dtype=complex)
    return state / np.linalg.norm(state)


def energy_at(
    initial_state: np.ndarray,
    caches,
    groups,
    values: np.ndarray,
) -> float:
    state = heisenberg_sector_state_from_initial_numpy(
        initial_state,
        caches,
        layers=1,
        parameters=values,
        parameterization="grouped",
        bond_groups=groups,
    )
    return heisenberg_sector_energy_cached_numpy(state, caches, pauli_scale=PAULI_SCALE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=1e-3)
    parser.add_argument(
        "--weighted-state",
        default=str(PROJECT_ROOT / "results" / "19site_weighted_rvb_state_n54.npz"),
    )
    args = parser.parse_args()

    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    exact = load_sector_exact_result(PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz")
    caches = build_sector_bond_index_cache(exact.basis, bonds)
    group_names = ordered_group_names(groups, len(bonds))
    initial_state = load_weighted_sector_state(Path(args.weighted_state))

    zero = np.zeros(len(group_names), dtype=float)
    baseline = energy_at(initial_state, caches, groups, zero)
    rows = []
    directions = [(group, np.eye(len(group_names))[index]) for index, group in enumerate(group_names)]
    directions.append(("all_colors_shared", np.ones(len(group_names))))
    directions.append(
        (
            "alternating_colors",
            np.array([1.0 if index % 2 == 0 else -1.0 for index in range(len(group_names))]),
        )
    )
    for label, direction in directions:
        plus = args.step * direction
        minus = -args.step * direction
        e_plus = energy_at(initial_state, caches, groups, plus)
        e_minus = energy_at(initial_state, caches, groups, minus)
        gradient = (e_plus - e_minus) / (2.0 * args.step)
        curvature = (e_plus - 2.0 * baseline + e_minus) / (args.step**2)
        rows.append(
            {
                "depth": 1,
                "direction": label,
                "theta0": 0.0,
                "energy_at_zero": baseline,
                "finite_difference_step": args.step,
                "energy_plus": e_plus,
                "energy_minus": e_minus,
                "gradient": gradient,
                "curvature": curvature,
            }
        )

    output_path = PROJECT_ROOT / "results" / "19site_hva_p1_gradients.csv"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"weighted-RVB E(theta=0)={baseline:.12f}")
    for row in rows:
        print(
            "{direction}: gradient={gradient:.8f}, curvature={curvature:.8f}".format(
                **row
            )
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
