from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    exact_ground_state_fixed_sz,
    fixed_sz_heisenberg_sparse,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
    ordered_group_names,
)


NUM_SITES = 19
N_DOWN = 9
PAULI_SCALE = 1.0
FIELDNAMES = [
    "scan_mode",
    "scan_label",
    "calibrated_bond_indices",
    "calibrated_bonds",
    "scan_group",
    "jprime",
    "calibrated_hamiltonian_energy",
    "target_energy",
    "target_energy_error",
    "target_fidelity",
    "reference_energy",
    "pauli_scale",
]


def parse_float_list(text: str) -> list[float]:
    return [float(piece.strip()) for piece in text.split(",") if piece.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def scan_key(scan_mode: str, scan_label: str, jprime: float | str) -> tuple[str, str, str]:
    return (str(scan_mode), str(scan_label), f"{float(jprime):.12g}")


def write_scan_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def merge_scan_rows(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str, str], dict] = {}
    for row in [*existing, *new_rows]:
        merged[scan_key(row.get("scan_mode", ""), row.get("scan_label", ""), row.get("jprime", 0.0))] = row
    return list(merged.values())


def graph_triangles(bonds: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    edges = {tuple(sorted(bond)) for bond in bonds}
    sites = sorted({site for bond in bonds for site in bond})
    triangles = []
    for a_index, a in enumerate(sites):
        for b in sites[a_index + 1 :]:
            if tuple(sorted((a, b))) not in edges:
                continue
            for c in sites:
                if c <= b:
                    continue
                if tuple(sorted((a, c))) in edges and tuple(sorted((b, c))) in edges:
                    triangles.append((a, b, c))
    return triangles


def bond_indices_for_triangle(
    triangle: tuple[int, int, int],
    bonds: list[tuple[int, int]],
) -> list[int]:
    triangle_edges = {
        tuple(sorted((triangle[0], triangle[1]))),
        tuple(sorted((triangle[0], triangle[2]))),
        tuple(sorted((triangle[1], triangle[2]))),
    }
    return [
        index
        for index, bond in enumerate(bonds)
        if tuple(sorted(bond)) in triangle_edges
    ]


def calibration_targets(args, bonds, groups) -> list[dict[str, object]]:
    if args.scan_mode == "group":
        active_groups = ordered_group_names(groups, len(bonds)) if args.all_groups else [args.scan_group]
        targets = []
        for group in active_groups:
            if group not in set(groups):
                raise ValueError(f"Unknown scan group: {group}")
            targets.append(
                {
                    "scan_mode": "group",
                    "scan_label": group,
                    "indices": [index for index, active_group in enumerate(groups) if active_group == group],
                }
            )
        return targets

    if args.scan_mode == "bond":
        if args.all_bonds:
            indices = range(len(bonds))
        else:
            if args.bond_index < 0 or args.bond_index >= len(bonds):
                raise ValueError(f"bond-index must be 0..{len(bonds) - 1}")
            indices = [args.bond_index]
        return [
            {
                "scan_mode": "bond",
                "scan_label": f"bond_{index}_{bonds[index][0]}-{bonds[index][1]}",
                "indices": [index],
            }
            for index in indices
        ]

    triangles = graph_triangles(bonds)
    if args.scan_mode == "triangle":
        if args.all_triangles:
            indices = range(len(triangles))
        else:
            if args.triangle_index < 0 or args.triangle_index >= len(triangles):
                raise ValueError(f"triangle-index must be 0..{len(triangles) - 1}")
            indices = [args.triangle_index]
        return [
            {
                "scan_mode": "triangle",
                "scan_label": "triangle_{}_{}-{}-{}".format(index, *triangles[index]),
                "indices": bond_indices_for_triangle(triangles[index], bonds),
            }
            for index in indices
        ]

    raise ValueError(f"Unsupported scan mode: {args.scan_mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-mode", choices=["group", "bond", "triangle"], default="group")
    parser.add_argument("--scan-group", default="color3")
    parser.add_argument("--all-groups", action="store_true")
    parser.add_argument("--bond-index", type=int, default=0)
    parser.add_argument("--all-bonds", action="store_true")
    parser.add_argument("--triangle-index", type=int, default=0)
    parser.add_argument("--all-triangles", action="store_true")
    parser.add_argument("--jprimes", default="1.00,1.05,1.10,1.15,1.20,1.25,1.30")
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=300000)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge new rows into results/19site_calibration_scan.csv instead of replacing it.",
    )
    parser.add_argument(
        "--max-new-rows",
        type=int,
        default=0,
        help="Stop after writing this many new rows; useful for resumable long scans.",
    )
    args = parser.parse_args()

    start = time.time()
    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    exact = load_sector_exact_result(PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz")
    target_h, target_basis = fixed_sz_heisenberg_sparse(
        NUM_SITES,
        bonds,
        n_down=N_DOWN,
        pauli_scale=PAULI_SCALE,
    )
    if not np.array_equal(target_basis, exact.basis):
        raise RuntimeError("Target sparse basis does not match cached exact basis")

    targets = calibration_targets(args, bonds, groups)
    jprimes = parse_float_list(args.jprimes)
    output_path = PROJECT_ROOT / "results" / "19site_calibration_scan.csv"
    output_path.parent.mkdir(exist_ok=True)
    rows = merge_scan_rows(read_csv_rows(output_path), []) if args.append and output_path.exists() else []
    seen = {
        scan_key(row.get("scan_mode", ""), row.get("scan_label", ""), row.get("jprime", 0.0))
        for row in rows
    }
    new_rows_written = 0
    stop_requested = False
    for target in targets:
        target_indices = set(int(index) for index in target["indices"])
        for jprime in jprimes:
            if args.max_new_rows and new_rows_written >= args.max_new_rows:
                stop_requested = True
                break
            key = scan_key(str(target["scan_mode"]), str(target["scan_label"]), jprime)
            if key in seen:
                print(f"{target['scan_label']} Jprime={jprime:.2f}: cached")
                continue
            weights = [jprime if index in target_indices else 1.0 for index in range(len(bonds))]
            calibrated = exact_ground_state_fixed_sz(
                NUM_SITES,
                bonds,
                n_down=N_DOWN,
                pauli_scale=PAULI_SCALE,
                bond_weights=weights,
                tol=args.tol,
                maxiter=args.maxiter,
            )
            target_energy = float(np.real(np.vdot(calibrated.state, target_h @ calibrated.state)))
            target_fidelity = float(abs(np.vdot(exact.state, calibrated.state)) ** 2)
            rows.append(
                {
                    "scan_mode": target["scan_mode"],
                    "scan_label": target["scan_label"],
                    "calibrated_bond_indices": " ".join(str(index) for index in sorted(target_indices)),
                    "calibrated_bonds": " ".join(
                        f"{bonds[index][0]}-{bonds[index][1]}" for index in sorted(target_indices)
                    ),
                    "scan_group": target["scan_label"] if target["scan_mode"] == "group" else "",
                    "jprime": jprime,
                    "calibrated_hamiltonian_energy": calibrated.energy,
                    "target_energy": target_energy,
                    "target_energy_error": target_energy - exact.energy,
                    "target_fidelity": target_fidelity,
                    "reference_energy": exact.energy,
                    "pauli_scale": PAULI_SCALE,
                }
            )
            seen.add(key)
            new_rows_written += 1
            if args.append:
                write_scan_rows(output_path, rows)
            print(
                f"{target['scan_label']} Jprime={jprime:.2f}: "
                f"E_target={target_energy:.8f}, err={target_energy - exact.energy:.8f}, "
                f"F={target_fidelity:.6f}"
            )
        if stop_requested:
            break

    write_scan_rows(output_path, rows)

    print(f"elapsed={time.time() - start:.2f}s")
    print(f"new_rows_written={new_rows_written}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
