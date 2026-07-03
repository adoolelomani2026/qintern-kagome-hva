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
    enumerate_maximum_dimer_coverings,
    fixed_sector_fidelity,
    heisenberg_energy_cached_numpy,
    heisenberg_sector_state_from_initial_numpy,
    literature_benchmark_row,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
    matching_from_bonds,
    max_abs_magnetization_numpy,
    rvb_state_from_coverings,
    sector_state_from_full_state,
    sector_weight,
    singlet_product_state_from_pairs,
    spin_z_expectations_numpy,
)


NUM_SITES = 19
PAULI_SCALE = 1.0
SELECTED_ENTROPY_CUTS = [1, 4, 7, 9, 12, 15, 18]


def parse_parameters(text: str) -> np.ndarray:
    if not text or not str(text).strip():
        return np.array([], dtype=float)
    return np.array([float(piece) for piece in str(text).split()], dtype=float)


def load_weighted_state(path: Path) -> np.ndarray:
    data = np.load(path)
    full = np.zeros(2**NUM_SITES, dtype=complex)
    full[np.asarray(data["basis"])] = np.asarray(data["sector_state"])
    return full / np.linalg.norm(full)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_diagnostics_cache(results_dir: Path, bonds: list[tuple[int, int]]) -> dict[str, dict]:
    cache: dict[str, dict] = {
        "bond_correlations": {},
        "magnetization": {},
        "entropy": {},
        "paper": {},
    }
    bond_path = results_dir / "19site_bond_correlations_by_state.csv"
    if bond_path.exists():
        by_state: dict[str, dict[tuple[int, int], float]] = {}
        for row in read_csv_dicts(bond_path):
            by_state.setdefault(row["state"], {})[(int(row["i"]), int(row["j"]))] = float(
                row["correlation_unscaled_pauli"]
            )
        for state, values_by_bond in by_state.items():
            if all(bond in values_by_bond for bond in bonds):
                cache["bond_correlations"][state] = np.array([values_by_bond[bond] for bond in bonds])

    magnetization_path = results_dir / "19site_site_magnetization.csv"
    if magnetization_path.exists():
        by_state_sites: dict[str, dict[int, float]] = {}
        for row in read_csv_dicts(magnetization_path):
            by_state_sites.setdefault(row["state"], {})[int(row["site"])] = float(row["sz"])
        for state, values_by_site in by_state_sites.items():
            if all(site in values_by_site for site in range(NUM_SITES)):
                cache["magnetization"][state] = np.array(
                    [values_by_site[site] for site in range(NUM_SITES)]
                )

    entropy_path = results_dir / "19site_entropy_profiles.csv"
    if entropy_path.exists():
        for row in read_csv_dicts(entropy_path):
            cache["entropy"].setdefault(row["state"], {})[int(row["cut"])] = float(row["entropy"])
    paper_path = results_dir / "19site_paper_diagnostics.csv"
    if paper_path.exists():
        for row in read_csv_dicts(paper_path):
            cache["paper"][row["state"]] = {
                key: float(value)
                for key, value in row.items()
                if key != "state" and value not in {"", "nan", "inf", "-inf"}
            }
    return cache


def default_hva_csv() -> Path:
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
            max_depth = max(depths)
            best_energy = min(energies)
            latest_mtime = path.stat().st_mtime
            scored.append((best_energy, -max_depth, -latest_mtime, path))
    if not scored:
        raise FileNotFoundError("No serious weighted HVA CSV found in results/")
    selected = min(scored, key=lambda item: item[0])[3]
    print(f"Auto-selected HVA CSV: {selected}")
    return selected


def metrics_row(
    state_name: str,
    depth: int | str,
    state: np.ndarray,
    bond_caches,
    exact,
    exact_correlations: np.ndarray,
    exact_magnetization: np.ndarray,
    static_error: float,
    notes: str,
    circuit_refinement: str,
    calibration: str,
    diagnostics: dict[str, dict] | None = None,
    diagnostic_state: str | None = None,
) -> dict[str, float | str | int]:
    correlations = None
    magnetization = None
    entropy_midcut = None
    entropy_profile = None
    if diagnostics is not None and diagnostic_state:
        correlations = diagnostics["bond_correlations"].get(diagnostic_state)
        magnetization = diagnostics["magnetization"].get(diagnostic_state)
        entropy_by_cut = diagnostics["entropy"].get(diagnostic_state, {})
        paper_metrics = diagnostics["paper"].get(diagnostic_state, {})
        if all(cut in entropy_by_cut for cut in SELECTED_ENTROPY_CUTS):
            entropy_profile = np.array([entropy_by_cut[cut] for cut in SELECTED_ENTROPY_CUTS])
            entropy_midcut = entropy_by_cut.get(NUM_SITES // 2)
    else:
        paper_metrics = {}

    if correlations is None:
        correlations = cached_bond_correlations(state, bond_caches)
    if magnetization is None:
        magnetization = spin_z_expectations_numpy(state, NUM_SITES)
    if entropy_profile is None:
        entropy_profile = selected_entropy_profile(state)
    if entropy_midcut is None:
        entropy_midcut = bipartite_entropy_numpy(state)

    delocalization = bond_delocalization_metrics(correlations)
    energy = float(np.sum(correlations))
    error = energy - exact.energy
    corr_delta = correlations - exact_correlations
    mag_delta = magnetization - exact_magnetization
    if np.std(correlations) > 0.0 and np.std(exact_correlations) > 0.0:
        pearson = float(np.corrcoef(correlations, exact_correlations)[0, 1])
    else:
        pearson = float("nan")
    return {
        "state": state_name,
        "depth": depth,
        "circuit_refinement": circuit_refinement,
        "calibration": calibration,
        "energy_unscaled_pauli": energy,
        "error_vs_exact": error,
        "energy_physical_spin": energy / 4.0,
        "error_physical_spin": error / 4.0,
        "gap_closed_vs_static_percent": 100.0 * (1.0 - error / static_error) if static_error else float("nan"),
        "fidelity": fixed_sector_fidelity(state, exact),
        "entropy_midcut": entropy_midcut,
        "entropy_selected_cut_mean": float(np.mean(entropy_profile)),
        "entropy_selected_cut_max": float(np.max(entropy_profile)),
        "max_magnetization": max_abs_magnetization_numpy(state, NUM_SITES),
        "sector_weight": sector_weight(state, exact.basis),
        "bond_corr_min": delocalization["bond_corr_min"],
        "bond_corr_max": delocalization["bond_corr_max"],
        "bond_corr_mean": delocalization["bond_corr_mean"],
        "bond_corr_std": delocalization["bond_corr_std"],
        "af_weight_participation": delocalization["af_weight_participation"],
        "af_participation_ratio": delocalization["af_participation_ratio"],
        "strong_dimer_count": delocalization["strong_dimer_count"],
        "strong_dimer_fraction": delocalization["strong_dimer_fraction"],
        "strong_dimer_af_weight_fraction": delocalization["strong_dimer_af_weight_fraction"],
        "spin_graph_corr_length": paper_metrics.get("spin_graph_corr_length", float("nan")),
        "bond_corr_l2_error": float(np.linalg.norm(corr_delta)),
        "bond_corr_mae": float(np.mean(np.abs(corr_delta))),
        "bond_corr_pearson_corr": pearson,
        "magnetization_l2_error": float(np.linalg.norm(mag_delta)),
        "magnetization_mae": float(np.mean(np.abs(mag_delta))),
        "notes": notes,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def selected_entropy_profile(state: np.ndarray) -> np.ndarray:
    return np.array([bipartite_entropy_numpy(state, cut=cut) for cut in SELECTED_ENTROPY_CUTS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hva-csv", default="", help="Serious weighted-HVA CSV to include.")
    parser.add_argument(
        "--use-cached-diagnostics",
        action="store_true",
        help="Reuse results/19site_* diagnostics CSVs for correlations, magnetization, and entropy.",
    )
    parser.add_argument(
        "--weighted-state",
        default=str(PROJECT_ROOT / "results" / "19site_weighted_rvb_state_n54.npz"),
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "results"
    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    exact = load_sector_exact_result(PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz")
    diagnostics = load_diagnostics_cache(output_dir, bonds) if args.use_cached_diagnostics else None
    full_caches = build_bond_index_cache(NUM_SITES, bonds)
    exact_full = embed_sector_state(exact.state, exact.basis, NUM_SITES)
    weighted_state = load_weighted_state(Path(args.weighted_state))
    coverings = enumerate_maximum_dimer_coverings(NUM_SITES, bonds)
    static_state = singlet_product_state_from_pairs(NUM_SITES, matching_from_bonds(bonds))
    equal_rvb_state = rvb_state_from_coverings(NUM_SITES, coverings)
    if diagnostics is not None and "exact" in diagnostics["bond_correlations"]:
        exact_correlations = diagnostics["bond_correlations"]["exact"]
    else:
        exact_correlations = cached_bond_correlations(exact_full, full_caches)
    if diagnostics is not None and "exact" in diagnostics["magnetization"]:
        exact_magnetization = diagnostics["magnetization"]["exact"]
    else:
        exact_magnetization = spin_z_expectations_numpy(exact_full, NUM_SITES)
    if diagnostics is not None and "static_dimer" in diagnostics["bond_correlations"]:
        static_energy = float(np.sum(diagnostics["bond_correlations"]["static_dimer"]))
    else:
        static_energy = float(np.sum(cached_bond_correlations(static_state, full_caches)))
    static_error = static_energy - exact.energy

    rows: list[dict] = [
        metrics_row(
            "Static dimer",
            0,
            static_state,
            full_caches,
            exact,
            exact_correlations,
            exact_magnetization,
            static_error,
            "Greedy 9-singlet product state.",
            "none",
            "none",
            diagnostics,
            "static_dimer",
        ),
        metrics_row(
            "Equal RVB-54",
            0,
            equal_rvb_state,
            full_caches,
            exact,
            exact_correlations,
            exact_magnetization,
            static_error,
            "Equal superposition of all deterministic maximum dimer coverings.",
            "none",
            "none",
            diagnostics,
            "equal_rvb_54",
        ),
        metrics_row(
            "Weighted RVB-54",
            0,
            weighted_state,
            full_caches,
            exact,
            exact_correlations,
            exact_magnetization,
            static_error,
            "Classical generalized-eigenproblem initializer in the 54-covering dimer subspace.",
            "none",
            "none",
            diagnostics,
            "weighted_rvb_54",
        ),
    ]

    hva_csv = Path(args.hva_csv) if args.hva_csv else default_hva_csv()
    hva_rows = [row for row in read_csv_dicts(hva_csv) if int(row["depth"]) > 0]
    weighted_sector = sector_state_from_full_state(weighted_state, exact.basis)
    sector_caches = build_sector_bond_index_cache(exact.basis, bonds)
    for row in hva_rows:
        depth = int(row["depth"])
        parameters = parse_parameters(row.get("best_parameters", ""))
        if len(parameters) == 0:
            continue
        sector_state = heisenberg_sector_state_from_initial_numpy(
            weighted_sector,
            sector_caches,
            depth,
            parameters,
            parameterization="grouped",
            bond_groups=groups,
        )
        full_state = embed_sector_state(sector_state, exact.basis, NUM_SITES)
        rows.append(
            metrics_row(
                f"Weighted RVB + HVA p={depth}",
                depth,
                full_state,
                full_caches,
                exact,
                exact_correlations,
                exact_magnetization,
                static_error,
                f"Cached fixed-Sz Heisenberg-HVA refinement from {hva_csv.name}.",
                "edge-colored Heisenberg HVA",
                "none",
                diagnostics,
                f"weighted_hva_p{depth}",
            )
        )

    rows.append(
        metrics_row(
            "Exact",
            "exact",
            exact_full,
            full_caches,
            exact,
            exact_correlations,
            exact_magnetization,
            static_error,
            "Sparse exact diagonalization in the n_down=9 fixed-Sz sector.",
            "reference",
            "none",
            diagnostics,
            "exact",
        )
    )

    output_dir.mkdir(exist_ok=True)
    final_path = output_dir / "final_result_table.csv"
    write_rows(final_path, rows)
    literature_path = output_dir / "literature_benchmark_comparison.csv"
    write_rows(
        literature_path,
        [
            {
                "state": row["state"],
                **literature_benchmark_row(NUM_SITES, float(row["energy_unscaled_pauli"])),
            }
            for row in rows
            if row["state"] != "Exact" or str(row["energy_unscaled_pauli"]).strip()
        ],
    )

    comparison_rows = [
        {
            "method": row["state"],
            "energy_unscaled_pauli": row["energy_unscaled_pauli"],
            "error_vs_exact": row["error_vs_exact"],
            "fidelity": row["fidelity"],
            "calibration": row["calibration"],
            "status": "computed",
        }
        for row in rows
        if row["state"] != "Exact"
    ]
    calibration_path = output_dir / "19site_calibration_scan.csv"
    calibration_mode_rows = []
    calibration_top_rows = []
    best_calibration = None
    if calibration_path.exists():
        scan_rows = read_csv_dicts(calibration_path)
        if scan_rows:
            nontrivial_rows = [
                row
                for row in scan_rows
                if abs(float(row.get("jprime", 1.0)) - 1.0) > 1e-12
            ]
            comparable_rows = nontrivial_rows or scan_rows
            best = min(comparable_rows, key=lambda item: abs(float(item["target_energy_error"])))
            best_calibration = best
            comparison_rows.append(
                {
                    "method": f"Calibrated Hamiltonian scan ({best.get('scan_label', best.get('scan_group', 'scan'))})",
                    "energy_unscaled_pauli": best["target_energy"],
                    "error_vs_exact": best["target_energy_error"],
                    "fidelity": best["target_fidelity"],
                    "calibration": f"{best.get('scan_mode', 'group')} Jprime={best['jprime']}",
                    "status": "computed",
                }
            )
            for mode in sorted({row.get("scan_mode", "scan") for row in comparable_rows}):
                mode_rows = [row for row in comparable_rows if row.get("scan_mode", "scan") == mode]
                mode_best = min(mode_rows, key=lambda item: abs(float(item["target_energy_error"])))
                calibration_mode_rows.append(
                    {
                        "scan_mode": mode,
                        "best_scan_label": mode_best.get("scan_label", ""),
                        "jprime": mode_best["jprime"],
                        "target_energy": mode_best["target_energy"],
                        "target_energy_error": mode_best["target_energy_error"],
                        "target_fidelity": mode_best["target_fidelity"],
                    }
                )
            for rank, row in enumerate(
                sorted(comparable_rows, key=lambda item: abs(float(item["target_energy_error"])))[:10],
                start=1,
            ):
                calibration_top_rows.append(
                    {
                        "rank": rank,
                        "scan_mode": row.get("scan_mode", ""),
                        "scan_label": row.get("scan_label", ""),
                        "jprime": row["jprime"],
                        "target_energy": row["target_energy"],
                        "target_energy_error": row["target_energy_error"],
                        "target_fidelity": row["target_fidelity"],
                        "calibrated_bond_indices": row.get("calibrated_bond_indices", ""),
                        "calibrated_bonds": row.get("calibrated_bonds", ""),
                    }
                )
    else:
        comparison_rows.append(
            {
                "method": "Calibrated Hamiltonian baseline",
                "energy_unscaled_pauli": "",
                "error_vs_exact": "",
                "fidelity": "",
                "calibration": "Jprime scan",
                "status": "not run; use scripts/run_19site_calibration_scan.py",
            }
        )
    comparison_path = output_dir / "calibration_comparison.csv"
    write_rows(comparison_path, comparison_rows)
    mode_summary_path = output_dir / "calibration_mode_summary.csv"
    if calibration_mode_rows:
        write_rows(mode_summary_path, calibration_mode_rows)
    top10_path = output_dir / "calibration_top10.csv"
    if calibration_top_rows:
        write_rows(top10_path, calibration_top_rows)
    comparison_detail_path = output_dir / "no_calibration_vs_best_calibration.csv"
    if best_calibration is not None:
        hva_summary_rows = [
            row
            for row in rows
            if str(row["state"]).startswith("Weighted RVB + HVA p=")
        ]
        best_hva_row = min(hva_summary_rows, key=lambda row: abs(float(row["error_vs_exact"])))
        write_rows(
            comparison_detail_path,
            [
                {
                    "method": best_hva_row["state"],
                    "type": "no-calibration HVA baseline",
                    "energy_unscaled_pauli": best_hva_row["energy_unscaled_pauli"],
                    "error_vs_exact": best_hva_row["error_vs_exact"],
                    "fidelity": best_hva_row["fidelity"],
                    "circuit_preparable": "refinement circuit yes; initializer classical",
                },
                {
                    "method": f"Best calibration: {best_calibration.get('scan_label', '')}",
                    "type": "exact calibrated-Hamiltonian reference",
                    "energy_unscaled_pauli": best_calibration["target_energy"],
                    "error_vs_exact": best_calibration["target_energy_error"],
                    "fidelity": best_calibration["target_fidelity"],
                    "circuit_preparable": "not yet shown",
                },
            ],
        )

    print(f"Using HVA CSV: {hva_csv}")
    print(f"Wrote {final_path}")
    print(f"Wrote {literature_path}")
    print(f"Wrote {comparison_path}")
    if calibration_mode_rows:
        print(f"Wrote {mode_summary_path}")
    if calibration_top_rows:
        print(f"Wrote {top10_path}")
    if best_calibration is not None:
        print(f"Wrote {comparison_detail_path}")


if __name__ == "__main__":
    main()
