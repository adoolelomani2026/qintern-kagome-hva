from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    bipartite_entropy_numpy,
    bond_correlation_vector_numpy,
    bond_delocalization_metrics,
    fixed_sector_fidelity,
    fixed_sz_heisenberg_sparse,
    enumerate_maximum_dimer_coverings,
    load_bonds_csv,
    load_sector_exact_result,
    max_abs_magnetization_numpy,
    random_dimer_coverings,
    sector_state_from_full_state,
    singlet_product_state_from_pairs,
)


NUM_SITES = 19
N_DOWN = 9
REFERENCE_ENERGY = None
PAULI_SCALE = 1.0
OVERLAP_THRESHOLD = 1e-10


def parse_checkpoints(text: str, max_coverings: int) -> list[int]:
    values = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise ValueError("checkpoints must be positive")
        values.append(min(value, max_coverings))
    values.append(max_coverings)
    return sorted(set(values))


def sector_columns_from_coverings(
    coverings,
    basis: np.ndarray,
) -> np.ndarray:
    columns = []
    for pairs in coverings:
        full_state = singlet_product_state_from_pairs(NUM_SITES, pairs)
        columns.append(sector_state_from_full_state(full_state, basis))
    return np.column_stack(columns)


def generalized_lowest_state(
    h_sub: np.ndarray,
    overlap: np.ndarray,
    threshold: float = OVERLAP_THRESHOLD,
) -> tuple[float, np.ndarray, int]:
    """Solve H c = E S c with overlap pruning for near dependencies."""

    overlap = 0.5 * (overlap + overlap.conj().T)
    h_sub = 0.5 * (h_sub + h_sub.conj().T)

    s_vals, s_vecs = np.linalg.eigh(overlap)
    keep = s_vals > threshold * max(1.0, float(np.max(s_vals)))
    if not np.any(keep):
        raise RuntimeError("Dimer overlap matrix has no numerically stable subspace")

    transform = s_vecs[:, keep] / np.sqrt(s_vals[keep])
    h_orth = transform.conj().T @ h_sub @ transform
    eigvals, eigvecs = np.linalg.eigh(0.5 * (h_orth + h_orth.conj().T))
    coeffs = transform @ eigvecs[:, 0]
    norm = np.sqrt(np.real(coeffs.conj().T @ overlap @ coeffs))
    coeffs = coeffs / norm
    return float(np.real(eigvals[0])), coeffs, int(np.sum(keep))


def overlap_condition(overlap: np.ndarray, threshold: float = OVERLAP_THRESHOLD) -> tuple[int, float, np.ndarray]:
    values = np.linalg.eigvalsh(0.5 * (overlap + overlap.conj().T))
    keep = values > threshold * max(1.0, float(np.max(values)))
    if not np.any(keep):
        return 0, float("inf"), values
    kept = values[keep]
    return int(np.sum(keep)), float(np.max(kept) / np.min(kept)), values


def normalize_sector_state(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("Cannot normalize zero state")
    return state / norm


def embed_sector_state(sector_state: np.ndarray, basis: np.ndarray) -> np.ndarray:
    full = np.zeros(2**NUM_SITES, dtype=complex)
    full[basis] = sector_state
    return full


def state_summary(
    label: str,
    sector_state: np.ndarray,
    basis: np.ndarray,
    h_sector,
    bonds,
    exact,
) -> dict[str, float | str]:
    sector_state = normalize_sector_state(sector_state)
    energy = float(np.real(np.vdot(sector_state, h_sector @ sector_state)))
    full_state = embed_sector_state(sector_state, basis)
    correlations = bond_correlation_vector_numpy(
        full_state,
        NUM_SITES,
        bonds,
        pauli_scale=PAULI_SCALE,
    )
    return {
        f"{label}_energy": energy,
        f"{label}_error": energy - REFERENCE_ENERGY,
        f"{label}_fidelity": fixed_sector_fidelity(full_state, exact),
        f"{label}_entropy": bipartite_entropy_numpy(full_state),
        f"{label}_max_magnetization": max_abs_magnetization_numpy(full_state, NUM_SITES),
        **{
            f"{label}_{key}": value
            for key, value in bond_delocalization_metrics(correlations).items()
        },
    }


def covering_to_text(pairs) -> str:
    return " ".join(f"{i}-{j}" for i, j in pairs)


def main() -> None:
    global REFERENCE_ENERGY
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverings", type=int, default=54)
    parser.add_argument("--covering-mode", choices=["deterministic", "random"], default="deterministic")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--checkpoints",
        default="1,2,8,12,16,24,32,54",
        help="Comma-separated covering counts to report.",
    )
    args = parser.parse_args()

    start = time.time()
    bonds = load_bonds_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    exact = load_sector_exact_result(PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz")
    REFERENCE_ENERGY = exact.energy
    h_sector, basis = fixed_sz_heisenberg_sparse(
        NUM_SITES,
        bonds,
        n_down=N_DOWN,
        pauli_scale=PAULI_SCALE,
    )
    if args.covering_mode == "deterministic":
        all_coverings = enumerate_maximum_dimer_coverings(NUM_SITES, bonds)
        coverings = all_coverings[: args.coverings]
        print(f"deterministic maximum dimer coverings: {len(all_coverings)}")
    else:
        coverings = random_dimer_coverings(
            NUM_SITES,
            bonds,
            count=args.coverings,
            seed=args.seed,
            require_maximal=True,
        )
    if len(coverings) < args.coverings:
        print(f"Warning: generated only {len(coverings)} distinct coverings")

    columns = sector_columns_from_coverings(coverings, basis)
    h_columns = h_sector @ columns
    overlap_all = columns.conj().T @ columns
    h_sub_all = columns.conj().T @ h_columns

    checkpoints = parse_checkpoints(args.checkpoints, len(coverings))
    rows: list[dict] = []
    coefficient_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    best_coeffs = None
    best_state = None
    best_n = None

    for n_coverings in checkpoints:
        v = columns[:, :n_coverings]
        overlap = overlap_all[:n_coverings, :n_coverings]
        h_sub = h_sub_all[:n_coverings, :n_coverings]
        stable_rank, condition_number, overlap_eigs = overlap_condition(overlap)

        equal_coeffs = np.ones(n_coverings, dtype=complex)
        equal_state = normalize_sector_state(v @ equal_coeffs)
        optimized_energy, coeffs, rank = generalized_lowest_state(h_sub, overlap)
        optimized_state = normalize_sector_state(v @ coeffs)

        row = {
            "coverings": n_coverings,
            "stable_rank": rank,
            "overlap_condition": condition_number,
            "overlap_threshold": OVERLAP_THRESHOLD,
            **state_summary("equal", equal_state, basis, h_sector, bonds, exact),
            **state_summary("optimized", optimized_state, basis, h_sector, bonds, exact),
        }
        rows.append(row)
        for eig_index, eig_value in enumerate(overlap_eigs):
            spectrum_rows.append(
                {
                    "coverings": n_coverings,
                    "eigenvalue_index": eig_index,
                    "overlap_eigenvalue": float(eig_value),
                    "kept": bool(eig_value > OVERLAP_THRESHOLD * max(1.0, float(np.max(overlap_eigs)))),
                }
            )
        print(
            "coverings={:3d}: equal E={:.8f}, F={:.6f}; opt E={:.8f}, F={:.6f}".format(
                n_coverings,
                row["equal_energy"],
                row["equal_fidelity"],
                row["optimized_energy"],
                row["optimized_fidelity"],
            )
        )

        if n_coverings == len(coverings):
            best_coeffs = coeffs
            best_state = optimized_state
            best_n = n_coverings
            for index, coeff in enumerate(coeffs):
                coefficient_rows.append(
                    {
                        "covering_index": index,
                        "coefficient_real": float(np.real(coeff)),
                        "coefficient_imag": float(np.imag(coeff)),
                        "coefficient_abs": float(abs(coeff)),
                        "covering": covering_to_text(coverings[index]),
                    }
                )

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    summary_path = output_dir / "19site_rvb_subspace.csv"
    coeff_path = output_dir / f"19site_rvb_subspace_coefficients_n{best_n}.csv"
    covering_path = output_dir / f"19site_rvb_coverings_n{len(coverings)}.csv"
    spectrum_path = output_dir / "19site_rvb_overlap_spectrum.csv"
    state_path = output_dir / f"19site_weighted_rvb_state_n{best_n}.npz"
    metadata_path = summary_path.with_suffix(".metadata.json")

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with coeff_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coefficient_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coefficient_rows)

    with covering_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["covering_index", "covering"])
        writer.writeheader()
        for index, covering in enumerate(coverings):
            writer.writerow({"covering_index": index, "covering": covering_to_text(covering)})

    with spectrum_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spectrum_rows[0].keys()))
        writer.writeheader()
        writer.writerows(spectrum_rows)

    np.savez_compressed(
        state_path,
        sector_state=best_state,
        basis=basis,
        coefficients=best_coeffs,
        energy=rows[-1]["optimized_energy"],
        fidelity=rows[-1]["optimized_fidelity"],
        coverings=best_n,
    )

    metadata = {
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "seed": args.seed,
        "covering_mode": args.covering_mode,
        "coverings_requested": args.coverings,
        "coverings_generated": len(coverings),
        "checkpoints": checkpoints,
        "reference_energy": REFERENCE_ENERGY,
        "pauli_scale": PAULI_SCALE,
        "hamiltonian_convention": "unscaled Pauli: sum_<ij>(XX+YY+ZZ)",
        "singlet_orientation": "|s_ij>=(|0_i 1_j>-|1_i 0_j>)/sqrt(2), with i<j",
        "overlap_threshold": OVERLAP_THRESHOLD,
        "bond_file": "data/19site_bonds.csv",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"elapsed={time.time() - start:.2f}s")
    print(f"Wrote {summary_path}")
    print(f"Wrote {coeff_path}")
    print(f"Wrote {covering_path}")
    print(f"Wrote {spectrum_path}")
    print(f"Wrote {state_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
