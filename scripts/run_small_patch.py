from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    embed_sector_state,
    exact_ground_state_fixed_sz,
    kagome_patch,
    run_depth_sweep_numpy,
)


def main() -> None:
    positions, bonds = kagome_patch(nx=2, ny=1)
    exact = exact_ground_state_fixed_sz(
        len(positions),
        bonds,
        n_down=len(positions) // 2,
        pauli_scale=0.25,
    )
    exact_full = embed_sector_state(exact.state, exact.basis, len(positions))
    results = run_depth_sweep_numpy(
        num_qubits=len(positions),
        bonds=bonds,
        max_layers=2,
        parameterization="bond",
        maxiter=120,
        seed=11,
        pauli_scale=0.25,
        exact_energy=exact.energy,
        exact_state=exact_full,
    )

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "small_patch_depth_sweep.csv"

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layers",
                "energy",
                "exact_energy",
                "error",
                "fidelity",
                "entropy",
                "evaluations",
                "success",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "layers": result.layers,
                    "energy": result.energy,
                    "exact_energy": result.exact_energy,
                    "error": result.error,
                    "fidelity": result.fidelity,
                    "entropy": result.entropy,
                    "evaluations": result.evaluations,
                    "success": result.success,
                }
            )

    print(f"6-site open Kagome patch: {len(positions)} sites, {len(bonds)} bonds")
    for result in results:
        print(
            "p={layers}: E={energy:.8f}, error={error:.8f}, "
            "fidelity={fidelity:.6f}, entropy={entropy:.4f}".format(
                layers=result.layers,
                energy=result.energy,
                error=result.error,
                fidelity=result.fidelity or 0.0,
                entropy=result.entropy,
            )
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
