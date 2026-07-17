from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import load_bonds_csv  # noqa: E402


NUM_SITES = 19


STATE_TO_FIGURE = {
    "static_dimer": "bond_map_static_dimer.png",
    "equal_rvb_54": "bond_map_equal_rvb.png",
    "weighted_rvb_54": "bond_map_weighted_rvb.png",
    "weighted_hva_p1": "bond_map_weighted_hva_p1.png",
    "weighted_hva_p2": "bond_map_weighted_hva_p2.png",
    "weighted_hva_p3": "bond_map_weighted_hva_p3.png",
    "weighted_hva_p4": "bond_map_weighted_hva_p4.png",
    "exact": "bond_map_exact.png",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def force_layout(
    num_sites: int,
    bonds: list[tuple[int, int]],
    iterations: int = 450,
    seed: int = 19,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, num_sites, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    positions += 0.05 * rng.normal(size=positions.shape)
    edges = np.asarray(bonds, dtype=int)
    area = 4.0
    k = math.sqrt(area / num_sites)
    temperature = 0.18
    for step in range(iterations):
        disp = np.zeros_like(positions)
        for i in range(num_sites):
            delta = positions[i] - positions
            dist = np.linalg.norm(delta, axis=1) + 1e-9
            force = (k * k / dist**2)[:, None] * delta
            disp[i] += np.sum(force, axis=0)
        for i, j in edges:
            delta = positions[i] - positions[j]
            dist = np.linalg.norm(delta) + 1e-9
            force = (dist * dist / k) * delta / dist
            disp[i] -= force
            disp[j] += force
        lengths = np.linalg.norm(disp, axis=1)
        scale = np.minimum(lengths, temperature) / (lengths + 1e-9)
        positions += disp * scale[:, None]
        positions -= np.mean(positions, axis=0)
        temperature *= 1.0 - (step + 1) / (iterations + 1)
    return positions


def bond_rows_by_state(path: Path) -> dict[str, dict[tuple[int, int], float]]:
    result: dict[str, dict[tuple[int, int], float]] = {}
    for row in read_csv_rows(path):
        state = row["state"]
        bond = (int(row["i"]), int(row["j"]))
        value = float(row["correlation_unscaled_pauli"])
        result.setdefault(state, {})[bond] = value
    return result


def plot_bond_map(
    positions: np.ndarray,
    bonds: list[tuple[int, int]],
    values_by_bond: dict[tuple[int, int], float],
    title: str,
    output: Path,
    *,
    difference: bool = False,
) -> None:
    values = np.array([values_by_bond[tuple(bond)] for bond in bonds])
    vmax = max(float(np.max(np.abs(values))), 1e-9) if difference else 3.0
    vmin = -vmax if difference else -3.0
    cmap = "bwr" if difference else "coolwarm"

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for bond, value in zip(bonds, values):
        i, j = bond
        x = [positions[i, 0], positions[j, 0]]
        y = [positions[i, 1], positions[j, 1]]
        width = 1.2 + 3.0 * abs(value) / max(vmax, 1e-9)
        ax.plot(x, y, color=plt.get_cmap(cmap)(norm(value)), linewidth=width, solid_capstyle="round")
    ax.scatter(positions[:, 0], positions[:, 1], s=120, color="white", edgecolor="black", zorder=3)
    for site, (x, y) in enumerate(positions):
        ax.text(x, y, str(site), ha="center", va="center", fontsize=7, zorder=4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.82)
    cbar.set_label("C_ij difference" if difference else "C_ij = <XX+YY+ZZ>")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def depth_rows(final_rows: list[dict[str, str]]) -> list[dict[str, float]]:
    selected = []
    for row in final_rows:
        if row["state"] == "Weighted RVB-54":
            depth = 0
        elif row["state"].startswith("Weighted RVB + HVA p="):
            depth = int(row["depth"])
        else:
            continue
        selected.append(
            {
                "depth": depth,
                "energy": float(row["energy_unscaled_pauli"]),
                "error": float(row["error_vs_exact"]),
                "fidelity": float(row["fidelity"]),
                "magnetization": float(row["max_magnetization"]),
                "entropy": float(row["entropy_midcut"]),
            }
        )
    return sorted(selected, key=lambda item: item["depth"])


def plot_depth_metric(rows: list[dict[str, float]], key: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([row["depth"] for row in rows], [row[key] for row in rows], marker="o", linewidth=2)
    ax.set_xlabel("HVA depth p")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_spin_distance_profile(profile_path: Path, output: Path) -> None:
    if not profile_path.exists():
        return
    rows = read_csv_rows(profile_path)
    labels = {
        "static_dimer": "Static dimer",
        "weighted_rvb_54": "Weighted RVB",
        "weighted_hva_p4": "Weighted HVA p=4",
        "exact": "Exact",
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for state, label in labels.items():
        state_rows = [
            row
            for row in rows
            if row["state"] == state and float(row["mean_abs_correlation"]) > 0.0
        ]
        if not state_rows:
            continue
        state_rows = sorted(state_rows, key=lambda row: int(row["graph_distance"]))
        ax.plot(
            [int(row["graph_distance"]) for row in state_rows],
            [float(row["mean_abs_correlation"]) for row in state_rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Graph distance")
    ax.set_ylabel("Mean |<XX+YY+ZZ>|")
    ax.set_title("Finite-patch spin-correlation decay proxy")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_one_page_summary(
    final_rows: list[dict[str, str]],
    depth_data: list[dict[str, float]],
    positions: np.ndarray,
    bonds: list[tuple[int, int]],
    state_bonds: dict[str, dict[tuple[int, int], float]],
    output: Path,
) -> None:
    rows_by_label = {row["state"]: row for row in final_rows}
    hva_rows = [row for row in final_rows if row["state"].startswith("Weighted RVB + HVA p=")]
    best_hva = min(hva_rows, key=lambda row: abs(float(row["error_vs_exact"])))
    best_depth = int(best_hva["depth"])
    best_state_key = f"weighted_hva_p{best_depth}"
    labels = ["Static dimer", "Weighted RVB-54", best_hva["state"], "Exact"]
    short_labels = ["Dimer", "Weighted RVB", f"HVA p={best_depth}", "Exact"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))

    errors = [abs(float(rows_by_label[label]["error_vs_exact"])) for label in labels]
    axes[0, 0].bar(short_labels, errors)
    axes[0, 0].set_ylabel("Error vs exact")
    axes[0, 0].set_title("19-site Kagome Heisenberg benchmark")

    axes[0, 1].bar(short_labels, [float(rows_by_label[label]["fidelity"]) for label in labels])
    axes[0, 1].set_ylabel("Fidelity")
    axes[0, 1].set_ylim(0.0, 1.05)

    axes[1, 0].plot(
        [row["depth"] for row in depth_data],
        [row["energy"] for row in depth_data],
        marker="o",
        linewidth=2,
    )
    axes[1, 0].set_xlabel("HVA depth p")
    axes[1, 0].set_ylabel("Energy, unscaled Pauli")
    axes[1, 0].grid(alpha=0.25)

    ax = axes[1, 1]
    diff_values = {
        bond: state_bonds[best_state_key][bond] - state_bonds["exact"][bond]
        for bond in bonds
    }
    values = np.array([diff_values[bond] for bond in bonds])
    vmax = max(float(np.max(np.abs(values))), 1e-9)
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    for bond in bonds:
        i, j = bond
        value = diff_values[bond]
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color=plt.get_cmap("bwr")(norm(value)),
            linewidth=1.2 + 3.0 * abs(value) / vmax,
            solid_capstyle="round",
        )
    ax.scatter(positions[:, 0], positions[:, 1], s=80, color="white", edgecolor="black", zorder=3)
    ax.set_title(f"Bond-correlation error: HVA p={best_depth} vs exact")
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="bwr", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68)
    cbar.set_label("Delta C_ij")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_calibration_scatter(final_rows: list[dict[str, str]], output: Path, *, zoom: bool = False) -> None:
    scan_path = PROJECT_ROOT / "results" / "19site_calibration_scan.csv"
    if not scan_path.exists():
        return
    scan_rows = read_csv_rows(scan_path)
    calibration = [
        row
        for row in scan_rows
        if abs(float(row.get("jprime", 1.0)) - 1.0) > 1e-12
        and row.get("target_energy_error", "")
        and row.get("target_fidelity", "")
    ]
    if not calibration:
        return
    hva_rows = [row for row in final_rows if row["state"].startswith("Weighted RVB + HVA p=")]
    if not hva_rows:
        return
    best_hva = min(hva_rows, key=lambda row: abs(float(row["error_vs_exact"])))
    best_hva_depth = int(best_hva["depth"])
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    styles = {
        "group": ("o", "#4c78a8"),
        "bond": ("s", "#59a14f"),
        "triangle": ("^", "#f28e2b"),
    }
    for mode in sorted({row.get("scan_mode", "scan") for row in calibration}):
        marker, color = styles.get(mode, ("o", "#4c78a8"))
        mode_rows = [row for row in calibration if row.get("scan_mode", "scan") == mode]
        ax.scatter(
            [abs(float(row["target_energy_error"])) for row in mode_rows],
            [float(row["target_fidelity"]) for row in mode_rows],
            marker=marker,
            label=f"{mode} calibration",
            color=color,
            alpha=0.75,
            edgecolors="none",
        )
    ax.scatter(
        [abs(float(best_hva["error_vs_exact"]))],
        [float(best_hva["fidelity"])],
        marker="*",
        s=180,
        color="#d62728",
        label=f"Weighted RVB + HVA p={best_hva_depth}",
        zorder=4,
    )
    best = min(calibration, key=lambda row: abs(float(row["target_energy_error"])))
    best_x = abs(float(best["target_energy_error"]))
    best_y = float(best["target_fidelity"])
    ax.scatter([best_x], [best_y], marker="o", s=90, facecolors="none", edgecolors="black", linewidths=1.2)
    ax.annotate(
        f"{best.get('scan_label', 'best')}, J'={float(best['jprime']):.2f}",
        xy=(best_x, best_y),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=7,
    )
    if zoom:
        ax.set_xlim(0.0, 0.15)
        ax.set_ylim(0.95, 1.01)
    else:
        ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Target energy error")
    ax.set_ylabel("Fidelity to exact")
    ax.set_title(
        "Calibration scan vs no-calibration HVA"
        if not zoom
        else "Calibration scan vs no-calibration HVA, zoom"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_workflow_flowchart(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.1))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.35)
    ax.axis("off")

    def add_box(
        x: float,
        y: float,
        title: str,
        detail: str,
        *,
        width: float,
        height: float = 0.72,
        dashed: bool = False,
    ) -> tuple[float, float, float, float]:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.025",
            linewidth=1.05,
            linestyle="--" if dashed else "-",
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + height * 0.63,
            title,
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="black",
        )
        ax.text(
            x + width / 2,
            y + height * 0.28,
            detail,
            ha="center",
            va="center",
            fontsize=7.4,
            color="black",
            linespacing=1.08,
        )
        return (x, y, width, height)

    def right(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, h = box
        return (x + w, y + h / 2)

    def left(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, _, h = box
        return (x, y + h / 2)

    def top(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, h = box
        return (x + w / 2, y + h)

    def bottom(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, _ = box
        return (x + w / 2, y)

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = "black",
        dashed: bool = False,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color=color,
                linestyle="--" if dashed else "-",
                shrinkA=5,
                shrinkB=5,
            )
        )

    hamiltonian = add_box(0.35, 3.04, "19-site Kagome Hamiltonian", r"$H=\sum_{\langle i,j\rangle}(XX+YY+ZZ)$", width=2.45)

    diagonalize = add_box(3.35, 3.04, "Sparse diagonalization", r"$E_0,\ |\psi_0\rangle$ in fixed $S^z$", width=2.35)
    metrics = add_box(9.65, 1.35, "Compare with exact", r"$E,\ F,\ C_{ij}$", width=1.55)
    arrow(right(hamiltonian), left(diagonalize))
    arrow(right(diagonalize), top(metrics))

    coverings = add_box(0.35, 1.35, "Dimer-covering subspace", "54 maximum coverings", width=2.25)
    rvb = add_box(3.00, 1.35, "Weighted RVB", "signed amplitudes from\nclassical subspace solve", width=2.10)
    hva = add_box(5.55, 1.35, "Heisenberg-HVA", "edge-colored layers\n$p=1,\\ldots,4$", width=2.00)
    trial = add_box(7.90, 1.35, "Trial state", r"$|\psi_p\rangle$", width=1.30)
    arrow(bottom(hamiltonian), top(coverings))
    for first, second in [(coverings, rvb), (rvb, hva), (hva, trial), (trial, metrics)]:
        arrow(right(first), left(second))

    calibration = add_box(
        5.55,
        0.18,
        "Calibration scans",
        "separate reference;\nfuture local block clues",
        width=2.00,
        dashed=True,
    )
    arrow(top(calibration), bottom(hva), dashed=True)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_ansatz_circuit_schematic(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.7))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 3.7)
    ax.axis("off")

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color="black",
                shrinkA=3,
                shrinkB=3,
            )
        )

    def block(x: float, y: float, w: float, h: float, title: str, subtitle: str = "") -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.62 * h, title, ha="center", va="center", fontsize=8.6, fontweight="bold")
        if subtitle:
            ax.text(x + w / 2, y + 0.28 * h, subtitle, ha="center", va="center", fontsize=7.2)

    ax.text(0.20, 3.30, "Edge-colored Heisenberg-HVA ansatz circuit", fontsize=10.0, fontweight="bold")
    ax.text(
        0.20,
        3.03,
        r"$|\psi_p(\theta)\rangle=\prod_{\ell=1}^{p}\prod_{c=0}^{3}G_c(\theta_{\ell,c})|\psi_{\rm RVB}\rangle$",
        fontsize=8.4,
    )

    y = 2.15
    ax.text(0.25, y + 0.25, r"$|\psi_{\rm RVB}\rangle$", fontsize=9.0, ha="center", va="center")
    x_positions = [1.00, 2.32, 3.64, 4.96]
    labels = ["color 0", "color 1", "color 2", "color 3"]
    for x, label in zip(x_positions, labels):
        block(x, y, 0.90, 0.55, rf"$G_{label[-1]}$", rf"$\theta_{{\ell,{label[-1]}}}$")
    ax.text(6.10, y + 0.27, r"$\cdots$", fontsize=13, ha="center", va="center")
    block(6.75, y, 0.90, 0.55, r"$G_0$", r"$\theta_{p,0}$")
    block(8.02, y, 0.90, 0.55, r"$G_3$", r"$\theta_{p,3}$")
    ax.text(9.95, y + 0.25, r"$|\psi_p\rangle$", fontsize=9.0, ha="center", va="center")

    arrow(0.47, y + 0.27, 0.98, y + 0.27)
    for x0, x1 in [(1.90, 2.32), (3.22, 3.64), (4.54, 4.96), (5.86, 6.75), (7.65, 8.02), (8.92, 9.62)]:
        arrow(x0, y + 0.27, x1, y + 0.27)

    ax.text(0.20, 1.30, "One color block", fontsize=9.0, fontweight="bold")
    ax.text(
        1.65,
        1.30,
        r"$G_c(\theta)=\prod_{(i,j)\in E_c}\exp[-i\theta(XX+YY+ZZ)]$",
        fontsize=8.2,
    )

    qi_y, qj_y = 0.78, 0.36
    ax.text(0.45, qi_y, r"$q_i$", fontsize=8.5, ha="right", va="center")
    ax.text(0.45, qj_y, r"$q_j$", fontsize=8.5, ha="right", va="center")
    ax.plot([0.55, 9.75], [qi_y, qi_y], color="black", linewidth=0.9)
    ax.plot([0.55, 9.75], [qj_y, qj_y], color="black", linewidth=0.9)
    gate_xs = [1.15, 2.25, 3.35]
    gate_labels = [r"$R_{XX}(2\theta)$", r"$R_{YY}(2\theta)$", r"$R_{ZZ}(2\theta)$"]
    for gx, glabel in zip(gate_xs, gate_labels):
        rect = FancyBboxPatch(
            (gx, qj_y - 0.16),
            0.86,
            qi_y - qj_y + 0.32,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            linewidth=1.0,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(gx + 0.43, (qi_y + qj_y) / 2, glabel, ha="center", va="center", fontsize=7.4)
    ax.text(
        4.65,
        0.56,
        "applied to every disjoint bond in the color group; different colors are applied sequentially",
        fontsize=7.8,
        va="center",
    )

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    bonds = load_bonds_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    positions = force_layout(NUM_SITES, bonds)
    state_bonds = bond_rows_by_state(results_dir / "19site_bond_correlations_by_state.csv")

    for state, filename in STATE_TO_FIGURE.items():
        if state in state_bonds:
            title = state.replace("_", " ")
            plot_bond_map(positions, bonds, state_bonds[state], title, figures_dir / filename)
    if "weighted_hva_p2" in state_bonds and "exact" in state_bonds:
        diff = {bond: state_bonds["weighted_hva_p2"][bond] - state_bonds["exact"][bond] for bond in bonds}
        plot_bond_map(
            positions,
            bonds,
            diff,
            "weighted HVA p=2 minus exact",
            figures_dir / "bond_map_error_p2_vs_exact.png",
            difference=True,
        )

    final_rows = read_csv_rows(results_dir / "final_result_table.csv")
    hva_rows = [row for row in final_rows if row["state"].startswith("Weighted RVB + HVA p=")]
    if hva_rows:
        best_hva = min(hva_rows, key=lambda row: abs(float(row["error_vs_exact"])))
        best_key = f"weighted_hva_p{int(best_hva['depth'])}"
        if best_key in state_bonds and "exact" in state_bonds:
            diff = {bond: state_bonds[best_key][bond] - state_bonds["exact"][bond] for bond in bonds}
            plot_bond_map(
                positions,
                bonds,
                diff,
                f"weighted HVA p={best_hva['depth']} minus exact",
                figures_dir / "bond_map_error_best_hva_vs_exact.png",
                difference=True,
            )
    depth_data = depth_rows(final_rows)
    plot_depth_metric(depth_data, "energy", "Energy, unscaled Pauli", figures_dir / "energy_vs_hva_depth.png")
    plot_depth_metric(depth_data, "error", "Error vs exact", figures_dir / "error_vs_hva_depth.png")
    plot_depth_metric(depth_data, "fidelity", "Fidelity", figures_dir / "fidelity_vs_hva_depth.png")
    plot_depth_metric(depth_data, "magnetization", "Max |<Sz>|", figures_dir / "magnetization_vs_hva_depth.png")
    plot_depth_metric(depth_data, "entropy", "Midcut entropy", figures_dir / "entropy_vs_hva_depth.png")
    plot_spin_distance_profile(
        results_dir / "19site_spin_distance_profile.csv",
        figures_dir / "spin_distance_profile.png",
    )
    plot_one_page_summary(
        final_rows,
        depth_data,
        positions,
        bonds,
        state_bonds,
        figures_dir / "one_page_summary.png",
    )
    plot_calibration_scatter(final_rows, figures_dir / "calibration_energy_vs_fidelity.png")
    plot_calibration_scatter(final_rows, figures_dir / "calibration_energy_vs_fidelity_zoom.png", zoom=True)
    plot_workflow_flowchart(figures_dir / "workflow_flowchart.png")
    plot_ansatz_circuit_schematic(figures_dir / "ansatz_circuit_schematic.png")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
