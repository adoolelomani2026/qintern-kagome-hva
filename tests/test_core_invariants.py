from __future__ import annotations

import sys
from pathlib import Path
import csv

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kagome_heisenberg_poc import (  # noqa: E402
    apply_heisenberg_gate_cached,
    apply_two_qubit_gate_numpy,
    build_bond_index_cache,
    build_heisenberg_ansatz,
    build_sector_bond_index_cache,
    enumerate_maximum_dimer_coverings,
    exact_ground_state_fixed_sz,
    fixed_sz_heisenberg_sparse,
    heisenberg_energy_numpy,
    heisenberg_gate_matrix,
    heisenberg_hamiltonian,
    heisenberg_sector_energy_cached_numpy,
    heisenberg_sector_state_from_initial_numpy,
    heisenberg_statevector_from_initial_numpy,
    heisenberg_statevector_numpy,
    kagome_patch,
    load_bonds_with_groups_csv,
    load_sector_exact_result,
    sector_state_from_full_state,
    sector_weight,
    validate_bond_groups_are_matchings,
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_final_summary import default_hva_csv  # noqa: E402


def dense_heisenberg_matrix_numpy(
    num_qubits: int,
    bonds: list[tuple[int, int]],
    pauli_scale: float = 1.0,
) -> np.ndarray:
    dim = 2**num_qubits
    matrix = np.zeros((dim, dim), dtype=float)
    for state in range(dim):
        for i, j in bonds:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i == bit_j:
                matrix[state, state] += pauli_scale
            else:
                matrix[state, state] -= pauli_scale
                flipped = state ^ ((1 << i) | (1 << j))
                matrix[flipped, state] += 2.0 * pauli_scale
    return matrix


def test_19site_bond_count_and_edge_coloring() -> None:
    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    assert len(bonds) == 30
    assert set(validate_bond_groups_are_matchings(bonds, groups).values()) == {True}
    assert len(set(groups)) == 4


def test_deterministic_19site_covering_count() -> None:
    bonds, _ = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    coverings = enumerate_maximum_dimer_coverings(19, bonds)
    assert len(coverings) == 54
    assert all(len(covering) == 9 for covering in coverings)


def test_6site_exact_energy_and_factor_of_four_convention() -> None:
    _, bonds = kagome_patch(nx=2, ny=1)
    physical = exact_ground_state_fixed_sz(6, bonds, n_down=3, pauli_scale=0.25)
    unscaled = exact_ground_state_fixed_sz(6, bonds, n_down=3, pauli_scale=1.0)
    assert physical.energy == pytest.approx(-2.25, abs=1e-10)
    assert unscaled.energy == pytest.approx(4.0 * physical.energy, abs=1e-10)


def test_fixed_sector_sparse_matches_dense_hamiltonian_on_small_patch() -> None:
    _, bonds = kagome_patch(nx=2, ny=1)
    sector_h, basis = fixed_sz_heisenberg_sparse(6, bonds, n_down=3, pauli_scale=1.0)
    dense_h = dense_heisenberg_matrix_numpy(6, bonds, pauli_scale=1.0)
    projected = dense_h[np.ix_(basis, basis)]
    np.testing.assert_allclose(sector_h.toarray(), projected, atol=1e-12)


def test_cached_heisenberg_gate_matches_dense_gate() -> None:
    rng = np.random.default_rng(7)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    state = state / np.linalg.norm(state)
    theta = 0.137
    cached = state.copy()
    apply_heisenberg_gate_cached(cached, build_bond_index_cache(3, [(0, 2)])[0], theta)
    dense = apply_two_qubit_gate_numpy(state, heisenberg_gate_matrix(theta), 0, 2)
    np.testing.assert_allclose(cached, dense, atol=1e-12)


def test_qiskit_ansatz_energy_matches_numpy_small_patch() -> None:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    _, bonds = kagome_patch(nx=2, ny=1)
    theta = np.array([0.041])
    circuit, params = build_heisenberg_ansatz(
        6,
        bonds,
        layers=1,
        parameterization="shared",
    )
    state_qiskit = Statevector.from_instruction(
        circuit.assign_parameters(dict(zip(params, theta)), inplace=False)
    )
    energy_qiskit = float(
        np.real(state_qiskit.expectation_value(heisenberg_hamiltonian(6, bonds, pauli_scale=1.0)))
    )
    state_numpy = heisenberg_statevector_numpy(
        6,
        bonds,
        layers=1,
        parameters=theta,
        parameterization="shared",
    )
    energy_numpy = heisenberg_energy_numpy(state_numpy, 6, bonds, pauli_scale=1.0)
    assert energy_qiskit == pytest.approx(energy_numpy, abs=1e-10)


def test_sector_hva_matches_full_hva_and_preserves_sector() -> None:
    bonds, groups = load_bonds_with_groups_csv(PROJECT_ROOT / "data" / "19site_bonds.csv")
    exact = load_sector_exact_result(PROJECT_ROOT / "results" / "19site_fixed_sz_exact_n9.npz")
    data = np.load(PROJECT_ROOT / "results" / "19site_weighted_rvb_state_n54.npz")
    sector_initial = np.asarray(data["sector_state"])
    sector_caches = build_sector_bond_index_cache(exact.basis, bonds)
    values = np.full(8, 0.025)
    sector_state = heisenberg_sector_state_from_initial_numpy(
        sector_initial,
        sector_caches,
        layers=2,
        parameters=values,
        parameterization="grouped",
        bond_groups=groups,
    )
    full_initial = np.zeros(2**19, dtype=complex)
    full_initial[exact.basis] = sector_initial
    full_state = heisenberg_statevector_from_initial_numpy(
        full_initial,
        build_bond_index_cache(19, bonds),
        layers=2,
        parameters=values,
        parameterization="grouped",
        bond_groups=groups,
    )
    assert sector_weight(full_state, exact.basis) == pytest.approx(1.0, abs=1e-10)
    full_energy = heisenberg_energy_numpy(full_state, 19, bonds, pauli_scale=1.0)
    sector_energy = heisenberg_sector_energy_cached_numpy(
        sector_state,
        sector_caches,
        pauli_scale=1.0,
    )
    assert sector_energy == pytest.approx(full_energy, abs=1e-10)
    np.testing.assert_allclose(sector_state, sector_state_from_full_state(full_state, exact.basis), atol=1e-12)


def test_weighted_rvb_state_is_normalized() -> None:
    path = PROJECT_ROOT / "results" / "19site_weighted_rvb_state_n54.npz"
    if not path.exists():
        pytest.skip("weighted RVB state has not been generated")
    data = np.load(path)
    assert np.linalg.norm(np.asarray(data["sector_state"])) == pytest.approx(1.0, abs=1e-12)


def test_final_result_table_uses_selected_hva_energy() -> None:
    final_path = PROJECT_ROOT / "results" / "final_result_table.csv"
    hva_path = default_hva_csv()
    if not final_path.exists() or not hva_path.exists():
        pytest.skip("final summary has not been generated")
    with final_path.open(newline="") as handle:
        final_rows = list(csv.DictReader(handle))
    with hva_path.open(newline="") as handle:
        hva_rows = list(csv.DictReader(handle))
    for hva_row in hva_rows:
        depth = int(hva_row["depth"])
        if depth == 0:
            continue
        final_row = next(row for row in final_rows if row["state"] == f"Weighted RVB + HVA p={depth}")
        assert float(final_row["energy_unscaled_pauli"]) == pytest.approx(float(hva_row["energy"]), abs=1e-8)
    best_final = min(
        (row for row in final_rows if row["state"].startswith("Weighted RVB + HVA p=")),
        key=lambda row: float(row["error_vs_exact"]),
    )
    assert best_final["state"] == "Weighted RVB + HVA p=4"
    assert float(best_final["energy_unscaled_pauli"]) == pytest.approx(-29.0376013518, abs=1e-8)


def test_default_hva_selection_excludes_smoke_files() -> None:
    selected = default_hva_csv()
    assert "smoke" not in selected.name
