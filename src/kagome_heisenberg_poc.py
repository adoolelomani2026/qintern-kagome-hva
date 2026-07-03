"""Utilities for Kagome Heisenberg proof-of-concept VQE experiments.

The code in this module is intentionally small-patch friendly.  It uses
statevector simulation and exact diagonalization for validation, then exposes
the same Hamiltonian/ansatz builders for a future 19-site benchmark once the
paper-specific bond list is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import minimize, minimize_scalar


Bond = tuple[int, int]
Parameterization = Literal["shared", "bond", "grouped"]
InitialState = Literal["singlet_pairs", "neel", "zero"]

DEPENBROCK_DMRG_PHYSICAL_ENERGY_PER_SITE = -0.4386
DEPENBROCK_DMRG_PHYSICAL_ENERGY_PER_SITE_UNCERTAINTY = 0.0005
DEPENBROCK_DMRG_TRIPLET_GAP = 0.13
DEPENBROCK_DMRG_TRIPLET_GAP_UNCERTAINTY = 0.01
AHSAN_19SITE_UNSCALED_ENERGY_ROUNDED = -29.14
AHSAN_DEFECT_JPRIME_CENTER = 1.95
AHSAN_DEFECT_JPRIME_GRID = (1.80, 1.90, 1.95, 2.00, 2.10)

LITERATURE_REFERENCES = {
    "depenbrock2012": {
        "citation": "Depenbrock, McCulloch, and Schollwoeck, PRL 109, 067201 (2012)",
        "arxiv": "1205.4858",
    },
    "ahsan2025": {
        "citation": "Ahsan, arXiv:2507.06361v3 (2025)",
        "arxiv": "2507.06361v3",
    },
}


@dataclass(frozen=True)
class VQEResult:
    """Summary for one ansatz depth."""

    layers: int
    energy: float
    exact_energy: float
    error: float
    fidelity: float | None
    entropy: float
    parameters: np.ndarray
    success: bool
    evaluations: int
    message: str


@dataclass(frozen=True)
class BondIndexCache:
    """Basis-index partitions for one two-qubit bond."""

    bond: Bond
    idx00: np.ndarray
    idx01: np.ndarray
    idx10: np.ndarray
    idx11: np.ndarray


@dataclass(frozen=True)
class SectorExactResult:
    """Exact diagonalization result in a fixed magnetization sector."""

    energy: float
    state: np.ndarray
    basis: np.ndarray
    n_down: int


def kagome_patch(nx: int = 2, ny: int = 1) -> tuple[np.ndarray, list[Bond]]:
    """Generate an open-boundary Kagome patch from unit cells.

    Each unit cell contains three sites.  Bonds are detected geometrically using
    the nearest-neighbor distance in the standard Kagome embedding.  For example,
    ``nx=2, ny=1`` gives a 6-site debugging patch, while ``nx=2, ny=2`` gives
    a 12-site patch.
    """

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")

    sqrt3 = np.sqrt(3.0)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, sqrt3 / 2.0])
    basis = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.25, sqrt3 / 4.0],
        ]
    )

    positions: list[np.ndarray] = []
    for iy in range(ny):
        for ix in range(nx):
            origin = ix * a1 + iy * a2
            positions.extend(origin + b for b in basis)

    coords = np.array(positions)
    bonds: list[Bond] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if np.isclose(np.linalg.norm(coords[i] - coords[j]), 0.5, atol=1e-9):
                bonds.append((i, j))
    return coords, bonds


def load_bonds_csv(path: str | Path) -> list[Bond]:
    """Load a two-column ``i,j`` bond CSV with optional comments/header."""

    bonds, _ = load_bonds_with_groups_csv(path)
    return bonds


def load_bonds_with_groups_csv(path: str | Path) -> tuple[list[Bond], list[str]]:
    """Load ``i,j[,group]`` bond CSV data with optional comments/header."""

    bonds: list[Bond] = []
    groups: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("i,"):
            continue
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) < 2:
            raise ValueError(f"Expected at least two comma-separated columns: {line}")
        bonds.append((int(pieces[0]), int(pieces[1])))
        groups.append(pieces[2] if len(pieces) > 2 and pieces[2] else "default")
    return bonds, groups


def ordered_group_names(bond_groups: list[str] | None, num_bonds: int) -> list[str]:
    """Return stable group names in first-seen order."""

    if bond_groups is None:
        bond_groups = ["default"] * num_bonds
    if len(bond_groups) != num_bonds:
        raise ValueError("bond_groups length must match bonds length")
    return list(dict.fromkeys(bond_groups))


def parameter_count(
    layers: int,
    bonds: list[Bond],
    parameterization: Parameterization,
    bond_groups: list[str] | None = None,
) -> int:
    """Number of variational parameters for an ansatz setting."""

    if parameterization == "shared":
        return layers
    if parameterization == "bond":
        return layers * len(bonds)
    if parameterization == "grouped":
        return layers * len(ordered_group_names(bond_groups, len(bonds)))
    raise ValueError("parameterization must be 'shared', 'bond', or 'grouped'")


def validate_bond_groups_are_matchings(
    bonds: list[Bond],
    bond_groups: list[str],
) -> dict[str, bool]:
    """Check whether each group is a disjoint matching."""

    if len(bonds) != len(bond_groups):
        raise ValueError("bond_groups length must match bonds length")
    status: dict[str, bool] = {}
    for group in ordered_group_names(bond_groups, len(bonds)):
        used: set[int] = set()
        ok = True
        for i, j in [bond for bond, active_group in zip(bonds, bond_groups) if active_group == group]:
            if i in used or j in used:
                ok = False
            used.update((i, j))
        status[group] = ok
    return status


def pauli_label(num_qubits: int, i: int, j: int, pauli: str) -> str:
    """Build a Qiskit Pauli label acting with ``pauli`` on qubits i and j."""

    label = ["I"] * num_qubits
    label[num_qubits - 1 - i] = pauli
    label[num_qubits - 1 - j] = pauli
    return "".join(label)


def heisenberg_hamiltonian(
    num_qubits: int,
    bonds: Iterable[Bond],
    j_coupling: float = 1.0,
    pauli_scale: float = 0.25,
) -> object:
    """Build H = J * pauli_scale * sum_<ij> (XX + YY + ZZ).

    Use ``pauli_scale=0.25`` for the physical spin-1/2 convention
    S_i dot S_j = (XX + YY + ZZ) / 4.  Use ``pauli_scale=1.0`` when comparing
    against papers or code that report energies in the unscaled Pauli convention.
    """

    return heisenberg_hamiltonian_weighted(
        num_qubits,
        bonds,
        bond_weights=None,
        j_coupling=j_coupling,
        pauli_scale=pauli_scale,
    )


def heisenberg_hamiltonian_weighted(
    num_qubits: int,
    bonds: Iterable[Bond],
    bond_weights: Iterable[float] | None = None,
    j_coupling: float = 1.0,
    pauli_scale: float = 0.25,
) -> object:
    """Build a weighted Qiskit SparsePauliOp Heisenberg Hamiltonian."""

    from qiskit.quantum_info import SparsePauliOp

    bond_list = list(bonds)
    weights = [1.0] * len(bond_list) if bond_weights is None else list(bond_weights)
    if len(weights) != len(bond_list):
        raise ValueError("bond_weights length must match bonds length")

    terms: list[tuple[str, complex]] = []
    for (i, j), weight in zip(bond_list, weights):
        coeff = j_coupling * pauli_scale * float(weight)
        for pauli in ("X", "Y", "Z"):
            terms.append((pauli_label(num_qubits, i, j, pauli), coeff))
    return SparsePauliOp.from_list(terms).simplify()


def matching_from_bonds(bonds: Iterable[Bond]) -> list[Bond]:
    """Greedy disjoint bond matching for singlet-product initial states."""

    used: set[int] = set()
    pairs: list[Bond] = []
    for i, j in sorted(tuple(sorted(bond)) for bond in bonds):
        if i not in used and j not in used:
            pairs.append((i, j))
            used.update((i, j))
    return pairs


def random_dimer_coverings(
    num_qubits: int,
    bonds: Iterable[Bond],
    count: int = 12,
    seed: int = 123,
    require_maximal: bool = True,
) -> list[list[Bond]]:
    """Generate distinct random maximum dimer coverings from graph bonds.

    For odd ``num_qubits`` this targets ``floor(num_qubits / 2)`` singlet pairs,
    leaving one unpaired spin-up site.  The deterministic greedy covering is
    included first when it has the target size.
    """

    target_pairs = num_qubits // 2
    bond_list = [tuple(sorted(bond)) for bond in bonds]
    rng = np.random.default_rng(seed)
    coverings: list[list[Bond]] = []
    seen: set[tuple[Bond, ...]] = set()

    def add_covering(pairs: list[Bond]) -> None:
        key = tuple(sorted(tuple(sorted(pair)) for pair in pairs))
        if key not in seen and (not require_maximal or len(key) == target_pairs):
            coverings.append(list(key))
            seen.add(key)

    add_covering(matching_from_bonds(bond_list))

    attempts = 0
    while len(coverings) < count and attempts < 4000:
        attempts += 1
        shuffled = list(bond_list)
        rng.shuffle(shuffled)
        used: set[int] = set()
        pairs: list[Bond] = []
        for i, j in shuffled:
            if i not in used and j not in used:
                pairs.append((i, j))
                used.update((i, j))
                if len(pairs) == target_pairs:
                    break
        add_covering(pairs)
    return coverings


def enumerate_maximum_dimer_coverings(
    num_qubits: int,
    bonds: Iterable[Bond],
) -> list[list[Bond]]:
    """Deterministically enumerate all maximum dimer coverings.

    Bonds and pairs are oriented with ``i < j``.  For odd systems such as the
    19-site patch, this enumerates all matchings with ``floor(N/2)`` dimers.
    """

    target_pairs = num_qubits // 2
    adjacency: dict[int, list[int]] = {site: [] for site in range(num_qubits)}
    for i, j in sorted(tuple(sorted(bond)) for bond in bonds):
        adjacency[i].append(j)
        adjacency[j].append(i)
    for site in adjacency:
        adjacency[site] = sorted(set(adjacency[site]))

    coverings: list[list[Bond]] = []

    def backtrack(unused: set[int], pairs: list[Bond]) -> None:
        if len(pairs) == target_pairs:
            coverings.append(list(pairs))
            return
        if len(pairs) + len(unused) // 2 < target_pairs:
            return

        site = min(unused)
        remaining_without_site = set(unused)
        remaining_without_site.remove(site)

        # Leave this site unmatched only if enough sites remain to finish a
        # maximum matching. For odd N this accounts for the one leftover spin.
        if len(pairs) + len(remaining_without_site) // 2 >= target_pairs:
            backtrack(remaining_without_site, pairs)

        for neighbor in adjacency[site]:
            if neighbor not in remaining_without_site:
                continue
            next_unused = set(remaining_without_site)
            next_unused.remove(neighbor)
            pairs.append(tuple(sorted((site, neighbor))))
            backtrack(next_unused, pairs)
            pairs.pop()

    backtrack(set(range(num_qubits)), [])
    unique: dict[tuple[Bond, ...], list[Bond]] = {}
    for covering in coverings:
        key = tuple(sorted(covering))
        unique[key] = list(key)
    return [unique[key] for key in sorted(unique)]


def singlet_product_state_from_pairs(
    num_qubits: int,
    pairs: Iterable[Bond],
    unpaired_down: Iterable[int] | None = None,
) -> np.ndarray:
    """Build a sparse singlet-product state from explicit dimer pairs.

    Each pair is prepared as (|0_i 1_j> - |1_i 0_j>) / sqrt(2).  Unpaired sites
    default to spin-up (bit 0), which keeps a 19-site state with 9 singlets in
    the n_down=9 sector.
    """

    pair_list = [tuple(pair) for pair in pairs]
    used = {qubit for pair in pair_list for qubit in pair}
    if len(used) != 2 * len(pair_list):
        raise ValueError("Pairs must be disjoint")
    if any(qubit < 0 or qubit >= num_qubits for qubit in used):
        raise ValueError("Pair contains an out-of-range qubit")

    base_state = 0
    for qubit in unpaired_down or []:
        if qubit in used:
            raise ValueError("Unpaired-down qubit cannot also be in a dimer")
        base_state |= 1 << qubit

    state = np.zeros(2**num_qubits, dtype=complex)
    norm = 1.0 / np.sqrt(2.0 ** len(pair_list))
    for assignment in range(2 ** len(pair_list)):
        basis_state = base_state
        sign = 1.0
        for pair_index, (i, j) in enumerate(pair_list):
            if (assignment >> pair_index) & 1:
                basis_state |= 1 << i
                sign *= -1.0
            else:
                basis_state |= 1 << j
        state[basis_state] += sign * norm
    return state


def rvb_state_from_coverings(
    num_qubits: int,
    coverings: Iterable[Iterable[Bond]],
) -> np.ndarray:
    """Equal-weight normalized superposition of singlet dimer coverings."""

    state = np.zeros(2**num_qubits, dtype=complex)
    used_count = 0
    for pairs in coverings:
        state += singlet_product_state_from_pairs(num_qubits, pairs)
        used_count += 1
    if used_count == 0:
        raise ValueError("At least one dimer covering is required")
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("RVB covering superposition has zero norm")
    return state / norm


def prepare_initial_state(
    circuit,
    bonds: Iterable[Bond],
    mode: InitialState = "singlet_pairs",
) -> None:
    """Prepare a simple physics-motivated initial state in-place."""

    if mode == "zero":
        return
    if mode == "neel":
        for qubit in range(1, circuit.num_qubits, 2):
            circuit.x(qubit)
        return
    if mode != "singlet_pairs":
        raise ValueError(f"Unknown initial state: {mode}")

    pairs = matching_from_bonds(bonds)
    paired = {qubit for pair in pairs for qubit in pair}
    leftovers = [qubit for qubit in range(circuit.num_qubits) if qubit not in paired]
    pairs.extend(
        (leftovers[index], leftovers[index + 1])
        for index in range(0, len(leftovers) - 1, 2)
    )

    for i, j in pairs:
        circuit.x(j)
        circuit.h(i)
        circuit.cx(i, j)
        circuit.z(i)


def build_heisenberg_ansatz(
    num_qubits: int,
    bonds: list[Bond],
    layers: int,
    parameterization: Parameterization = "shared",
    initial_state: InitialState = "singlet_pairs",
    bond_groups: list[str] | None = None,
    ry_layer: bool = False,
) -> tuple[object, list[object]]:
    """Build a layered exp[-i theta (XX + YY + ZZ)] ansatz.

    Qiskit's ``rxx(phi)``, ``ryy(phi)``, and ``rzz(phi)`` implement
    exp[-i phi/2 PP], so each interaction uses ``phi = 2 * theta``.
    """

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    if layers < 0:
        raise ValueError("layers must be non-negative")
    if parameterization not in {"shared", "bond", "grouped"}:
        raise ValueError("parameterization must be 'shared', 'bond', or 'grouped'")

    circuit = QuantumCircuit(num_qubits)
    prepare_initial_state(circuit, bonds, initial_state)
    group_names = ordered_group_names(bond_groups, len(bonds))

    parameters: list[Parameter] = []
    if ry_layer:
        for qubit in range(num_qubits):
            alpha = Parameter(f"alpha_{qubit}")
            parameters.append(alpha)
            circuit.ry(alpha, qubit)

    for layer in range(layers):
        if parameterization == "shared":
            theta = Parameter(f"theta_{layer}")
            parameters.append(theta)
            layer_parameters = [theta] * len(bonds)
        elif parameterization == "bond":
            layer_parameters = [
                Parameter(f"theta_{layer}_{bond_index}")
                for bond_index in range(len(bonds))
            ]
            parameters.extend(layer_parameters)
        else:
            group_parameters = {
                group: Parameter(f"theta_{layer}_{group}")
                for group in group_names
            }
            parameters.extend(group_parameters[group] for group in group_names)
            active_groups = bond_groups or ["default"] * len(bonds)
            layer_parameters = [group_parameters[group] for group in active_groups]

        for (i, j), theta_ij in zip(bonds, layer_parameters):
            circuit.rxx(2.0 * theta_ij, i, j)
            circuit.ryy(2.0 * theta_ij, i, j)
            circuit.rzz(2.0 * theta_ij, i, j)

    return circuit, parameters


def statevector_from_params(
    circuit,
    parameters: list[object],
    values: np.ndarray,
) -> object:
    """Bind ansatz parameters and return its statevector."""

    from qiskit.quantum_info import Statevector

    if len(parameters) != len(values):
        raise ValueError("Parameter vector length does not match circuit parameters")
    bound = circuit.assign_parameters(dict(zip(parameters, values)), inplace=False)
    return Statevector.from_instruction(bound)


def expectation_value(state, operator) -> float:
    """Return Re(<state|operator|state>)."""

    return float(np.real(state.expectation_value(operator)))


def exact_ground_state(operator) -> tuple[float, np.ndarray]:
    """Dense exact diagonalization for small patches."""

    matrix = operator.to_matrix(sparse=False)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    index = int(np.argmin(eigvals))
    return float(np.real(eigvals[index])), np.asarray(eigvecs[:, index])


def fixed_magnetization_basis(num_qubits: int, n_down: int) -> np.ndarray:
    """Return computational basis states with exactly ``n_down`` one bits."""

    from itertools import combinations
    from math import comb

    if n_down < 0 or n_down > num_qubits:
        raise ValueError("n_down must be between 0 and num_qubits")

    states = np.empty(comb(num_qubits, n_down), dtype=np.int64)
    for index, combo in enumerate(combinations(range(num_qubits), n_down)):
        state = 0
        for qubit in combo:
            state |= 1 << qubit
        states[index] = state
    return states


def fixed_sz_heisenberg_sparse(
    num_qubits: int,
    bonds: Iterable[Bond],
    n_down: int,
    pauli_scale: float = 1.0,
    bond_weights: Iterable[float] | None = None,
):
    """Build H in a fixed-n_down sector as a SciPy CSR sparse matrix.

    The Hamiltonian convention is
    ``pauli_scale * sum_bonds weight_ij * (XX + YY + ZZ)``.
    """

    from scipy.sparse import coo_matrix

    basis = fixed_magnetization_basis(num_qubits, n_down)
    index_by_state = {int(state): index for index, state in enumerate(basis)}
    bond_list = list(bonds)
    weights = [1.0] * len(bond_list) if bond_weights is None else list(bond_weights)
    if len(weights) != len(bond_list):
        raise ValueError("bond_weights length must match bonds length")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diagonal = np.zeros(len(basis), dtype=float)

    for col, state in enumerate(basis):
        state_int = int(state)
        for (i, j), weight in zip(bond_list, weights):
            coeff = pauli_scale * float(weight)
            bit_i = (state_int >> i) & 1
            bit_j = (state_int >> j) & 1
            if bit_i == bit_j:
                diagonal[col] += coeff
            else:
                diagonal[col] -= coeff
                flipped = state_int ^ ((1 << i) | (1 << j))
                rows.append(index_by_state[flipped])
                cols.append(col)
                data.append(2.0 * coeff)

    rows.extend(range(len(basis)))
    cols.extend(range(len(basis)))
    data.extend(diagonal.tolist())

    matrix = coo_matrix((data, (rows, cols)), shape=(len(basis), len(basis))).tocsr()
    return matrix, basis


def exact_ground_state_fixed_sz(
    num_qubits: int,
    bonds: Iterable[Bond],
    n_down: int,
    pauli_scale: float = 1.0,
    bond_weights: Iterable[float] | None = None,
    tol: float = 1e-10,
    maxiter: int | None = None,
) -> SectorExactResult:
    """Sparse exact diagonalization in one fixed magnetization sector."""

    from scipy.sparse.linalg import eigsh

    matrix, basis = fixed_sz_heisenberg_sparse(
        num_qubits,
        bonds,
        n_down,
        pauli_scale=pauli_scale,
        bond_weights=bond_weights,
    )
    eigvals, eigvecs = eigsh(matrix, k=1, which="SA", tol=tol, maxiter=maxiter)
    state = eigvecs[:, 0]
    state = state / np.linalg.norm(state)
    return SectorExactResult(
        energy=float(eigvals[0]),
        state=np.asarray(state),
        basis=basis,
        n_down=n_down,
    )


def save_sector_exact_result(path: str | Path, result: SectorExactResult) -> None:
    """Save a fixed-sector exact result to ``.npz``."""

    np.savez_compressed(
        path,
        energy=result.energy,
        state=result.state,
        basis=result.basis,
        n_down=result.n_down,
    )


def load_sector_exact_result(path: str | Path) -> SectorExactResult:
    """Load a fixed-sector exact result from ``.npz``."""

    data = np.load(path)
    return SectorExactResult(
        energy=float(data["energy"]),
        state=np.asarray(data["state"]),
        basis=np.asarray(data["basis"]),
        n_down=int(data["n_down"]),
    )


def sector_state_from_full_state(state, basis: np.ndarray) -> np.ndarray:
    """Project a full statevector onto a fixed-sector basis."""

    data = state.data if hasattr(state, "data") else np.asarray(state)
    return np.asarray(data)[basis]


def fixed_sector_fidelity(state, exact: SectorExactResult) -> float:
    """Return |<exact_sector|full_state>|^2."""

    projected = sector_state_from_full_state(state, exact.basis)
    return float(abs(np.vdot(exact.state, projected)) ** 2)


def state_fidelity(state, exact_state: np.ndarray | None) -> float | None:
    """Return |<exact|state>|^2 when an exact state is available."""

    if exact_state is None:
        return None
    data = state.data if hasattr(state, "data") else np.asarray(state)
    return float(abs(np.vdot(exact_state, data)) ** 2)


def bipartite_entropy(state, cut: int | None = None) -> float:
    """Von Neumann entropy in bits for a simple left/right qubit cut."""

    from qiskit.quantum_info import entropy, partial_trace

    num_qubits = state.num_qubits
    if cut is None:
        cut = num_qubits // 2
    traced_out = list(range(cut, num_qubits))
    reduced = partial_trace(state, traced_out)
    return float(entropy(reduced, base=2))


def bipartite_entropy_numpy(state: np.ndarray, cut: int | None = None) -> float:
    """Von Neumann entropy in bits for a NumPy statevector.

    This forms the smaller reduced density matrix instead of computing an SVD
    of the full bipartition matrix. It is faster and more stable for the
    19-site RVB/HVA statevectors used in this project.
    """

    data = np.asarray(state, dtype=complex)
    num_qubits = int(np.log2(data.size))
    if 2**num_qubits != data.size:
        raise ValueError("Statevector size must be a power of two")
    if cut is None:
        cut = num_qubits // 2
    if cut <= 0 or cut >= num_qubits:
        return 0.0

    left_dim = 2**cut
    right_dim = 2 ** (num_qubits - cut)
    matrix = data.reshape((right_dim, left_dim))
    if left_dim <= right_dim:
        rho = matrix.conj().T @ matrix
    else:
        rho = matrix @ matrix.conj().T
    eigenvalues = np.real(np.linalg.eigvalsh(rho))
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def apply_one_qubit_gate_numpy(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
) -> np.ndarray:
    """Apply a one-qubit gate to a little-endian statevector."""

    num_qubits = int(np.log2(state.size))
    axis = num_qubits - 1 - qubit
    tensor = state.reshape((2,) * num_qubits)
    moved = np.moveaxis(tensor, axis, 0)
    shape = moved.shape
    updated = gate @ moved.reshape(2, -1)
    moved = updated.reshape(shape)
    return np.moveaxis(moved, 0, axis).reshape(-1)


def apply_two_qubit_gate_numpy(
    state: np.ndarray,
    gate: np.ndarray,
    qubit_a: int,
    qubit_b: int,
) -> np.ndarray:
    """Apply a two-qubit gate to a little-endian statevector."""

    if qubit_a == qubit_b:
        raise ValueError("Two-qubit gate requires distinct qubits")
    num_qubits = int(np.log2(state.size))
    axes = [num_qubits - 1 - qubit_a, num_qubits - 1 - qubit_b]
    tensor = state.reshape((2,) * num_qubits)
    moved = np.moveaxis(tensor, axes, [0, 1])
    shape = moved.shape
    updated = gate @ moved.reshape(4, -1)
    moved = updated.reshape(shape)
    return np.moveaxis(moved, [0, 1], axes).reshape(-1)


def heisenberg_gate_matrix(theta: float) -> np.ndarray:
    """Matrix for exp[-i theta (XX + YY + ZZ)] on two qubits."""

    phase_same = np.exp(-1j * theta)
    phase_mixed = np.exp(1j * theta)
    cos_term = np.cos(2.0 * theta)
    sin_term = np.sin(2.0 * theta)
    return np.array(
        [
            [phase_same, 0.0, 0.0, 0.0],
            [0.0, phase_mixed * cos_term, -1j * phase_mixed * sin_term, 0.0],
            [0.0, -1j * phase_mixed * sin_term, phase_mixed * cos_term, 0.0],
            [0.0, 0.0, 0.0, phase_same],
        ],
        dtype=complex,
    )


def build_bond_index_cache(
    num_qubits: int,
    bonds: Iterable[Bond],
) -> list[BondIndexCache]:
    """Precompute basis partitions for fast two-qubit bond kernels."""

    index_dtype = np.int32 if num_qubits < 31 else np.int64
    indices = np.arange(2**num_qubits, dtype=index_dtype)
    caches: list[BondIndexCache] = []
    for i, j in bonds:
        bit_i = (indices >> i) & 1
        bit_j = (indices >> j) & 1
        caches.append(
            BondIndexCache(
                bond=(i, j),
                idx00=indices[(bit_i == 0) & (bit_j == 0)],
                idx01=indices[(bit_i == 0) & (bit_j == 1)],
                idx10=indices[(bit_i == 1) & (bit_j == 0)],
                idx11=indices[(bit_i == 1) & (bit_j == 1)],
            )
        )
    return caches


def build_sector_bond_index_cache(
    basis: np.ndarray,
    bonds: Iterable[Bond],
) -> list[BondIndexCache]:
    """Precompute fixed-sector basis partitions for Heisenberg kernels.

    The mixed-spin arrays are explicitly paired so ``idx10[k]`` is the sector
    position obtained from flipping the two qubits in ``idx01[k]``.
    """

    sector_basis = np.asarray(basis, dtype=np.int64)
    index_by_state = {int(state): index for index, state in enumerate(sector_basis)}
    position_dtype = np.int32 if len(sector_basis) < 2**31 else np.int64
    positions = np.arange(len(sector_basis), dtype=position_dtype)
    caches: list[BondIndexCache] = []
    for i, j in bonds:
        bit_i = (sector_basis >> i) & 1
        bit_j = (sector_basis >> j) & 1
        idx01_states = sector_basis[(bit_i == 0) & (bit_j == 1)]
        idx01 = []
        idx10 = []
        flip_mask = (1 << i) | (1 << j)
        for state in idx01_states:
            idx01.append(index_by_state[int(state)])
            idx10.append(index_by_state[int(state) ^ flip_mask])
        caches.append(
            BondIndexCache(
                bond=(i, j),
                idx00=positions[(bit_i == 0) & (bit_j == 0)],
                idx01=np.asarray(idx01, dtype=position_dtype),
                idx10=np.asarray(idx10, dtype=position_dtype),
                idx11=positions[(bit_i == 1) & (bit_j == 1)],
            )
        )
    return caches


def apply_heisenberg_gate_cached(
    state: np.ndarray,
    cache: BondIndexCache,
    theta: float,
) -> None:
    """Apply exp[-i theta (XX + YY + ZZ)] in-place for one cached bond."""

    phase_same = np.exp(-1j * theta)
    phase_mixed = np.exp(1j * theta)
    cos_term = np.cos(2.0 * theta)
    sin_term = np.sin(2.0 * theta)

    state[cache.idx00] *= phase_same
    state[cache.idx11] *= phase_same
    amp01 = state[cache.idx01].copy()
    amp10 = state[cache.idx10].copy()
    state[cache.idx01] = phase_mixed * (cos_term * amp01 - 1j * sin_term * amp10)
    state[cache.idx10] = phase_mixed * (-1j * sin_term * amp01 + cos_term * amp10)


def initial_statevector_numpy(
    num_qubits: int,
    bonds: Iterable[Bond],
    mode: InitialState = "singlet_pairs",
) -> np.ndarray:
    """Prepare the same initial state as ``prepare_initial_state``."""

    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1.0
    if mode == "zero":
        return state

    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z_gate = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    h_gate = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cx_gate = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=complex,
    )

    if mode == "neel":
        for qubit in range(1, num_qubits, 2):
            state = apply_one_qubit_gate_numpy(state, x_gate, qubit)
        return state
    if mode != "singlet_pairs":
        raise ValueError(f"Unknown initial state: {mode}")

    pairs = matching_from_bonds(bonds)
    paired = {qubit for pair in pairs for qubit in pair}
    leftovers = [qubit for qubit in range(num_qubits) if qubit not in paired]
    pairs.extend(
        (leftovers[index], leftovers[index + 1])
        for index in range(0, len(leftovers) - 1, 2)
    )

    for i, j in pairs:
        state = apply_one_qubit_gate_numpy(state, x_gate, j)
        state = apply_one_qubit_gate_numpy(state, h_gate, i)
        state = apply_two_qubit_gate_numpy(state, cx_gate, i, j)
        state = apply_one_qubit_gate_numpy(state, z_gate, i)
    return state


def heisenberg_statevector_numpy(
    num_qubits: int,
    bonds: list[Bond],
    layers: int,
    parameters: np.ndarray,
    parameterization: Parameterization = "shared",
    initial_state: InitialState = "singlet_pairs",
    bond_groups: list[str] | None = None,
) -> np.ndarray:
    """Fast statevector for the layered Heisenberg-gate ansatz."""

    parameters = np.asarray(parameters, dtype=float)
    expected = parameter_count(layers, bonds, parameterization, bond_groups)
    if len(parameters) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(parameters)}")

    state = initial_statevector_numpy(num_qubits, bonds, mode=initial_state)
    bond_caches = build_bond_index_cache(num_qubits, bonds)
    return heisenberg_statevector_from_initial_numpy(
        state,
        bond_caches,
        layers,
        parameters,
        parameterization=parameterization,
        bond_groups=bond_groups,
    )


def heisenberg_statevector_from_initial_numpy(
    initial_state: np.ndarray,
    bond_caches: list[BondIndexCache],
    layers: int,
    parameters: np.ndarray,
    parameterization: Parameterization = "shared",
    bond_groups: list[str] | None = None,
) -> np.ndarray:
    """Fast statevector evolution from a precomputed initial state."""

    parameters = np.asarray(parameters, dtype=float)
    bonds = [cache.bond for cache in bond_caches]
    expected = parameter_count(layers, bonds, parameterization, bond_groups)
    if len(parameters) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(parameters)}")

    state = np.array(initial_state, copy=True)
    cursor = 0
    group_names = ordered_group_names(bond_groups, len(bonds))
    active_groups = bond_groups or ["default"] * len(bonds)
    for layer in range(layers):
        if parameterization == "shared":
            theta = float(parameters[layer])
            for cache in bond_caches:
                apply_heisenberg_gate_cached(state, cache, theta)
        elif parameterization == "bond":
            for cache in bond_caches:
                apply_heisenberg_gate_cached(state, cache, float(parameters[cursor]))
                cursor += 1
        else:
            theta_by_group = {
                group: float(parameters[cursor + index])
                for index, group in enumerate(group_names)
            }
            cursor += len(group_names)
            for cache, group in zip(bond_caches, active_groups):
                apply_heisenberg_gate_cached(state, cache, theta_by_group[group])
    return state


def heisenberg_sector_state_from_initial_numpy(
    initial_sector_state: np.ndarray,
    sector_bond_caches: list[BondIndexCache],
    layers: int,
    parameters: np.ndarray,
    parameterization: Parameterization = "shared",
    bond_groups: list[str] | None = None,
) -> np.ndarray:
    """Fast Heisenberg-gate evolution inside one fixed-``S_z`` sector."""

    return heisenberg_statevector_from_initial_numpy(
        initial_sector_state,
        sector_bond_caches,
        layers,
        parameters,
        parameterization=parameterization,
        bond_groups=bond_groups,
    )


def heisenberg_energy_cached_numpy(
    state: np.ndarray,
    bond_caches: list[BondIndexCache],
    j_coupling: float = 1.0,
    pauli_scale: float = 0.25,
    weights: Iterable[float] | None = None,
) -> float:
    """Fast cached expectation of J * pauli_scale * sum(XX + YY + ZZ)."""

    probabilities = np.abs(state) ** 2
    weight_list = [1.0] * len(bond_caches) if weights is None else list(weights)
    total = 0.0
    for cache, weight in zip(bond_caches, weight_list):
        zz = (
            float(np.sum(probabilities[cache.idx00]))
            + float(np.sum(probabilities[cache.idx11]))
            - float(np.sum(probabilities[cache.idx01]))
            - float(np.sum(probabilities[cache.idx10]))
        )
        xy = 4.0 * float(np.real(np.vdot(state[cache.idx01], state[cache.idx10])))
        total += float(weight) * (zz + xy)
    return j_coupling * pauli_scale * total


def heisenberg_sector_energy_cached_numpy(
    sector_state: np.ndarray,
    sector_bond_caches: list[BondIndexCache],
    j_coupling: float = 1.0,
    pauli_scale: float = 0.25,
    weights: Iterable[float] | None = None,
) -> float:
    """Cached Heisenberg expectation in a fixed magnetization sector."""

    return heisenberg_energy_cached_numpy(
        sector_state,
        sector_bond_caches,
        j_coupling=j_coupling,
        pauli_scale=pauli_scale,
        weights=weights,
    )


def embed_sector_state(
    sector_state: np.ndarray,
    basis: np.ndarray,
    num_qubits: int,
) -> np.ndarray:
    """Embed a fixed-sector vector into the full computational basis."""

    full_state = np.zeros(2**num_qubits, dtype=complex)
    full_state[np.asarray(basis)] = np.asarray(sector_state)
    return full_state


def heisenberg_energy_numpy(
    state: np.ndarray,
    num_qubits: int,
    bonds: Iterable[Bond],
    j_coupling: float = 1.0,
    pauli_scale: float = 0.25,
    weights: Iterable[float] | None = None,
) -> float:
    """Fast expectation of J * pauli_scale * sum(XX + YY + ZZ)."""

    bond_caches = build_bond_index_cache(num_qubits, bonds)
    return heisenberg_energy_cached_numpy(
        state,
        bond_caches,
        j_coupling=j_coupling,
        pauli_scale=pauli_scale,
        weights=weights,
    )


def bond_correlation_vector_numpy(
    state: np.ndarray,
    num_qubits: int,
    bonds: Iterable[Bond],
    pauli_scale: float = 1.0,
) -> np.ndarray:
    """Return per-bond <pauli_scale * (XX + YY + ZZ)> correlations."""

    return np.array(
        [
            heisenberg_energy_numpy(
                state,
                num_qubits,
                [bond],
                pauli_scale=pauli_scale,
            )
            for bond in bonds
        ]
    )


def bond_delocalization_metrics(correlations: np.ndarray) -> dict[str, float]:
    """Summarize how spread out antiferromagnetic bond weight is."""

    values = np.asarray(correlations, dtype=float)
    singlet_weight = np.maximum(0.0, -values)
    total_weight = float(np.sum(singlet_weight))
    if total_weight > 0:
        probabilities = singlet_weight / total_weight
        participation = float(1.0 / np.sum(probabilities**2))
    else:
        participation = 0.0
    strong_mask = values < -2.0
    strong_weight = float(np.sum(singlet_weight[strong_mask]))
    frozen_weight_fraction = strong_weight / total_weight if total_weight > 0 else 0.0
    return {
        "bond_corr_min": float(np.min(values)),
        "bond_corr_max": float(np.max(values)),
        "bond_corr_mean": float(np.mean(values)),
        "bond_corr_std": float(np.std(values)),
        "af_weight_participation": participation,
        "af_participation_ratio": participation / len(values) if len(values) else 0.0,
        "strong_dimer_count": float(np.sum(strong_mask)),
        "strong_dimer_fraction": float(np.mean(strong_mask)) if len(values) else 0.0,
        "strong_dimer_af_weight_fraction": frozen_weight_fraction,
    }


def spin_correlation_matrix_numpy(
    state: np.ndarray,
    num_qubits: int,
    pauli_scale: float = 1.0,
) -> np.ndarray:
    """Return dense pair correlation matrix for XX+YY+ZZ convention."""

    matrix = np.zeros((num_qubits, num_qubits), dtype=float)
    for i in range(num_qubits):
        matrix[i, i] = 3.0 * pauli_scale
        for j in range(i + 1, num_qubits):
            value = heisenberg_energy_numpy(
                state,
                num_qubits,
                [(i, j)],
                pauli_scale=pauli_scale,
            )
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def graph_distance_matrix(num_sites: int, bonds: Iterable[Bond]) -> np.ndarray:
    """Return all-pairs shortest-path distances on the bond graph."""

    distances = np.full((num_sites, num_sites), np.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)
    for i, j in bonds:
        distances[i, j] = 1.0
        distances[j, i] = 1.0
    for k in range(num_sites):
        distances = np.minimum(distances, distances[:, [k]] + distances[[k], :])
    return distances


def spin_correlation_distance_profile(
    matrix: np.ndarray,
    distances: np.ndarray,
) -> list[dict[str, float | int]]:
    """Average absolute spin correlation by graph distance.

    Depenbrock et al. diagnose the kagome spin liquid with short-ranged
    spin correlations.  For this finite open 19-site patch, graph distance is a
    conservative geometry-free proxy for that decay diagnostic.
    """

    corr = np.asarray(matrix, dtype=float)
    dist = np.asarray(distances, dtype=float)
    if corr.shape != dist.shape or corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("matrix and distances must be square arrays with matching shape")

    rows: list[dict[str, float | int]] = []
    finite_distances = sorted(
        int(value)
        for value in np.unique(dist[np.isfinite(dist)])
        if value > 0
    )
    for distance in finite_distances:
        mask = np.triu(dist == distance, k=1)
        values = corr[mask]
        if values.size == 0:
            continue
        abs_values = np.abs(values)
        rows.append(
            {
                "graph_distance": distance,
                "pair_count": int(values.size),
                "mean_abs_correlation": float(np.mean(abs_values)),
                "max_abs_correlation": float(np.max(abs_values)),
                "rms_correlation": float(np.sqrt(np.mean(values**2))),
            }
        )
    return rows


def exponential_decay_length_from_profile(
    profile: Iterable[dict[str, float | int]],
    *,
    value_key: str = "mean_abs_correlation",
) -> float:
    """Fit ``value ~= a * exp(-distance / xi)`` and return ``xi``.

    The fit is intentionally simple and is only used as a compact finite-patch
    diagnostic.  It should not be read as a thermodynamic correlation length.
    """

    distances = []
    values = []
    for row in profile:
        distance = float(row["graph_distance"])
        value = float(row[value_key])
        if distance > 0.0 and value > 0.0 and np.isfinite(value):
            distances.append(distance)
            values.append(value)
    if len(values) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.asarray(distances), np.log(np.asarray(values)), deg=1)
    if slope >= 0.0:
        return float("inf")
    return float(-1.0 / slope)


def entropy_profile_numpy(state: np.ndarray, cuts: Iterable[int] | None = None) -> np.ndarray:
    """Return bipartite entropy for selected contiguous qubit cuts."""

    num_qubits = int(np.log2(np.asarray(state).size))
    active_cuts = range(1, num_qubits) if cuts is None else cuts
    return np.array([bipartite_entropy_numpy(state, cut=cut) for cut in active_cuts])


def sector_weight(state: np.ndarray, basis: np.ndarray) -> float:
    """Return ||P_sector psi||^2 for a sector basis."""

    data = np.asarray(state)
    return float(np.sum(np.abs(data[basis]) ** 2))


def literature_benchmark_row(
    num_sites: int,
    energy_unscaled_pauli: float,
) -> dict[str, float | str]:
    """Convert a finite-patch energy into paper benchmark conventions."""

    physical_energy = float(energy_unscaled_pauli) / 4.0
    physical_per_site = physical_energy / float(num_sites)
    dmrg_per_site = DEPENBROCK_DMRG_PHYSICAL_ENERGY_PER_SITE
    return {
        "reference": "Depenbrock2012 thermodynamic DMRG / Ahsan2025 rounded 19-site calibration",
        "energy_unscaled_pauli": float(energy_unscaled_pauli),
        "energy_physical_spin": physical_energy,
        "energy_physical_per_site": physical_per_site,
        "depenbrock_dmrg_physical_per_site": dmrg_per_site,
        "delta_vs_depenbrock_dmrg_per_site": physical_per_site - dmrg_per_site,
        "ahsan_19site_rounded_unscaled_energy": AHSAN_19SITE_UNSCALED_ENERGY_ROUNDED,
        "delta_vs_ahsan_rounded_19site_unscaled": float(energy_unscaled_pauli)
        - AHSAN_19SITE_UNSCALED_ENERGY_ROUNDED,
    }


def exact_reference_energy(path: str | Path = "results/19site_fixed_sz_exact_n9.npz") -> float:
    """Load the exact reference energy from a cached fixed-sector result."""

    return load_sector_exact_result(path).energy


def optimize_vqe_numpy(
    num_qubits: int,
    bonds: list[Bond],
    layers: int,
    parameterization: Parameterization = "shared",
    initial_state: InitialState = "singlet_pairs",
    bond_groups: list[str] | None = None,
    exact_energy: float | None = None,
    exact_state: np.ndarray | None = None,
    pauli_scale: float = 0.25,
    maxiter: int = 120,
    seed: int = 7,
    restarts: int = 3,
) -> VQEResult:
    """Optimize the Heisenberg ansatz using the fast NumPy statevector path."""

    num_parameters = parameter_count(layers, bonds, parameterization, bond_groups)
    initial_state_data = initial_statevector_numpy(num_qubits, bonds, mode=initial_state)
    bond_caches = build_bond_index_cache(num_qubits, bonds)

    def objective(values: np.ndarray) -> float:
        state = heisenberg_statevector_from_initial_numpy(
            initial_state_data,
            bond_caches,
            layers,
            np.asarray(values, dtype=float),
            parameterization=parameterization,
            bond_groups=bond_groups,
        )
        return heisenberg_energy_cached_numpy(
            state,
            bond_caches,
            pauli_scale=pauli_scale,
        )

    if num_parameters == 0:
        parameters = np.array([])
        state = heisenberg_statevector_from_initial_numpy(
            initial_state_data,
            bond_caches,
            layers,
            parameters,
            parameterization=parameterization,
            bond_groups=bond_groups,
        )
        energy = heisenberg_energy_cached_numpy(
            state,
            bond_caches,
            pauli_scale=pauli_scale,
        )
        reference = exact_energy if exact_energy is not None else energy
        return VQEResult(
            layers=layers,
            energy=energy,
            exact_energy=reference,
            error=energy - reference,
            fidelity=state_fidelity(state, exact_state),
            entropy=bipartite_entropy_numpy(state),
            parameters=parameters,
            success=True,
            evaluations=1,
            message="No variational parameters.",
        )

    if num_parameters == 1:
        result = minimize_scalar(
            lambda value: objective(np.array([value])),
            bounds=(-np.pi, np.pi),
            method="bounded",
            options={"maxiter": maxiter, "xatol": 1e-7},
        )
        parameters = np.array([float(result.x)])
        evaluations = int(getattr(result, "nfev", 0))
        success = bool(result.success)
        message = str(result.message)
        energy = float(result.fun)
    else:
        rng = np.random.default_rng(seed)
        starts = [np.zeros(num_parameters)]
        starts.extend(
            rng.uniform(-0.2, 0.2, num_parameters)
            for _ in range(max(0, restarts - 1))
        )
        best = None
        evaluations = 0
        for x0 in starts:
            result = minimize(
                objective,
                x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.25, "tol": 1e-7},
            )
            evaluations += int(getattr(result, "nfev", 0))
            if best is None or result.fun < best.fun:
                best = result
        if best is None:
            raise RuntimeError("Optimizer did not return a result")
        parameters = np.asarray(best.x)
        success = bool(best.success)
        message = str(best.message)
        energy = float(best.fun)

    state = heisenberg_statevector_from_initial_numpy(
        initial_state_data,
        bond_caches,
        layers,
        parameters,
        parameterization=parameterization,
        bond_groups=bond_groups,
    )
    reference = exact_energy if exact_energy is not None else energy
    return VQEResult(
        layers=layers,
        energy=energy,
        exact_energy=reference,
        error=energy - reference,
        fidelity=state_fidelity(state, exact_state),
        entropy=bipartite_entropy_numpy(state),
        parameters=parameters,
        success=success,
        evaluations=evaluations,
        message=message,
    )


def run_depth_sweep_numpy(
    num_qubits: int,
    bonds: list[Bond],
    max_layers: int = 4,
    parameterization: Parameterization = "shared",
    initial_state: InitialState = "singlet_pairs",
    bond_groups: list[str] | None = None,
    exact_energy: float | None = None,
    exact_state: np.ndarray | None = None,
    pauli_scale: float = 0.25,
    maxiter: int = 120,
    seed: int = 7,
) -> list[VQEResult]:
    """Run p=0..max_layers with the fast NumPy Heisenberg ansatz backend."""

    results: list[VQEResult] = []
    for layers in range(max_layers + 1):
        results.append(
            optimize_vqe_numpy(
                num_qubits,
                bonds,
                layers,
                parameterization=parameterization,
                initial_state=initial_state,
                bond_groups=bond_groups,
                exact_energy=exact_energy,
                exact_state=exact_state,
                pauli_scale=pauli_scale,
                maxiter=maxiter,
                seed=seed + 100 * layers,
            )
        )
    return results


def spin_z_expectations(state) -> np.ndarray:
    """Compute <S_i^z> = <Z_i>/2 for all sites."""

    from qiskit.quantum_info import SparsePauliOp

    values = []
    for qubit in range(state.num_qubits):
        op = SparsePauliOp.from_list([(pauli_label(state.num_qubits, qubit, qubit, "Z"), 0.5)])
        values.append(expectation_value(state, op))
    return np.array(values)


def spin_z_expectations_numpy(state: np.ndarray, num_qubits: int | None = None) -> np.ndarray:
    """Compute <S_i^z> = <Z_i>/2 for all sites from a full statevector."""

    data = np.asarray(state)
    if num_qubits is None:
        num_qubits = int(np.log2(data.size))
    indices = np.arange(data.size, dtype=np.int64)
    probabilities = np.abs(data) ** 2
    values = []
    for qubit in range(num_qubits):
        z_factor = 1.0 - 2.0 * ((indices >> qubit) & 1)
        values.append(0.5 * float(np.dot(probabilities, z_factor)))
    return np.array(values)


def max_abs_magnetization_numpy(state: np.ndarray, num_qubits: int | None = None) -> float:
    """Return max_i |<S_i^z>|."""

    return float(np.max(np.abs(spin_z_expectations_numpy(state, num_qubits=num_qubits))))


def bond_correlations(
    state,
    num_qubits: int,
    bonds: Iterable[Bond],
    pauli_scale: float = 0.25,
) -> dict[Bond, float]:
    """Compute <S_i dot S_j> for each requested bond."""

    correlations: dict[Bond, float] = {}
    for i, j in bonds:
        op = heisenberg_hamiltonian(
            num_qubits,
            [(i, j)],
            j_coupling=1.0,
            pauli_scale=pauli_scale,
        )
        correlations[(i, j)] = expectation_value(state, op)
    return correlations


def optimize_vqe(
    circuit,
    parameters: list[object],
    hamiltonian,
    exact_energy: float | None = None,
    exact_state: np.ndarray | None = None,
    maxiter: int = 250,
    seed: int = 7,
    restarts: int = 4,
) -> VQEResult:
    """Run a small statevector VQE optimization."""

    from qiskit.quantum_info import Statevector

    if not parameters:
        state = Statevector.from_instruction(circuit)
        energy = expectation_value(state, hamiltonian)
        reference = exact_energy if exact_energy is not None else energy
        return VQEResult(
            layers=0,
            energy=energy,
            exact_energy=reference,
            error=energy - reference,
            fidelity=state_fidelity(state, exact_state),
            entropy=bipartite_entropy(state),
            parameters=np.array([]),
            success=True,
            evaluations=1,
            message="No variational parameters.",
        )

    rng = np.random.default_rng(seed)
    starts = [np.zeros(len(parameters))]
    starts.extend(rng.uniform(-0.2, 0.2, len(parameters)) for _ in range(restarts - 1))

    best = None
    best_value = np.inf
    best_evaluations = 0

    def objective(values: np.ndarray) -> float:
        state = statevector_from_params(circuit, parameters, values)
        return expectation_value(state, hamiltonian)

    for x0 in starts:
        result = minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.25, "tol": 1e-7},
        )
        best_evaluations += int(getattr(result, "nfev", 0))
        if result.fun < best_value:
            best_value = float(result.fun)
            best = result

    if best is None:
        raise RuntimeError("Optimizer did not return a result")

    state = statevector_from_params(circuit, parameters, np.asarray(best.x))
    reference = exact_energy if exact_energy is not None else best_value
    depth = max((int(param.name.split("_")[1]) for param in parameters), default=-1) + 1
    return VQEResult(
        layers=depth,
        energy=best_value,
        exact_energy=reference,
        error=best_value - reference,
        fidelity=state_fidelity(state, exact_state),
        entropy=bipartite_entropy(state),
        parameters=np.asarray(best.x),
        success=bool(best.success),
        evaluations=best_evaluations,
        message=str(best.message),
    )


def run_depth_sweep(
    num_qubits: int,
    bonds: list[Bond],
    max_layers: int = 4,
    parameterization: Parameterization = "shared",
    initial_state: InitialState = "singlet_pairs",
    pauli_scale: float = 0.25,
    maxiter: int = 250,
    seed: int = 7,
) -> tuple[list[VQEResult], object, np.ndarray]:
    """Run exact diagonalization plus VQE depths p=0..max_layers."""

    hamiltonian = heisenberg_hamiltonian(num_qubits, bonds, pauli_scale=pauli_scale)
    exact_energy, exact_state = exact_ground_state(hamiltonian)

    results: list[VQEResult] = []
    for layers in range(max_layers + 1):
        circuit, parameters = build_heisenberg_ansatz(
            num_qubits,
            bonds,
            layers,
            parameterization=parameterization,
            initial_state=initial_state,
        )
        result = optimize_vqe(
            circuit,
            parameters,
            hamiltonian,
            exact_energy=exact_energy,
            exact_state=exact_state,
            maxiter=maxiter,
            seed=seed + 100 * layers,
        )
        results.append(result)
    return results, hamiltonian, exact_state
