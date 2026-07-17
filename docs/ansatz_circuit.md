# Ansatz Circuit

This page records the circuit ansatz used for the reported weighted-RVB + HVA
results.

## State Prepared

The reported state has the form

```text
|psi_p(theta)> = U_p(theta) |psi_RVB>
```

where `|psi_RVB>` is the signed weighted-RVB initializer obtained from the
54-covering classical dimer subspace calculation. The HVA refinement circuit is

```text
U_p(theta) =
prod_{layer=1..p} prod_{color=0..3} G_color(theta_layer,color)
```

with

```text
G_color(theta) =
prod_{(i,j) in E_color} exp[-i theta (X_i X_j + Y_i Y_j + Z_i Z_j)].
```

The current best run uses `p = 4`, so the edge-colored HVA has `4p = 16`
parameters.

## Qiskit Gate Decomposition

Qiskit implements

```text
RXX(phi) = exp[-i phi XX / 2]
RYY(phi) = exp[-i phi YY / 2]
RZZ(phi) = exp[-i phi ZZ / 2]
```

so each Heisenberg interaction is implemented as

```python
qc.rxx(2 * theta, i, j)
qc.ryy(2 * theta, i, j)
qc.rzz(2 * theta, i, j)
```

This realizes

```text
exp[-i theta (XX + YY + ZZ)].
```

The source implementation is
`src/kagome_heisenberg_poc.py::build_heisenberg_ansatz`.

## Edge-Color Groups

Bonds in one color group are disjoint, so all gates in that group can be applied
in parallel on hardware with suitable connectivity.

| Color | Bonds |
| --- | --- |
| `color0` | `(1,2)`, `(4,6)`, `(8,10)`, `(11,12)`, `(14,16)`, `(17,18)` |
| `color1` | `(0,1)`, `(2,4)`, `(6,8)`, `(10,11)`, `(12,14)`, `(16,17)` |
| `color2` | `(2,3)`, `(4,5)`, `(6,7)`, `(8,9)`, `(10,12)`, `(13,14)`, `(15,16)`, `(1,11)`, `(0,17)` |
| `color3` | `(3,4)`, `(5,6)`, `(7,8)`, `(9,10)`, `(12,13)`, `(14,15)`, `(16,18)`, `(2,11)`, `(1,17)` |

## Minimal Qiskit Template

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

num_qubits = 19
edge_groups = {
    "color0": [(1, 2), (4, 6), (8, 10), (11, 12), (14, 16), (17, 18)],
    "color1": [(0, 1), (2, 4), (6, 8), (10, 11), (12, 14), (16, 17)],
    "color2": [(2, 3), (4, 5), (6, 7), (8, 9), (10, 12), (13, 14), (15, 16), (1, 11), (0, 17)],
    "color3": [(3, 4), (5, 6), (7, 8), (9, 10), (12, 13), (14, 15), (16, 18), (2, 11), (1, 17)],
}

def append_heisenberg_hva(qc: QuantumCircuit, depth: int) -> list[Parameter]:
    parameters = []
    for layer in range(depth):
        for color, bonds in edge_groups.items():
            theta = Parameter(f"theta_{layer}_{color}")
            parameters.append(theta)
            for i, j in bonds:
                qc.rxx(2 * theta, i, j)
                qc.ryy(2 * theta, i, j)
                qc.rzz(2 * theta, i, j)
    return parameters

qc = QuantumCircuit(num_qubits)
theta = append_heisenberg_hva(qc, depth=4)
```

Important limitation: the current weighted-RVB initializer is prepared
classically for exact-state simulation. A hardware state-preparation circuit for
`|psi_RVB>` is future work. The HVA refinement itself is Qiskit-native and
expressed directly in two-qubit Pauli rotation gates.
