# Figures

Figure assets are grouped into small folders:

- `ansatz/` - ansatz circuit schematics and gate-level Qiskit views.
- `summary/` - workflow and compact summary figures.
- `bond_maps/` - bond-correlation maps and map differences.
- `depth/` - depth-sweep and correlation-distance diagnostics.
- `calibration/` - calibration-reference comparison plots.

Main presentation figures:

- `summary/one_page_summary.png`
- `summary/workflow_flowchart.png`
- `ansatz/ansatz_circuit_schematic.png`

The main comparison panel uses error vs exact rather than raw energy so the
RVB-to-HVA improvement is visible at presentation scale.

`summary/workflow_flowchart.png` summarizes the proof-of-concept flow: 19-site
graph, weighted-RVB initializer, Heisenberg-HVA refinement, exact diagnostics,
and calibration-informed future circuit design.

`ansatz/ansatz_circuit_schematic.png` is a block-level edge-colored
Heisenberg-HVA circuit schematic used after weighted-RVB initialization. It
shows the layer structure, the four color blocks per layer, and the Qiskit
two-qubit decomposition into `RXX(2theta)`, `RYY(2theta)`, and `RZZ(2theta)`.
`ansatz/full_gate_level_hva_p1_qiskit.png` and
`ansatz/full_gate_level_hva_p4_qiskit.png` show representative/full gate-level
Qiskit views. The full 19-qubit `p=4` circuit is omitted from the main report
because it contains 120 Heisenberg bond gates, or 360 Pauli-rotation gates.

Depth plots:

- `depth/energy_vs_hva_depth.png`
- `depth/error_vs_hva_depth.png`
- `depth/fidelity_vs_hva_depth.png`
- `depth/magnetization_vs_hva_depth.png`
- `depth/entropy_vs_hva_depth.png`
- `depth/spin_distance_profile.png`

Bond maps:

- `bond_maps/bond_map_static_dimer.png`
- `bond_maps/bond_map_equal_rvb.png`
- `bond_maps/bond_map_weighted_rvb.png`
- `bond_maps/bond_map_weighted_hva_p1.png`
- `bond_maps/bond_map_weighted_hva_p2.png`
- `bond_maps/bond_map_weighted_hva_p3.png`
- `bond_maps/bond_map_weighted_hva_p4.png`
- `bond_maps/bond_map_exact.png`
- `bond_maps/bond_map_error_p2_vs_exact.png`
- `bond_maps/bond_map_error_best_hva_vs_exact.png`

Calibration:

- `calibration/calibration_energy_vs_fidelity.png`
- `calibration/calibration_energy_vs_fidelity_zoom.png`

This scatter plot reads all rows from `results/19site_calibration_scan.csv` and
highlights the best no-calibration weighted-RVB + HVA result. The main plot uses
a symlog x-axis; the zoomed plot shows the high-fidelity low-error region on a
linear scale.

The bond-map layout is deterministic and graph-theoretic for visualization. It is not a physical Kagome embedding.
