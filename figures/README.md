# Figures

Main presentation figure:

- `one_page_summary.png`
- `workflow_flowchart.png`

The main comparison panel uses error vs exact rather than raw energy so the
RVB-to-HVA improvement is visible at presentation scale.

`workflow_flowchart.png` summarizes the proof-of-concept flow: 19-site graph,
weighted-RVB initializer, Heisenberg-HVA refinement, exact diagnostics, and
calibration-informed future circuit design.

Depth plots:

- `energy_vs_hva_depth.png`
- `error_vs_hva_depth.png`
- `fidelity_vs_hva_depth.png`
- `magnetization_vs_hva_depth.png`
- `entropy_vs_hva_depth.png`
- `spin_distance_profile.png`

Bond maps:

- `bond_map_static_dimer.png`
- `bond_map_equal_rvb.png`
- `bond_map_weighted_rvb.png`
- `bond_map_weighted_hva_p1.png`
- `bond_map_weighted_hva_p2.png`
- `bond_map_weighted_hva_p3.png`
- `bond_map_weighted_hva_p4.png`
- `bond_map_exact.png`
- `bond_map_error_p2_vs_exact.png`
- `bond_map_error_best_hva_vs_exact.png`

Calibration:

- `calibration_energy_vs_fidelity.png`
- `calibration_energy_vs_fidelity_zoom.png`

This scatter plot reads all rows from `results/19site_calibration_scan.csv` and
highlights the best no-calibration weighted-RVB + HVA result. The main plot uses
a symlog x-axis; the zoomed plot shows the high-fidelity low-error region on a
linear scale.

The bond-map layout is deterministic and graph-theoretic for visualization. It is not a physical Kagome embedding.
