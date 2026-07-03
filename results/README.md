# Results

Official files:

- `final_result_table.csv` - main result table.
- `19site_heisenberg_hva_fast_weighted_p4_neldermead_*.csv` - current best serious no-calibration HVA p=4 result.
- `19site_heisenberg_hva_fast_weighted_p2_reference.csv` - cached quick-check no-calibration HVA p=2 reference.
- `19site_heisenberg_hva_fast_weighted_p2_reference_history.csv` - optimizer history with start IDs, parameter vectors, and best-so-far energy.
- `19site_heisenberg_hva_fast_weighted_p2_reference.metadata.json` - reproducibility metadata.
- `19site_rvb_subspace.csv` - deterministic 54-covering RVB generalized-eigenproblem summary.
- `19site_calibration_scan.csv` - current exact calibrated-Hamiltonian reference scan output.
- `calibration_comparison.csv` - method-vs-calibrated-reference comparison table.
- `calibration_mode_summary.csv` - best calibration row by scan mode.
- `calibration_top10.csv` - ten closest nontrivial calibrated-Hamiltonian reference rows.
- `no_calibration_vs_best_calibration.csv` - compact best-HVA vs best calibrated-reference comparison.

Diagnostic files:

- `19site_bond_correlations_by_state.csv`
- `19site_entropy_profiles.csv`
- `19site_site_magnetization.csv`
- `19site_spin_correlation_*.csv`
- `19site_hva_p1_gradients.csv`

Historical result CSVs are excluded from the release zip to avoid confusing the current official story. In the working tree they may live under `results/legacy/`. Smoke-test outputs may live under `results/smoke/` and are not official results.
