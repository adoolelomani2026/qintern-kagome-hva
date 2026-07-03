# Weighted-RVB Initialization and Heisenberg-HVA Refinement for the 19-Site Kagome Antiferromagnetic Heisenberg Model

We study the 19-site Kagome antiferromagnetic Heisenberg model in the unscaled
Pauli convention using exact fixed-`Sz` benchmarking. A static 9-singlet dimer
state gives `E=-27.0000` and `F=0.0142`. A classically optimized signed
weighted-RVB state over the 54 maximum dimer coverings improves this to
`E=-29.0121` and `F=0.9785`. Shallow edge-colored Heisenberg-HVA refinement
further improves the state to `E=-29.0376` and `F=0.9825` at depth `p=4`,
without Hamiltonian calibration.

These results provide finite-size QSL-like evidence through energy, fidelity,
local magnetization, entanglement, and distributed bond-correlation diagnostics.
They do not constitute scalable or hardware-ready QSL preparation.

**Main claim:** On the 19-site Kagome Heisenberg benchmark, a classically
optimized signed weighted-RVB state over 54 maximum dimer coverings reaches
`E=-29.0121`, `F=0.9785`. Shallow edge-colored Heisenberg-HVA refinement
improves this to `E=-29.0376`, `F=0.9825` at `p=4`, without Hamiltonian
calibration. This is finite-size evidence, not scalable QSL preparation.

A broader exact calibrated-Hamiltonian scan currently finds even closer
finite-size reference states. The p=4 HVA result should therefore be read as
the strongest current no-calibration circuit-refinement baseline, not as a
replacement for Hamiltonian calibration.

Project context: QIntern 2026 project `qi26_09`, mentored by Dr. Muhammad
Ahsan through QWorld Association, QResearch Department.

## Provenance And Scope

The original Ahsan Kagome-VQE material is used here as provenance and
methodology context, not as code to copy into the current workflow. In
particular, it provides the source cross-check for the 19-site Kagome
connectivity, context for a hardware-efficient Qiskit VQE setup, and the
calibrated-Hamiltonian/defect-triangle strategy. The present proof-of-concept
develops a separate no-calibration route based on signed weighted-RVB
initialization and shallow edge-colored Heisenberg-HVA refinement.

The roles are:

| Component | Role |
| --- | --- |
| External Ahsan Kagome-VQE material | Original calibrated-Hamiltonian/VQE provenance |
| 19-site bond list | Confirmed against Ahsan's notebook |
| Weighted RVB | Main physics initializer and insight |
| Heisenberg-HVA | No-calibration circuit-refinement path |
| Calibration scans | Exact-state reference comparison, not the main claim |
| ODR / IBM Runtime ideas | Future hardware/noise-mitigation extension |

When discussing the original calibrated-Hamiltonian direction, use cautious
language: defect or triangle couplings are increased above the uniform value,
with reported calibrated values depending on the patch, Hamiltonian convention,
defect set, and scan. The current repository reports exact values only for the
calibration scans reproduced in `results/19site_calibration_scan.csv`.

The Ahsan archive itself is not included in this repository; none of the
reproduction scripts import it or require it.

## Install

Core NumPy/SciPy workflow:

```powershell
pip install -r requirements-core.txt
pip install -r requirements-dev.txt
```

Optional Qiskit notebooks and legacy baselines:

```powershell
pip install -r requirements-qiskit.txt
```

The exact submitted environment is recorded in `requirements-lock.txt`; the
combined pinned environment is still available in `requirements.txt`. Tested on
Python 3.14.3; use Python >= 3.10. If package resolution fails on Python 3.14,
use Python 3.11 or 3.12 with compatible package versions.

Conda users can start from:

```powershell
conda env create -f environment.yml
```

Editable install:

```powershell
pip install -e .
```

## Reproduce

For stable performance, the run scripts set common BLAS/OpenMP thread pools to
one by default:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

Fast quick check from cached HVA data:

```powershell
.\run_all.ps1
```

Linux/macOS:

```bash
bash run_all.sh
```

To recompute the old p=2 HVA reference instead of using cached result CSVs:

```powershell
.\run_all.ps1 -RecomputeHva
```

```bash
bash run_all.sh --recompute-hva
```

Serious p=4 and wider calibration commands are wired as:

```powershell
.\run_all.ps1 -Serious
```

Those jobs are intentionally heavier. The current official release result is
p=0,1,2,3,4 from the serious Nelder-Mead run.
Serious mode regenerates diagnostics after the p=4 jobs and lets
`build_final_summary.py` auto-select the lowest-energy weighted-HVA CSV. The
quick path auto-selects the lowest-energy cached weighted-HVA CSV unless an
explicit `--hva-csv` / `HVA_CSV` is used. Serious mode may take hours depending
on hardware.

## Active Files

- `scripts/run_19site_rvb_subspace.py` - deterministic 54-covering RVB
  generalized eigenproblem, `H c = E S c`.
- `scripts/run_19site_heisenberg_hva_fast.py` - fixed-`Sz` cached NumPy
  edge-colored Heisenberg-HVA optimizer.
- `scripts/build_final_summary.py` - final table, cached-diagnostics summary,
  and calibration comparison.
- `scripts/run_19site_diagnostics.py` - bond correlations, magnetization,
  spin-correlation matrices, and entropy profiles.
- `scripts/make_figures.py` - bond maps, depth plots, and one-page summary.
- `scripts/run_19site_calibration_scan.py` - group, bond, or triangle Jprime
  calibration scans.
- `notebooks/main_19site_weighted_rvb_hva_result.ipynb` - clean presentation
  notebook.

Older diagnostic notebooks and scripts are in `notebooks/legacy/` and
`scripts/legacy/`. Older result CSVs are excluded from the release package to
avoid confusion. The p=4 smoke test is excluded from final tables/plots.

## Hamiltonian Convention

Main results use:

```text
H = sum_<ij> (XX + YY + ZZ)
```

Physical spin energies are one quarter of this:

```text
H_phys = sum_<ij> S_i . S_j
       = (1/4) sum_<ij> (XX + YY + ZZ)
```

The exact benchmark is loaded from `results/19site_fixed_sz_exact_n9.npz`:

```text
E0 = -29.146168109350135
```

## Current Result

The main output is `results/final_result_table.csv`.

| State | Energy | Error | Fidelity | Gap Closed | Entropy | Max Mag. | AF Part. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static dimer | -27.00000000 | 2.14616811 | 0.014241 | 0.00% | 1.000000 | 0.500000 | 9.000 |
| Equal RVB-54 | -26.39781307 | 2.74835504 | 0.004371 | -28.06% | 3.366154 | 0.129849 | 21.870 |
| Weighted RVB-54 | -29.01207210 | 0.13409601 | 0.978536 | 93.75% | 3.129502 | 0.107755 | 20.429 |
| Weighted RVB + HVA p=1 | -29.02302627 | 0.12314184 | 0.979995 | 94.26% | 3.129583 | 0.107528 | 20.406 |
| Weighted RVB + HVA p=2 | -29.03633182 | 0.10983629 | 0.982446 | 94.88% | 3.140691 | 0.104773 | 20.434 |
| Weighted RVB + HVA p=3 | -29.03749397 | 0.10867414 | 0.982442 | 94.94% | 3.141921 | 0.104388 | 20.443 |
| Weighted RVB + HVA p=4 | -29.03760135 | 0.10856676 | 0.982518 | 94.94% | 3.155489 | 0.104055 | 20.443 |
| Exact | -29.14616811 | 0.00000000 | 1.000000 | 100.00% | 3.212669 | 0.094007 | 20.461 |

The table also includes physical-spin energies, selected-cut entropy summaries,
bond-correlation L2/MAE/Pearson errors, magnetization vector errors, and sector
weight.

## Exact Calibrated-Hamiltonian Reference Scan

`results/calibration_comparison.csv` includes the best row from the current
exact calibrated-Hamiltonian reference scan, while
`results/calibration_mode_summary.csv` reports the best row by scan mode. The
best nontrivial row found so far is triangle
`triangle_4_6-7-8, J'=1.02`, with target energy error `0.0000370` and fidelity
`0.999966`. This is closer than the current no-calibration HVA p=4 baseline in
both energy and fidelity.

| Method | Energy error | Fidelity |
| --- | ---: | ---: |
| Weighted RVB + HVA p=4 | 0.108567 | 0.982518 |
| Calibrated group `color1`, `J'=1.02` | 0.003945 | 0.994967 |
| Calibrated bond `bond_4_14-16`, `J'=1.02` | 0.000510 | 0.999626 |
| Calibrated triangle `triangle_4_6-7-8`, `J'=1.02` | 0.000037 | 0.999966 |

These exact calibrated-Hamiltonian reference scans therefore produce closer
finite-size exact-state references than the current no-calibration HVA baseline.
The no-calibration p=4 result should be interpreted as a strong
circuit-compatible refinement path from a high-quality weighted-RVB initializer,
not as outperforming Hamiltonian calibration.

The scan uses the exact ground state of a modified Hamiltonian and then
evaluates that state against the original Hamiltonian. It is not yet a direct
hardware-efficient circuit-preparation comparison.

Run the broader scan with:

```powershell
python scripts/run_19site_calibration_scan.py --scan-mode group --all-groups --jprimes 1.00,1.02,1.04,1.05,1.06,1.08,1.10,1.15,1.20,1.25,1.30,1.40,1.50,1.75,1.95,2.10,2.30 --append
```

Additional modes:

```powershell
python scripts/run_19site_calibration_scan.py --scan-mode bond --all-bonds --jprimes 1.00,1.02,1.04,1.06,1.08,1.10,1.15,1.20 --append
python scripts/run_19site_calibration_scan.py --scan-mode triangle --all-triangles --jprimes 1.00,1.02,1.04,1.06,1.08,1.10,1.15,1.20 --append
```

After scans, `scripts/build_final_summary.py` writes both
`results/calibration_comparison.csv` and
`results/calibration_mode_summary.csv`, plus
`results/calibration_top10.csv` and
`results/no_calibration_vs_best_calibration.csv`.

## Diagnostics And Figures

AF bond participation is:

```text
p_ij = max(0, -C_ij) / sum_kl max(0, -C_kl)
P_AF = 1 / sum_ij p_ij^2
C_ij = <XX + YY + ZZ>_ij
```

`strong_dimer_count` counts bonds with `C_ij < -2` in the unscaled Pauli
convention. A perfect singlet has `C_ij = -3`.

Generated figures live in `figures/`:

- `bond_map_static_dimer.png`
- `bond_map_equal_rvb.png`
- `bond_map_weighted_rvb.png`
- `bond_map_weighted_hva_p1.png`
- `bond_map_weighted_hva_p2.png`
- `bond_map_exact.png`
- `bond_map_error_p2_vs_exact.png`
- `energy_vs_hva_depth.png`
- `error_vs_hva_depth.png`
- `fidelity_vs_hva_depth.png`
- `magnetization_vs_hva_depth.png`
- `entropy_vs_hva_depth.png`
- `calibration_energy_vs_fidelity.png`
- `calibration_energy_vs_fidelity_zoom.png`
- `one_page_summary.png`

`one_page_summary.png` uses error vs exact in its main comparison panel so the
RVB-to-HVA improvement is visible on the presentation scale.
`calibration_energy_vs_fidelity.png` reads all rows from
`results/19site_calibration_scan.csv`, not just the best-row comparison table,
and uses a symlog x-axis. `calibration_energy_vs_fidelity_zoom.png` shows the
high-fidelity low-error region on a linear scale.

Spin-structure factor is not included because reliable geometric coordinates
for this extracted 19-site patch are not part of the input data. The bond maps
use a deterministic graph layout instead of fake physical coordinates.

## RVB Notes

The 54 dimer coverings are enumerated deterministically. Equal positive RVB-54
is poor because relative signs and amplitudes of dimer coverings are crucial:
the optimized signed/weighted RVB state dramatically improves energy and
fidelity.

All dimers use:

```text
|s_ij> = (|0_i 1_j> - |1_i 0_j>) / sqrt(2), with i < j
```

The optimized coefficients depend on this gauge choice, while the physical
state does not. The 54-covering overlap matrix is linearly dependent; with
threshold `1e-10`, the stable rank is 36 and the condition number is about
10.52. Do not overinterpret individual coefficients.

The weighted-RVB state is a classical variational/reference initializer, not a
hardware-preparable quantum circuit. The Heisenberg-HVA layers are
circuit-compatible refinements.

Authorship and citation metadata in `CITATION.cff` should be finalized with the
QIntern mentor/team before public release.

## Circuit-Preparation Status

| Object | How it is obtained | Hardware-preparable? |
| --- | --- | --- |
| Weighted RVB-54 | Classical generalized eigenproblem | Not yet |
| HVA p=1,p=2,p=3,p=4 | Heisenberg-HVA circuit refinement | In principle yes |
| Calibration scan state | Exact ground state of modified Hamiltonian | Not yet |
| Hardware-ready QSL state | Not shown | No |

## Runtime Expectations

- Tests: seconds.
- Small patch: seconds.
- Final summary from cached diagnostics/results: under 1 minute on the tested machine.
- p=2 HVA recomputation: about 1 minute on the tested machine.
- p=4 serious HVA run: potentially hours depending on hardware.
- Repeated-optimizer robustness campaign: not included in this frozen package.

For stable performance, set `OPENBLAS_NUM_THREADS=1` and the related thread
variables shown above.

## Limitations

- 19-site result is finite-size only.
- Patch is open-boundary and in total `Sz = 1/2`.
- Weighted RVB is classical, not yet hardware-preparable.
- HVA improvement is real but modest through p=4.
- Current no-calibration HVA p=4 is not yet competitive with the best small
  calibrated-Hamiltonian exact reference in this finite-size energy/fidelity
  comparison.
- No statistical robustness claim is made from repeated optimizer campaigns.
- No noise model or hardware execution is included.
- No claim of scalable QSL preparation is made.
