param(
    [switch]$Serious,
    [switch]$RecomputeHva,
    [string]$HvaCsv = ""
)

$ErrorActionPreference = "Stop"

if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "1" }
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "1" }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = "1" }
if (-not $env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS = "1" }

python scripts/run_small_patch.py
if ($RecomputeHva) {
    python scripts/run_19site_rvb_subspace.py --coverings 54 --checkpoints 1,2,8,12,16,24,32,54 --covering-mode deterministic
    python scripts/run_19site_heisenberg_hva_fast.py --init weighted --max-layers 2 --method Nelder-Mead --maxiter 35 --restarts 0 --tag p2_reference
    python scripts/run_19site_hva_gradients.py
}
python scripts/run_19site_diagnostics.py
if ($HvaCsv) {
    python scripts/build_final_summary.py --use-cached-diagnostics --hva-csv $HvaCsv
} else {
    python scripts/build_final_summary.py --use-cached-diagnostics
}
python scripts/make_figures.py

if ($Serious) {
    python scripts/run_19site_heisenberg_hva_fast.py --init weighted --max-layers 4 --method Nelder-Mead --maxiter 800 --restarts 6 --start-scale 0.08 --auto-tag
    python scripts/run_19site_heisenberg_hva_fast.py --init weighted --max-layers 4 --method Powell --maxiter 800 --restarts 6 --start-scale 0.08 --bounded --bound 0.15 --auto-tag
    python scripts/run_19site_calibration_scan.py --scan-mode group --all-groups --jprimes 1.00,1.02,1.04,1.05,1.06,1.08,1.10,1.15,1.20,1.25,1.30,1.40,1.50,1.75,1.95,2.10,2.30 --append
    python scripts/run_19site_calibration_scan.py --scan-mode bond --all-bonds --jprimes 1.00,1.02,1.04,1.06,1.08,1.10,1.15,1.20 --append
    python scripts/run_19site_calibration_scan.py --scan-mode triangle --all-triangles --jprimes 1.00,1.02,1.04,1.06,1.08,1.10,1.15,1.20 --append
    python scripts/run_19site_diagnostics.py
    python scripts/build_final_summary.py --use-cached-diagnostics
    python scripts/make_figures.py
}
