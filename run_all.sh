#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

HVA_CSV="${HVA_CSV:-}"
SERIOUS=0
RECOMPUTE_HVA=0
for arg in "$@"; do
  case "$arg" in
    --serious)
      SERIOUS=1
      ;;
    --recompute-hva)
      RECOMPUTE_HVA=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

python scripts/run_small_patch.py
if [[ "$RECOMPUTE_HVA" == "1" ]]; then
  python scripts/run_19site_rvb_subspace.py --coverings 54 --checkpoints 1,2,8,12,16,24,32,54 --covering-mode deterministic
  python scripts/run_19site_heisenberg_hva_fast.py --init weighted --max-layers 2 --method Nelder-Mead --maxiter 35 --restarts 0 --tag p2_reference
  python scripts/run_19site_hva_gradients.py
fi
python scripts/run_19site_diagnostics.py
if [[ -n "$HVA_CSV" ]]; then
  python scripts/build_final_summary.py --use-cached-diagnostics --hva-csv "$HVA_CSV"
else
  python scripts/build_final_summary.py --use-cached-diagnostics
fi
python scripts/make_figures.py

if [[ "$SERIOUS" == "1" ]]; then
  python scripts/run_19site_heisenberg_hva_fast.py --init weighted --max-layers 4 --method Nelder-Mead --maxiter 800 --restarts 6 --start-scale 0.08 --auto-tag
  python scripts/run_19site_heisenberg_hva_fast.py --init weighted --max-layers 4 --method Powell --maxiter 800 --restarts 6 --start-scale 0.08 --bounded --bound 0.15 --auto-tag
  python scripts/run_19site_calibration_scan.py --scan-mode group --all-groups --jprimes 1.00,1.02,1.04,1.05,1.06,1.08,1.10,1.15,1.20,1.25,1.30,1.40,1.50,1.75,1.95,2.10,2.30 --append
  python scripts/run_19site_calibration_scan.py --scan-mode bond --all-bonds --jprimes 1.00,1.02,1.04,1.06,1.08,1.10,1.15,1.20 --append
  python scripts/run_19site_calibration_scan.py --scan-mode triangle --all-triangles --jprimes 1.00,1.02,1.04,1.06,1.08,1.10,1.15,1.20 --append
  python scripts/run_19site_diagnostics.py
  python scripts/build_final_summary.py --use-cached-diagnostics
  python scripts/make_figures.py
fi
