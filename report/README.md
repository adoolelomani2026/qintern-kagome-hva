# Report

This folder contains the paper-style LaTeX report for the frozen 19-site
weighted-RVB + Heisenberg-HVA proof-of-concept.

Build from the project root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/kagome_hva_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/kagome_hva_report.tex
```

`latexmk` also works when Perl is installed:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=report report/kagome_hva_report.tex
```

The report uses images from `figures/` via relative paths and does not require
copying figure files into this directory.
