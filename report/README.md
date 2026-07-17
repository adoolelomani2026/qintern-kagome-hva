# Reports

This folder contains the Report 1 / initial progress report:

- `report_1_initial_report.pdf`: Report 1 / initial progress report. This is
  the file to send for an early project update, potential solution,
  preliminary evidence, figure interpretation, references, and next-step plan.

Build Report 1 from the project root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/report_1_initial_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/report_1_initial_report.tex
```

`latexmk` also works when Perl is installed:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=report report/report_1_initial_report.tex
```

The report uses images from organized subfolders under `figures/` via relative
paths and does not require copying figure files into this directory.
