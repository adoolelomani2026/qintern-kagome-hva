# Reports

This folder contains two report artifacts:

- `report_1_initial_report.pdf`: Report 1 / initial progress report. This is
  the file to send for an early project update, potential solution,
  preliminary evidence, figure interpretation, references, and next-step plan.
- `kagome_hva_report.pdf`: longer paper-style draft with full details, tables,
  figures, and citations.

Build Report 1 from the project root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/report_1_initial_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/report_1_initial_report.tex
```

Build the longer paper-style draft from the project root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/kagome_hva_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/kagome_hva_report.tex
```

`latexmk` also works when Perl is installed:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=report report/kagome_hva_report.tex
```

The report uses images from organized subfolders under `figures/` via relative
paths and does not require copying figure files into this directory.
