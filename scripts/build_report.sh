#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../report"
mkdir -p ../output/pdf
xelatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf main.tex
cd ../output/pdf
# BibTeX resolves the source bibliography through BIBINPUTS.
BIBINPUTS="../../report:" bibtex main
cd ../../report
xelatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf main.tex
cp ../output/pdf/main.pdf ../output/pdf/Muon_Report.pdf
