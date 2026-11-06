#!/bin/bash
# Regenerate every figure and number from the raw results, then rebuild the PDF.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
# Prefer the project's virtual environment; set PYTHON to use another interpreter.
for candidate in "$ROOT/.venv/bin/python" "$ROOT/.venv/Scripts/python.exe"; do
    if [ -z "$PYTHON" ] && [ -x "$candidate" ]; then PYTHON="$candidate"; fi
done
PY="${PYTHON:-python3}"
cd "$ROOT/src" && "$PY" figures.py > /dev/null
cd ../paper
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
echo "built: $(pdfinfo main.pdf | grep Pages)"
MISSING=$("$PY" - <<'PY'
import json, re, os
nums = set(json.load(open('../results/numbers.json')).keys())
need = set()
for f in ['main.tex', 'appendix.tex']:
    need |= set(re.findall(r'\\([A-Z][A-Za-z]*)\{\}', open(f).read()))
print(len(need - nums), 'macros still unresolved:', ' '.join(sorted(need - nums)) or 'none')
PY
)
echo "$MISSING"
