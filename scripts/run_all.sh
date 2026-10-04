#!/bin/sh
# Reproduce every result and figure. Run from the repository root.
set -e
python src/benchmark.py
python src/adapt.py
python src/deeper.py
python src/robust.py
python src/figures_decoding.py
python src/figures_imagery.py
python src/animate.py P1
