#!/usr/bin/env bash
# ConceptARC (Moskvichev et al., 2023): 16 concept groups x 10 tasks, 3 test inputs each.
# Flattened into data/conceptarc/<Concept><n>.json so `--split conceptarc` (SPLIT=conceptarc) loads it.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
[ -d data/ConceptARC ] || git clone --depth 1 https://github.com/victorvikram/ConceptARC.git data/ConceptARC
mkdir -p data/conceptarc
cp data/ConceptARC/corpus/*/*.json data/conceptarc/
echo "conceptarc: $(ls data/conceptarc | wc -l) tasks"
