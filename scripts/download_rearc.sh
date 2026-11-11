#!/usr/bin/env bash
# Fetch re-arc (Hodel 2024) and unzip its 1000 pre-generated examples per training task (~720 MB unzipped).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
[ -d data/re-arc ] || git clone --depth 1 https://github.com/michaelhodel/re-arc.git data/re-arc
cd data/re-arc
[ -d re_arc/tasks ] || unzip -q re_arc.zip -x "__MACOSX/*"
echo "re-arc tasks: $(ls re_arc/tasks | wc -l)"
