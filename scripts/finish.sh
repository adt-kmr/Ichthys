#!/usr/bin/env bash
set -euo pipefail
echo "Ichthys dashboard commit count: $(git log --oneline | wc -l)"