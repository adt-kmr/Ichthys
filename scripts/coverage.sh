#!/usr/bin/env bash
set -euo pipefail
python -m pytest tests/ --tb=short -q --cov=utils --cov=viz