#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp -n backend/.env.example backend/.env || true
python backend/app.py
