#!/bin/bash
# LiveTranslate - Linux launcher
cd "$(dirname "$0")"
source .venv/bin/activate
python3 main.py "$@"
