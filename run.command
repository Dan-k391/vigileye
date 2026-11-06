#!/bin/zsh
# Double-click to start VigilEye in Terminal, which holds the camera permission.
cd "$(dirname "$0")"
.venv/bin/python -m vigileye
