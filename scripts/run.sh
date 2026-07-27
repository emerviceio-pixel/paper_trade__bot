#!/bin/bash

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Run the bot
cd "$PROJECT_ROOT"
python3 src/main.py

# Deactivate on exit
deactivate

#chmod +x scripts/run.sh to deactivate the script and make it executable.