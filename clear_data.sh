#!/bin/bash

# Clear script for LLM_API development data
# Clears: prompts.log, scratch directory, sessions directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

echo "=== LLM_API Data Cleanup ==="
echo ""

# Clear prompts.log
PROMPTS_LOG="$DATA_DIR/logs/prompts.log"
if [ -f "$PROMPTS_LOG" ]; then
    > "$PROMPTS_LOG"
    echo "[OK] Cleared prompts.log"
else
    echo "[--] prompts.log not found (skipped)"
fi

# Clear scratch directory (remove contents, keep directory)
SCRATCH_DIR="$DATA_DIR/scratch"
if [ -d "$SCRATCH_DIR" ]; then
    rm -rf "$SCRATCH_DIR"/*
    rm -rf "$SCRATCH_DIR"/.[!.]*  # Hidden files/folders
    echo "[OK] Cleared scratch directory"
else
    mkdir -p "$SCRATCH_DIR"
    echo "[--] scratch directory created (was missing)"
fi

# Clear sessions directory (remove contents, keep directory)
SESSIONS_DIR="$DATA_DIR/sessions"
if [ -d "$SESSIONS_DIR" ]; then
    rm -rf "$SESSIONS_DIR"/*
    rm -rf "$SESSIONS_DIR"/.[!.]*  # Hidden files/folders
    echo "[OK] Cleared sessions directory"
else
    mkdir -p "$SESSIONS_DIR"
    echo "[--] sessions directory created (was missing)"
fi

echo ""
echo "=== Cleanup Complete ==="
