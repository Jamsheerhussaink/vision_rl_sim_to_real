#!/usr/bin/env bash
# PostToolUse: ruff-format edited .py files. (Tests are slow - run /reward-audit instead.)
cd "$CLAUDE_PROJECT_DIR" || exit 0
f=$(python3 -c 'import json,sys;print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')
case "$f" in *.py) ruff format -q "$f" 2>/dev/null;; esac
exit 0
