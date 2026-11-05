#!/usr/bin/env bash
# PreToolUse: block edits to run artifacts, build output and binary meshes.
f=$(python3 -c 'import json,sys;print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')
case "$f" in
  */runs/*|*.egg-info/*|*/envs/assets/meshes/*)
    echo "Blocked: $f is a generated/binary artifact (runs/, egg-info, meshes). Edit robot.xml or code instead." >&2
    exit 2;;
esac
exit 0
