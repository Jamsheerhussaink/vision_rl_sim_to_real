#!/usr/bin/env bash
# Record IK-teacher pick-and-place examples for imitation learning.
# The cube start moves within +/-1 cm (the limit staged_task.py allows) so the copied policy
# sees small variations around the fixed position.
#
# Run only when RL training is stopped (the GPU must be free):
#   scripts/record_teacher_demos.sh            # 30 attempts
#   scripts/record_teacher_demos.sh 3          # quick test: first 3 attempts only
set -u
cd "$(dirname "$0")/.."

COUNT="${1:-30}"
OUT_DIR="outputs/four_stage"
PREFIX="teacher_fixed"
OFFSETS=(-0.01 -0.005 0.0 0.005 0.01)

# Everything printed below is also saved here, so it can be watched live with: tail -f <log file>
LOG_DIR="logs/teacher_demos"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/record_$(date +%Y-%m-%d_%H-%M-%S).log"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log file: $LOG_FILE"

if pgrep -f "scripts/train.py" >/dev/null; then
    echo "RL training is still running. Stop it first (Ctrl+C in its window)." >&2
    exit 1
fi

# Attempt 1 is the exact fixed position; then every x/y offset pair; then repeats of the center.
attempts=("0.0 0.0 101")
for x in "${OFFSETS[@]}"; do
    for y in "${OFFSETS[@]}"; do
        attempts+=("$x $y $((200 + ${#attempts[@]}))")
    done
done
for s in 1 2 3 4 5; do
    attempts+=("0.0 0.0 $((300 + s))")
done

n=0
for a in "${attempts[@]}"; do
    n=$((n + 1))
    [ "$n" -gt "$COUNT" ] && break
    read -r ox oy seed <<<"$a"
    name="$(printf '%s_%03d' "$PREFIX" "$n")"
    echo "=== attempt $n/$COUNT  offset x=$ox y=$oy  seed=$seed  -> $name ==="
    conda run --no-capture-output -n env_isaaclab python scripts/staged_task.py \
        --mode ik-teacher --headless --max-steps 1350 --record-demo \
        --lift-rise-min 0.10 --lift-rise-max 0.12 --lift-hold-seconds 0.5 \
        --cube-offset-x "$ox" --cube-offset-y "$oy" --seed "$seed" \
        --output "$OUT_DIR/$name.json" || echo "attempt $n crashed"
done

echo
echo "=== summary ==="
python3 - "$OUT_DIR" "$PREFIX" <<'EOF'
import glob, json, sys
d, prefix = sys.argv[1:3]
ok = tot = 0
for f in sorted(glob.glob(f"{d}/{prefix}_*.json")):
    r = json.load(open(f))
    tot += 1
    ok += bool(r.get("success"))
    print(f"{f.split('/')[-1]:32s} success={r.get('success')} end={r.get('end_reason')} steps={r.get('steps')}")
print(f"\n{ok} of {tot} teacher attempts completed the full pick-and-place")
EOF
