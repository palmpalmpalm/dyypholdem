#!/usr/bin/env bash
# Start the real DyypHoldem continual resolver against Slumbot's public API.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 RUN_NAME HANDS SEED" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="$1"
HANDS="$2"
SEED="$3"
RUN_DIR="$PROJECT_DIR/runs/play-ui/$RUN_NAME"

case "$RUN_NAME" in
  *[!A-Za-z0-9._-]*|'') echo "invalid run name" >&2; exit 2 ;;
esac
[[ "$HANDS" =~ ^[0-9]+$ ]] && [ "$HANDS" -gt 0 ] || { echo "invalid hands" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "invalid seed" >&2; exit 2; }

mkdir -p "$RUN_DIR" /root/logs
chmod 700 "$RUN_DIR"

for pid_file in "$RUN_DIR"/*.pid; do
  [ -e "$pid_file" ] || continue
  old_pid="$(tr -cd '0-9' < "$pid_file")"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "session process already running: $pid_file" >&2
    exit 1
  fi
done

cd "$PROJECT_DIR/src"
setsid nohup python3 player/dyypholdem_slumbot_player.py "$HANDS" \
  --seed "$SEED" \
  --telemetry "$RUN_DIR/decisions.jsonl" \
  --report "$RUN_DIR/timing_report.json" \
  --text-report "$RUN_DIR/timing_report.txt" \
  --events "$RUN_DIR/slumbot-events.jsonl" \
  --summary "$RUN_DIR/slumbot-summary.json" \
  >"$RUN_DIR/bot.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/bot.pid"

cat > "$RUN_DIR/environment.json" <<EOF2
{
  "run_name": "$RUN_NAME",
  "hands": $HANDS,
  "seed": $SEED,
  "opponent": "slumbot",
  "opponent_host": "slumbot.com"
}
EOF2

echo "SLUMBOT_STARTED run=$RUN_NAME"
