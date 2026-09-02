#!/usr/bin/env bash
# Start one or more DyypHoldem continual resolvers against Slumbot's public API.
# Each session is an independent process with its own Slumbot token, seed, and
# artifact directory: runs/play-ui/RUN_NAME/session-<i>/.
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 RUN_NAME TOTAL_HANDS SEED [SESSIONS]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="$1"
TOTAL_HANDS="$2"
SEED="$3"
SESSIONS="${4:-1}"
RUN_DIR="$PROJECT_DIR/runs/play-ui/$RUN_NAME"

case "$RUN_NAME" in
  *[!A-Za-z0-9._-]*|'') echo "invalid run name" >&2; exit 2 ;;
esac
[[ "$TOTAL_HANDS" =~ ^[0-9]+$ ]] && [ "$TOTAL_HANDS" -gt 0 ] || { echo "invalid hands" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "invalid seed" >&2; exit 2; }
[[ "$SESSIONS" =~ ^[0-9]+$ ]] && [ "$SESSIONS" -ge 1 ] && [ "$SESSIONS" -le 16 ] || { echo "invalid sessions" >&2; exit 2; }
[ $(( TOTAL_HANDS % SESSIONS )) -eq 0 ] || { echo "hands must divide evenly across sessions" >&2; exit 2; }
HANDS_PER_SESSION=$(( TOTAL_HANDS / SESSIONS ))

mkdir -p "$RUN_DIR" /root/logs
chmod 700 "$RUN_DIR"

for pid_file in "$RUN_DIR"/*.pid "$RUN_DIR"/session-*/bot.pid; do
  [ -e "$pid_file" ] || continue
  old_pid="$(tr -cd '0-9' < "$pid_file")"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "session process already running: $pid_file" >&2
    exit 1
  fi
done

cd "$PROJECT_DIR/src"
for index in $(seq 0 $(( SESSIONS - 1 ))); do
  SESSION_DIR="$RUN_DIR/session-$index"
  mkdir -p "$SESSION_DIR"
  chmod 700 "$SESSION_DIR"
  setsid nohup python3 player/dyypholdem_slumbot_player.py "$HANDS_PER_SESSION" \
    --seed "$(( SEED + index ))" \
    --telemetry "$SESSION_DIR/decisions.jsonl" \
    --report "$SESSION_DIR/timing_report.json" \
    --text-report "$SESSION_DIR/timing_report.txt" \
    --events "$SESSION_DIR/slumbot-events.jsonl" \
    --summary "$SESSION_DIR/slumbot-summary.json" \
    >"$SESSION_DIR/bot.log" 2>&1 < /dev/null &
  echo "$!" > "$SESSION_DIR/bot.pid"
done

cat > "$RUN_DIR/environment.json" <<EOF2
{
  "run_name": "$RUN_NAME",
  "hands": $TOTAL_HANDS,
  "hands_per_session": $HANDS_PER_SESSION,
  "sessions": $SESSIONS,
  "seed": $SEED,
  "opponent": "slumbot",
  "opponent_host": "slumbot.com"
}
EOF2

echo "SLUMBOT_STARTED run=$RUN_NAME sessions=$SESSIONS hands_per_session=$HANDS_PER_SESSION"
