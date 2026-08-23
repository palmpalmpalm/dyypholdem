#!/usr/bin/env bash
# Start the dealer, authenticated human web seat, and real DyypHoldem player.
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 RUN_NAME HANDS SEED TOKEN_FILE" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="$1"
HANDS="$2"
SEED="$3"
TOKEN_FILE="$4"
RUN_DIR="$PROJECT_DIR/runs/play-ui/$RUN_NAME"

case "$RUN_NAME" in
  *[!A-Za-z0-9._-]*|'') echo "invalid run name" >&2; exit 2 ;;
esac
[[ "$HANDS" =~ ^[0-9]+$ ]] && [ "$HANDS" -gt 0 ] || { echo "invalid hands" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "invalid seed" >&2; exit 2; }
[ -s "$TOKEN_FILE" ] || { echo "missing session token" >&2; exit 2; }

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

cd "$PROJECT_DIR/acpc_server"
ln -sfn "$RUN_DIR/$RUN_NAME.log" "$PROJECT_DIR/acpc_server/$RUN_NAME.log"
ln -sfn "$RUN_DIR/$RUN_NAME.tlog" "$PROJECT_DIR/acpc_server/$RUN_NAME.tlog"
setsid nohup ./dealer "$RUN_NAME" holdem.nolimit.2p.reverse_blinds.game \
  "$HANDS" "$SEED" Hero DyypHoldem -p 18901,18902 \
  --t_per_hand 600000 --start_timeout 600000 \
  >"$RUN_DIR/dealer.stdout.log" 2>"$RUN_DIR/dealer.stderr.log" < /dev/null &
echo "$!" > "$RUN_DIR/dealer.pid"

cd "$PROJECT_DIR/src"
setsid nohup python3 player/web_acpc_player.py \
  --host 0.0.0.0 --port 8000 \
  --dealer-host 127.0.0.1 --dealer-port 18901 \
  --token-file "$TOKEN_FILE" \
  --events "$RUN_DIR/safe-events.jsonl" \
  --report "$RUN_DIR/timing_report.json" \
  >"$RUN_DIR/ui.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/ui.pid"

ui_ready=0
for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/healthz >/dev/null; then
    ui_ready=1
    break
  fi
  sleep 1
done
[ "$ui_ready" = 1 ] || { echo "web UI did not become healthy" >&2; exit 1; }

setsid nohup python3 player/dyypholdem_acpc_player.py 127.0.0.1 18902 \
  --telemetry "$RUN_DIR/decisions.jsonl" \
  --report "$RUN_DIR/timing_report.json" \
  --text-report "$RUN_DIR/timing_report.txt" \
  >"$RUN_DIR/bot.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/bot.pid"

cat > "$RUN_DIR/environment.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "hands": $HANDS,
  "seed": $SEED,
  "http_port": 8000,
  "dealer_ports": [18901, 18902]
}
EOF

echo "PLAY_UI_STARTED run=$RUN_NAME"
