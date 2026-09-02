#!/usr/bin/env bash
# Start the ACPC dealer, the real DyypHoldem player publishing its full-hand
# strategy, and the local best-response opponent. Artifacts live under
# runs/play-ui/RUN_NAME/session-0/ so the headless controller machinery
# built for concurrent Slumbot sessions can monitor a single LBR match.
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
SESSION_DIR="$RUN_DIR/session-0"
LBR_RAISE_MENU="${DYYPHOLDEM_LBR_RAISE_MENU:-pot,all_in}"
# LBR rebuilds a terminal-equity matrix for every new public board, which on
# the flop averages 1,081 river runouts. On CPU that measured ~15 s per new
# flop against the bot's 2.7 s decision; the pod's GPU is mostly idle while
# LBR thinks, so "cuda" is available for future runs.
LBR_DEVICE="${DYYPHOLDEM_LBR_DEVICE:-cpu}"
case "$LBR_DEVICE" in
  cpu|cuda) ;;
  *) echo "DYYPHOLDEM_LBR_DEVICE must be cpu or cuda" >&2; exit 2 ;;
esac

case "$RUN_NAME" in
  *[!A-Za-z0-9._-]*|'') echo "invalid run name" >&2; exit 2 ;;
esac
[[ "$HANDS" =~ ^[0-9]+$ ]] && [ "$HANDS" -gt 0 ] || { echo "invalid hands" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "invalid seed" >&2; exit 2; }
case "$LBR_RAISE_MENU" in
  *[!a-z_,]*) echo "invalid LBR raise menu" >&2; exit 2 ;;
esac

mkdir -p "$SESSION_DIR" /root/logs
chmod 700 "$RUN_DIR" "$SESSION_DIR"

for pid_file in "$RUN_DIR"/*.pid "$SESSION_DIR"/*.pid; do
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
  "$HANDS" "$SEED" LBR DyypHoldem -p 18901,18902 \
  --t_per_hand 600000 --start_timeout 600000 \
  >"$RUN_DIR/dealer.stdout.log" 2>"$RUN_DIR/dealer.stderr.log" < /dev/null &
echo "$!" > "$RUN_DIR/dealer.pid"

cd "$PROJECT_DIR/src"
# The LBR opponent connects first so the bot's seat (18902) is the one still open.
DYYPHOLDEM_DEVICE="$LBR_DEVICE" setsid nohup python3 player/lbr_acpc_player.py 127.0.0.1 18901 \
  --hands "$HANDS" \
  --strategy-channel "$SESSION_DIR/bot-strategy.jsonl" \
  --raise-menu "$LBR_RAISE_MENU" \
  --events "$SESSION_DIR/lbr-events.jsonl" \
  --summary "$SESSION_DIR/lbr-summary.json" \
  >"$SESSION_DIR/lbr.log" 2>&1 < /dev/null &
echo "$!" > "$SESSION_DIR/autoplay.pid"

setsid nohup python3 player/dyypholdem_acpc_player.py 127.0.0.1 18902 \
  --seed "$SEED" \
  --telemetry "$SESSION_DIR/decisions.jsonl" \
  --report "$SESSION_DIR/timing_report.json" \
  --text-report "$SESSION_DIR/timing_report.txt" \
  --strategy-channel "$SESSION_DIR/bot-strategy.jsonl" \
  >"$SESSION_DIR/bot.log" 2>&1 < /dev/null &
echo "$!" > "$SESSION_DIR/bot.pid"

cat > "$RUN_DIR/environment.json" <<EOF2
{
  "run_name": "$RUN_NAME",
  "hands": $HANDS,
  "hands_per_session": $HANDS,
  "sessions": 1,
  "seed": $SEED,
  "opponent": "lbr",
  "lbr_raise_menu": "$LBR_RAISE_MENU",
  "lbr_device": "$LBR_DEVICE",
  "dealer_ports": [18901, 18902]
}
EOF2

echo "LBR_STARTED run=$RUN_NAME hands=$HANDS raise_menu=$LBR_RAISE_MENU"
