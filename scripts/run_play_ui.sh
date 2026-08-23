#!/usr/bin/env bash
# Guarded live RTX 4090 session with periodic telemetry copyback.
set -euo pipefail

COMMAND="${1:-start}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_HELPER="$PROJECT_DIR/scripts/runpod_ui_pod.py"
WATCHDOG_HELPER="$PROJECT_DIR/scripts/run_play_ui_watchdog.sh"
ENV_FILE="${DYYPHOLDEM_ENV_FILE:-/Users/palm/Documents/poker-supremus/.env.local}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"
SESSION_ROOT="$PROJECT_DIR/runs/play-ui"
CURRENT_MANIFEST="$SESSION_ROOT/current.json"
LAUNCH_LOCK_DIR="$SESSION_ROOT/start.lock"
GUARD_SECONDS="${DYYPHOLDEM_UI_GUARD_SECONDS:-3600}"
GPU_TYPE="${DYYPHOLDEM_GPU_TYPE:-NVIDIA GeForce RTX 4090}"
CLOUD_TYPE="${DYYPHOLDEM_GPU_CLOUD_TYPE:-SECURE}"
HANDS="${DYYPHOLDEM_UI_HANDS:-100}"
SEED="${DYYPHOLDEM_UI_SEED:-20260823}"
MODEL_ROOT="${DYYPHOLDEM_COMPACT_MODEL_PATH:-$PROJECT_DIR/runs/model-recovery/compact}"
HTTP_PORT=8000
FINALIZE_MARGIN_SECONDS=90
MATCH_COMPLETE_GRACE_SECONDS=60

POD_ID=""
POD_NAME=""
RUN_NAME=""
PUBLIC_URL=""
AUTHENTICATED_URL=""
COST_PER_HOUR="unknown"
LOCAL_RUN_DIR=""
SSH_CONFIG=""
REMOTE_READY=0
ABSOLUTE_STOP_EPOCH=""
SESSION_DEADLINE_EPOCH=""
WATCHDOG_PID=""
WATCHDOG_CONFIG=""
WATCHDOG_CANCEL_FILE=""
WATCHDOG_STATUS_FILE=""
WATCHDOG_OWNED=0
LOCK_HELD=0
COPY_STATUS="not_started"
LAST_COPY_AT=""
TERMINATION_VERIFIED=false
FINAL_REASON=""
MANIFEST_STATUS=""
POD_ACQUIRED=0
ACQUISITION_ATTEMPTED=0

usage() {
  printf '%s\n' "usage: $0 [start|stop|status|logs|dry-run]"
}

json_field() {
  "$LOCAL_PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1])).get(sys.argv[2], ""); print("" if value is None else value)' "$1" "$2"
}

validate_uint() {
  label="$1"
  value="$2"
  minimum="$3"
  maximum="$4"
  case "$value" in
    *[!0-9]*|'') echo "$label must be an integer from $minimum through $maximum" >&2; return 1 ;;
  esac
  [ "$value" -ge "$minimum" ] && [ "$value" -le "$maximum" ] || {
    echo "$label must be an integer from $minimum through $maximum" >&2
    return 1
  }
}

validate_config() {
  validate_uint DYYPHOLDEM_UI_GUARD_SECONDS "$GUARD_SECONDS" 900 14400
  validate_uint DYYPHOLDEM_UI_HANDS "$HANDS" 1 1000
  validate_uint DYYPHOLDEM_UI_SEED "$SEED" 0 2147483647
  [ "$CLOUD_TYPE" = "SECURE" ] || [ "$CLOUD_TYPE" = "COMMUNITY" ] || {
    echo "DYYPHOLDEM_GPU_CLOUD_TYPE must be SECURE or COMMUNITY" >&2
    return 1
  }
  [ -n "$GPU_TYPE" ] || { echo "DYYPHOLDEM_GPU_TYPE must not be empty" >&2; return 1; }
}

validate_pod_identity() {
  case "$POD_ID" in *[!A-Za-z0-9]*|'') echo "invalid RunPod pod id" >&2; return 1 ;; esac
  case "$POD_NAME" in *[!A-Za-z0-9._-]*|'') echo "invalid RunPod pod name" >&2; return 1 ;; esac
}

load_credentials() {
  [ -f "$ENV_FILE" ] || { echo "missing ignored RunPod credential file: $ENV_FILE" >&2; return 1; }
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
  [ -n "${RUNPOD_API_KEY:-}" ] || { echo "RUNPOD_API_KEY is missing from $ENV_FILE" >&2; return 1; }
}

verify_models() {
  for relative in preflop-aux/final_compact.pt flop/final_compact.pt turn/final_compact.pt river/final_compact.pt; do
    [ -s "$MODEL_ROOT/$relative" ] || { echo "missing compact model: $MODEL_ROOT/$relative" >&2; return 1; }
  done
}

acquire_launch_lock() {
  if mkdir "$LAUNCH_LOCK_DIR" 2>/dev/null; then
    LOCK_HELD=1
  else
    lock_pid=""
    [ -s "$LAUNCH_LOCK_DIR/pid" ] && lock_pid="$(tr -cd '0-9' < "$LAUNCH_LOCK_DIR/pid")"
    if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
      echo "another DyypHoldem UI launch is in progress (pid $lock_pid)" >&2
      return 1
    fi
    [ ! -e "$LAUNCH_LOCK_DIR/pid" ] || unlink "$LAUNCH_LOCK_DIR/pid"
    rmdir "$LAUNCH_LOCK_DIR" 2>/dev/null || {
      echo "could not reclaim stale DyypHoldem launch lock" >&2
      return 1
    }
    mkdir "$LAUNCH_LOCK_DIR" || return 1
    LOCK_HELD=1
  fi
  printf '%s\n' "$$" > "$LAUNCH_LOCK_DIR/pid"
}

release_launch_lock() {
  [ "$LOCK_HELD" = 1 ] || return 0
  lock_pid=""
  [ -s "$LAUNCH_LOCK_DIR/pid" ] && lock_pid="$(tr -cd '0-9' < "$LAUNCH_LOCK_DIR/pid")"
  if [ "$lock_pid" != "$$" ]; then
    echo "refusing to release a launch lock owned by another process" >&2
    return 1
  fi
  unlink "$LAUNCH_LOCK_DIR/pid"
  rmdir "$LAUNCH_LOCK_DIR" 2>/dev/null || true
  LOCK_HELD=0
}

write_manifest() {
  manifest_status="$1"
  MANIFEST_STATUS="$manifest_status"
  "$LOCAL_PYTHON" - "$CURRENT_MANIFEST" "$manifest_status" "$POD_ID" "$POD_NAME" "$RUN_NAME" \
    "$PUBLIC_URL" "$AUTHENTICATED_URL" "$COST_PER_HOUR" "$LOCAL_RUN_DIR" "$SSH_CONFIG" "$$" \
    "$ABSOLUTE_STOP_EPOCH" "$SESSION_DEADLINE_EPOCH" "$WATCHDOG_PID" "$COPY_STATUS" \
    "$LAST_COPY_AT" "$TERMINATION_VERIFIED" "$FINAL_REASON" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

(path, status, pod_id, pod_name, run_name, public_url, authenticated_url,
 cost, local_run_dir, ssh_config, launcher_pid, absolute_stop_epoch,
 session_deadline_epoch, watchdog_pid, copy_status, last_copy_at,
 termination_verified, final_reason) = sys.argv[1:]

def optional_int(value):
    return int(value) if value.isdigit() else None

payload = {
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "pod_id": pod_id,
    "pod_name": pod_name,
    "run_name": run_name,
    "public_url": public_url,
    "authenticated_url": authenticated_url,
    "cost_per_hour": cost,
    "local_run_dir": local_run_dir,
    "ssh_config": ssh_config,
    "launcher_pid": int(launcher_pid),
    "absolute_stop_epoch": optional_int(absolute_stop_epoch),
    "session_deadline_epoch": optional_int(session_deadline_epoch),
    "watchdog_pid": optional_int(watchdog_pid),
    "copy_status": copy_status,
    "last_successful_copy_at": last_copy_at or None,
    "termination_verified": termination_verified == "true",
    "final_reason": final_reason or None,
}
target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.chmod(0o600)
temporary.replace(target)
PY
}

exact_name_ids() {
  list_file="$SESSION_ROOT/.pods-$PPID-$$.json"
  if ! "$LOCAL_PYTHON" "$POD_HELPER" list > "$list_file" 2>/dev/null; then
    return 1
  fi
  "$LOCAL_PYTHON" - "$POD_NAME" "$list_file" <<'PY'
import json
import sys

name, path = sys.argv[1:]
for pod in json.load(open(path)):
    if pod.get("name") == name and pod.get("id"):
        print(pod["id"])
PY
}

verify_exact_name_absent() {
  successful_empty=0
  for _ in $(seq 1 30); do
    if ids="$(exact_name_ids)"; then
      if [ -z "$ids" ]; then
        successful_empty=$((successful_empty + 1))
        [ "$successful_empty" -ge 6 ] && return 0
      else
        successful_empty=0
      fi
    fi
    sleep 2
  done
  return 1
}

terminate_owned_id() {
  target_id="$1"
  case "$target_id" in *[!A-Za-z0-9]*|'') return 1 ;; esac

  for _ in $(seq 1 3); do
    "$LOCAL_PYTHON" "$POD_HELPER" stop --pod-id "$target_id" >/dev/null 2>&1 && break
    sleep 2
  done

  snapshot="$SESSION_ROOT/.exists-$target_id-$$.json"
  for _ in $(seq 1 18); do
    if "$LOCAL_PYTHON" "$POD_HELPER" exists --pod-id "$target_id" > "$snapshot" 2>/dev/null; then
      read -r exists name desired < <("$LOCAL_PYTHON" - "$snapshot" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
pod = value.get("pod") or {}
print(str(bool(value.get("exists"))).lower(), pod.get("name") or "", pod.get("desiredStatus") or "")
PY
)
      [ "$exists" = "false" ] && break
      if [ "$name" != "$POD_NAME" ]; then
        echo "refusing to terminate pod id whose name changed: $target_id" >&2
        return 1
      fi
      [ "$desired" = "EXITED" ] && break
    fi
    sleep 2
  done

  for _ in $(seq 1 3); do
    "$LOCAL_PYTHON" "$POD_HELPER" terminate --pod-id "$target_id" >/dev/null 2>&1 && return 0
    sleep 3
  done
  return 1
}

safe_terminate() {
  [ -n "$POD_NAME" ] || return 0
  listed=0
  owned_ids=""
  for _ in $(seq 1 6); do
    if owned_ids="$(exact_name_ids)"; then
      listed=1
      break
    fi
    sleep 3
  done
  [ "$listed" = 1 ] || return 1

  while IFS= read -r owned_id; do
    [ -n "$owned_id" ] || continue
    terminate_owned_id "$owned_id" || true
  done <<EOF
$owned_ids
EOF
  verify_exact_name_absent
}

copy_back_once() {
  [ "$REMOTE_READY" = 1 ] && [ -s "$SSH_CONFIG" ] || return 2
  mkdir -p "$LOCAL_RUN_DIR"
  if rsync -az --partial --partial-dir=.rsync-partial --timeout=30 \
      -e "ssh -F $SSH_CONFIG -o BatchMode=yes -o ConnectTimeout=10" \
      "dyyui:/root/dyypholdem/runs/play-ui/$RUN_NAME/" \
      "$LOCAL_RUN_DIR/" >>"$LOCAL_RUN_DIR/copyback.log" 2>&1; then
    LAST_COPY_AT="$($LOCAL_PYTHON -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat())')"
    COPY_STATUS="succeeded"
    return 0
  fi
  COPY_STATUS="failed"
  return 1
}

copy_back_final() {
  [ "$REMOTE_READY" = 1 ] || { COPY_STATUS="unavailable_before_remote_ready"; return 2; }
  for attempt in $(seq 1 3); do
    copy_back_once && return 0
    echo "final telemetry copyback failed (attempt $attempt/3)" >&2
    sleep 3
  done
  COPY_STATUS="failed"
  return 1
}

quiesce_remote() {
  [ "$REMOTE_READY" = 1 ] && [ -s "$SSH_CONFIG" ] || return 2
  ssh -n -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=10 dyyui \
    "for pid_file in /root/dyypholdem/runs/play-ui/$RUN_NAME/*.pid; do [ -s \"\$pid_file\" ] || continue; pid=\$(tr -cd '0-9' < \"\$pid_file\"); [ -n \"\$pid\" ] && kill -TERM \"\$pid\" 2>/dev/null || true; done; sleep 2" \
    >/dev/null 2>&1
}

cancel_local_watchdog() {
  [ -n "$WATCHDOG_CANCEL_FILE" ] || return 0
  : > "$WATCHDOG_CANCEL_FILE"
  if [ "$WATCHDOG_OWNED" = 1 ] && [ -n "$WATCHDOG_PID" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
}

arm_local_watchdog() {
  WATCHDOG_CONFIG="$LOCAL_RUN_DIR/watchdog-config.json"
  WATCHDOG_CANCEL_FILE="$LOCAL_RUN_DIR/watchdog-cancelled"
  WATCHDOG_STATUS_FILE="$LOCAL_RUN_DIR/watchdog-status.json"
  "$LOCAL_PYTHON" - "$WATCHDOG_CONFIG" "$POD_ID" "$POD_NAME" "$ABSOLUTE_STOP_EPOCH" \
    "$POD_HELPER" "$WATCHDOG_CANCEL_FILE" "$WATCHDOG_STATUS_FILE" <<'PY'
import json
from pathlib import Path
import sys

path, pod_id, pod_name, deadline, helper, cancel_file, status_file = sys.argv[1:]
target = Path(path)
target.write_text(json.dumps({
    "pod_id": pod_id,
    "pod_name": pod_name,
    "deadline_epoch": int(deadline),
    "pod_helper": helper,
    "cancel_file": cancel_file,
    "status_file": status_file,
}, indent=2, sort_keys=True) + "\n")
target.chmod(0o600)
PY
  nohup "$WATCHDOG_HELPER" "$WATCHDOG_CONFIG" "$ENV_FILE" \
    >"$LOCAL_RUN_DIR/watchdog.log" 2>&1 < /dev/null &
  WATCHDOG_PID=$!
  WATCHDOG_OWNED=1
  sleep 1
  kill -0 "$WATCHDOG_PID" 2>/dev/null || {
    echo "local RunPod deadline watchdog failed to start" >&2
    return 1
  }
}

finalize_session() {
  FINAL_REASON="$1"
  final_rc=0
  if ! quiesce_remote; then
    [ "$REMOTE_READY" = 1 ] && echo "warning: remote processes could not be quiesced before copyback" >&2
  fi
  if ! copy_back_final; then
    [ "$REMOTE_READY" = 1 ] && final_rc=1
  fi
  if safe_terminate; then
    TERMINATION_VERIFIED=true
    cancel_local_watchdog
  else
    TERMINATION_VERIFIED=false
    MANIFEST_STATUS="cleanup_unverified"
    write_manifest "$MANIFEST_STATUS"
    if [ -n "$WATCHDOG_PID" ]; then
      echo "ERROR: could not verify exact-name DyypHoldem pod absence; hard-deadline watchdog remains armed" >&2
    else
      echo "ERROR: could not verify exact-name DyypHoldem pod absence and watchdog arming had not completed" >&2
    fi
    return 1
  fi

  if [ "$COPY_STATUS" = "failed" ]; then
    MANIFEST_STATUS="terminated_copy_failed"
    final_rc=1
  elif [ "$COPY_STATUS" = "unavailable_before_remote_ready" ]; then
    MANIFEST_STATUS="terminated_no_remote_artifacts"
  else
    MANIFEST_STATUS="terminated"
  fi
  write_manifest "$MANIFEST_STATUS"
  return "$final_rc"
}

load_manifest_session() {
  [ -s "$CURRENT_MANIFEST" ] || { echo "no DyypHoldem UI session manifest" >&2; return 1; }
  POD_ID="$(json_field "$CURRENT_MANIFEST" pod_id)"
  POD_NAME="$(json_field "$CURRENT_MANIFEST" pod_name)"
  RUN_NAME="$(json_field "$CURRENT_MANIFEST" run_name)"
  PUBLIC_URL="$(json_field "$CURRENT_MANIFEST" public_url)"
  AUTHENTICATED_URL="$(json_field "$CURRENT_MANIFEST" authenticated_url)"
  COST_PER_HOUR="$(json_field "$CURRENT_MANIFEST" cost_per_hour)"
  LOCAL_RUN_DIR="$(json_field "$CURRENT_MANIFEST" local_run_dir)"
  SSH_CONFIG="$(json_field "$CURRENT_MANIFEST" ssh_config)"
  ABSOLUTE_STOP_EPOCH="$(json_field "$CURRENT_MANIFEST" absolute_stop_epoch)"
  SESSION_DEADLINE_EPOCH="$(json_field "$CURRENT_MANIFEST" session_deadline_epoch)"
  WATCHDOG_PID="$(json_field "$CURRENT_MANIFEST" watchdog_pid)"
  COPY_STATUS="$(json_field "$CURRENT_MANIFEST" copy_status)"
  LAST_COPY_AT="$(json_field "$CURRENT_MANIFEST" last_successful_copy_at)"
  WATCHDOG_CONFIG="$LOCAL_RUN_DIR/watchdog-config.json"
  WATCHDOG_CANCEL_FILE="$LOCAL_RUN_DIR/watchdog-cancelled"
  WATCHDOG_STATUS_FILE="$LOCAL_RUN_DIR/watchdog-status.json"
  [ -s "$SSH_CONFIG" ] && REMOTE_READY=1 || REMOTE_READY=0
  validate_pod_identity
}

fetch_ui_state() {
  state_file="$LOCAL_RUN_DIR/last-ui-state.json"
  curl -fsS --max-time 10 -H "X-Session-Token: $SESSION_TOKEN" \
    "$PUBLIC_URL/api/state" > "$state_file"
  chmod 600 "$state_file"
  "$LOCAL_PYTHON" - "$state_file" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print(value.get("status") or "unknown", int(value.get("hands_completed") or 0))
PY
}

remote_process_state() {
  ssh -n -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=10 dyyui \
    "for role in dealer ui bot; do file=/root/dyypholdem/runs/play-ui/$RUN_NAME/\$role.pid; pid=\$([ -s \"\$file\" ] && tr -cd '0-9' < \"\$file\"); if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then printf '%s=running ' \"\$role\"; else printf '%s=dead ' \"\$role\"; fi; done; echo"
}

if [ "$COMMAND" = "dry-run" ]; then
  validate_config
  printf '%s\n' \
    "DyypHoldem live UI dry run" \
    "  GPU: one $CLOUD_TYPE $GPU_TYPE" \
    "  public service: authenticated HTTPS proxy on port $HTTP_PORT" \
    "  solver: real ACPC dealer + ContinualResolving, 1,000 CFR iterations" \
    "  models: four checksum-verified compact recovered networks" \
    "  telemetry: private JSONL plus safe live/final per-street reports" \
    "  hard guard: $GUARD_SECONDS seconds; authenticated remote stop plus independent local stop/delete watchdog" \
    "  shutdown: quiesce, retry final copyback, exact-name stop/delete, six successful absence checks"
  exit 0
fi

if [ "$COMMAND" = "status" ] || [ "$COMMAND" = "logs" ] || [ "$COMMAND" = "stop" ]; then
  load_manifest_session
  if [ "$COMMAND" = "logs" ]; then
    if [ "$REMOTE_READY" = 1 ]; then
      copy_back_once || { echo "could not refresh logs from the live pod" >&2; exit 1; }
    fi
    [ -s "$LOCAL_RUN_DIR/timing_report.txt" ] && sed -n '1,240p' "$LOCAL_RUN_DIR/timing_report.txt"
    [ -s "$LOCAL_RUN_DIR/safe-events.jsonl" ] && tail -20 "$LOCAL_RUN_DIR/safe-events.jsonl"
    exit 0
  fi
  load_credentials
  if [ "$COMMAND" = "status" ]; then
    "$LOCAL_PYTHON" "$POD_HELPER" exists --pod-id "$POD_ID"
    printf 'authenticatedUrl=%s\nlocalRunDir=%s\ncopyStatus=%s\nlastSuccessfulCopyAt=%s\n' \
      "$AUTHENTICATED_URL" "$LOCAL_RUN_DIR" "$COPY_STATUS" "$LAST_COPY_AT"
    exit 0
  fi
  if finalize_session user_stop; then
    echo "DyypHoldem UI session stopped; exact-name absence verified"
    exit 0
  fi
  exit 1
fi

[ "$COMMAND" = "start" ] || { usage >&2; exit 2; }
validate_config
verify_models
load_credentials
mkdir -p "$SESSION_ROOT"
acquire_launch_lock

# Invoked indirectly by the EXIT/INT/TERM trap below.
# shellcheck disable=SC2329
cleanup() {
  original_rc=$?
  trap - EXIT INT TERM
  set +e
  release_launch_lock
  if [ "$POD_ACQUIRED" = 1 ]; then
    finalize_session "launcher_exit_$original_rc"
    final_rc=$?
    [ "$original_rc" -ne 0 ] || original_rc=$final_rc
  elif [ "$ACQUISITION_ATTEMPTED" = 1 ]; then
    safe_terminate || original_rc=1
  fi
  exit "$original_rc"
}
trap cleanup EXIT INT TERM

if [ -s "$CURRENT_MANIFEST" ]; then
  existing_status="$(json_field "$CURRENT_MANIFEST" status)"
  if [ "$existing_status" = "running" ] || [ "$existing_status" = "starting" ] || [ "$existing_status" = "cleanup_unverified" ]; then
    echo "an existing DyypHoldem UI session is recorded; run status or stop first" >&2
    exit 1
  fi
fi

RUN_NAME="dyypholdem-ui-$(date -u +%Y%m%dT%H%M%SZ)"
POD_NAME="$RUN_NAME-$PPID-$$"
LOCAL_RUN_DIR="$SESSION_ROOT/$RUN_NAME"
mkdir -p "$LOCAL_RUN_DIR"
chmod 700 "$LOCAL_RUN_DIR"
SSH_CONFIG="$LOCAL_RUN_DIR/ssh-config"
CREATE_JSON="$LOCAL_RUN_DIR/create.json"
PODS_JSON="$LOCAL_RUN_DIR/pods.json"
URL_JSON="$LOCAL_RUN_DIR/public-url.json"
TOKEN_FILE="$LOCAL_RUN_DIR/session-token"
SESSION_TOKEN="$($LOCAL_PYTHON -c 'import secrets; print(secrets.token_urlsafe(32))')"
printf '%s\n' "$SESSION_TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"

"$LOCAL_PYTHON" "$POD_HELPER" list > "$PODS_JSON"
existing_ui_pods="$($LOCAL_PYTHON - "$PODS_JSON" <<'PY'
import json
import sys

pods = json.load(open(sys.argv[1]))
print("\n".join(str(pod.get("id") or "") for pod in pods if str(pod.get("name") or "").startswith("dyypholdem-ui-")))
PY
)"
[ -z "$existing_ui_pods" ] || {
  echo "a DyypHoldem UI pod already exists; inspect or stop it before launching another" >&2
  exit 1
}
echo "RunPod preflight complete; no existing pod was changed"

for attempt in $(seq 1 10); do
  ACQUISITION_ATTEMPTED=1
  if "$LOCAL_PYTHON" "$POD_HELPER" create \
      --name "$POD_NAME" --gpu-type "$GPU_TYPE" --cloud-type "$CLOUD_TYPE" \
      --http-port "$HTTP_PORT" > "$CREATE_JSON" 2>"$LOCAL_RUN_DIR/create-error.log"; then
    POD_ID="$(json_field "$CREATE_JSON" id)"
  fi
  if [ -z "$POD_ID" ]; then
    if recovered_ids="$(exact_name_ids)" && [ -n "$recovered_ids" ]; then
      POD_ID="$(printf '%s\n' "$recovered_ids" | head -1)"
      POD_ACQUIRED=1
      recovered_count="$(printf '%s\n' "$recovered_ids" | sed '/^$/d' | wc -l | tr -d ' ')"
      [ "$recovered_count" = 1 ] || {
        echo "multiple exact-name pods appeared during acquisition; aborting to verified cleanup" >&2
        exit 1
      }
    fi
  fi
  [ -n "$POD_ID" ] && POD_ACQUIRED=1
  [ -n "$POD_ID" ] && break
  echo "no $CLOUD_TYPE $GPU_TYPE capacity (attempt $attempt/10)"
  sleep 20
done
[ -n "$POD_ID" ] || { echo "unable to acquire requested RTX 4090" >&2; exit 1; }
validate_pod_identity

# The absolute paid-resource deadline is fixed immediately after acquisition.
ABSOLUTE_STOP_EPOCH=$(( $(date +%s) + GUARD_SECONDS ))
SESSION_DEADLINE_EPOCH=$(( ABSOLUTE_STOP_EPOCH - FINALIZE_MARGIN_SECONDS ))
if [ ! -s "$CREATE_JSON" ] || [ "$(json_field "$CREATE_JSON" id 2>/dev/null || true)" != "$POD_ID" ]; then
  "$LOCAL_PYTHON" "$POD_HELPER" status --pod-id "$POD_ID" > "$CREATE_JSON"
fi
COST_PER_HOUR="$(json_field "$CREATE_JSON" costPerHr)"
FINAL_REASON="setup"
write_manifest starting
arm_local_watchdog
write_manifest starting

"$LOCAL_PYTHON" "$POD_HELPER" public-url --pod-id "$POD_ID" --http-port "$HTTP_PORT" > "$URL_JSON"
PUBLIC_URL="$(json_field "$URL_JSON" url)"
[ -n "$PUBLIC_URL" ] || { echo "RunPod helper returned no public UI URL" >&2; exit 1; }
AUTHENTICATED_URL="$PUBLIC_URL/?token=$SESSION_TOKEN"
write_manifest starting
release_launch_lock
echo "acquired isolated RTX 4090 at \$$COST_PER_HOUR/hour; local hard-deadline watchdog armed"

ssh_ready=0
for _ in $(seq 1 48); do
  if "$LOCAL_PYTHON" "$POD_HELPER" ssh-config --pod-id "$POD_ID" --out "$SSH_CONFIG" >/dev/null 2>&1; then
    ssh_ready=1
    break
  fi
  sleep 5
done
[ "$ssh_ready" = 1 ] || { echo "pod never exposed SSH" >&2; exit 1; }
SSH=(ssh -n -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=15)
SSH_STDIN=(ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=15)

connected=0
for _ in $(seq 1 36); do
  if "${SSH[@]}" dyyui true 2>/dev/null; then connected=1; break; fi
  sleep 5
done
[ "$connected" = 1 ] || { echo "pod SSH never became ready" >&2; exit 1; }

remote_guard_seconds=$(( ABSOLUTE_STOP_EPOCH - $(date +%s) ))
[ "$remote_guard_seconds" -gt 60 ] || { echo "setup left too little time to arm remote guard" >&2; exit 1; }
printf '#!/usr/bin/env bash\nset -eu\nsleep %s\nrunpodctl stop pod %s\n' "$remote_guard_seconds" "$POD_ID" | \
  "${SSH_STDIN[@]}" dyyui 'command -v runpodctl >/dev/null && umask 077 && cat > /root/dyypholdem_self_stop.sh && chmod 700 /root/dyypholdem_self_stop.sh'
remote_guard_command="IFS= read -r RUNPOD_API_KEY && test -n \"\$RUNPOD_API_KEY\" && export RUNPOD_API_KEY && runpodctl get pod >/dev/null && (setsid nohup /root/dyypholdem_self_stop.sh >/root/dyypholdem_self_stop.log 2>&1 < /dev/null & guard=\$!; sleep 1; kill -0 \"\$guard\")"
printf '%s\n' "$RUNPOD_API_KEY" | "${SSH_STDIN[@]}" dyyui "$remote_guard_command"
echo "authenticated remote hard-deadline stop guard armed"

"${SSH[@]}" dyyui 'if command -v rsync >/dev/null && command -v curl >/dev/null; then :; else (apt-get update -qq && apt-get install -y -qq rsync curl) >/dev/null 2>&1 || exit 1; fi; mkdir -p /root/dyypholdem /root/logs'
rsync -az -e "ssh -F $SSH_CONFIG -o BatchMode=yes" \
  --exclude .git --exclude .DS_Store --exclude __pycache__ --exclude runs \
  "$PROJECT_DIR/src" "$PROJECT_DIR/scripts" "$PROJECT_DIR/acpc_server" \
  dyyui:/root/dyypholdem/
"${SSH[@]}" dyyui "python3 -m pip install --quiet gdown loguru; mkdir -p /root/dyypholdem/runs/model-recovery/compact /root/dyypholdem/runs/play-ui/$RUN_NAME"
rsync -az -e "ssh -F $SSH_CONFIG -o BatchMode=yes" "$MODEL_ROOT/" dyyui:/root/dyypholdem/runs/model-recovery/compact/
rsync -az -e "ssh -F $SSH_CONFIG -o BatchMode=yes" "$TOKEN_FILE" dyyui:/root/dyypholdem/session-token
REMOTE_READY=1

gpu_healthy=0
for _ in $(seq 1 18); do
  if "${SSH[@]}" dyyui 'timeout 20s nvidia-smi -L >/dev/null 2>&1 && timeout 20s python3 -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1'; then
    gpu_healthy=1
    break
  fi
  sleep 10
done
[ "$gpu_healthy" = 1 ] || { echo "unhealthy RTX 4090 host" >&2; exit 1; }
"${SSH[@]}" dyyui 'python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"'

echo "downloading and checksum-verifying full play assets"
"${SSH[@]}" dyyui "cd /root/dyypholdem && timeout 1800s python3 scripts/materialize_assets.py --profile play > runs/play-ui/$RUN_NAME/play-assets.json"
echo "validating recovered networks on CUDA"
"${SSH[@]}" dyyui "cd /root/dyypholdem && timeout 600s python3 scripts/model_gpu_validation.py --model-root runs/model-recovery/compact --repeats 2 --source-commit $(git -C "$PROJECT_DIR" rev-parse HEAD) --output runs/play-ui/$RUN_NAME/model-validation.json > runs/play-ui/$RUN_NAME/model-validation.log 2>&1"

echo "starting dealer, authenticated UI, and real continual resolver"
"${SSH[@]}" dyyui "export DYYPHOLDEM_COMPACT_MODEL_PATH=/root/dyypholdem/runs/model-recovery/compact DYYPHOLDEM_SOURCE_COMMIT=$(git -C "$PROJECT_DIR" rev-parse HEAD); cd /root/dyypholdem && ./scripts/start_play_ui_remote.sh '$RUN_NAME' '$HANDS' '$SEED' /root/dyypholdem/session-token"

ai_ready=0
for _ in $(seq 1 120); do
  if "${SSH[@]}" dyyui "test -s /root/dyypholdem/runs/play-ui/$RUN_NAME/timing_report.json && grep -q AI_READY /root/dyypholdem/runs/play-ui/$RUN_NAME/bot.log" 2>/dev/null; then
    ai_ready=1
    break
  fi
  sleep 5
done
[ "$ai_ready" = 1 ] || { echo "DyypHoldem did not finish initialization" >&2; exit 1; }

proxy_ready=0
for _ in $(seq 1 90); do
  if curl -fsS --max-time 10 "$PUBLIC_URL/healthz" >/dev/null 2>&1; then
    proxy_ready=1
    break
  fi
  sleep 4
done
[ "$proxy_ready" = 1 ] || { echo "RunPod HTTPS proxy did not become healthy" >&2; exit 1; }

# AI_READY is emitted before the ACPC socket necessarily receives its first MATCHSTATE.
# Do not publish the URL until the authenticated browser bridge proves real game state.
first_matchstate=0
for _ in $(seq 1 120); do
  if read -r ui_status hands_completed < <(fetch_ui_state 2>/dev/null); then
    case "$ui_status" in
      your_turn|bot_thinking|hand_complete) first_matchstate=1; break ;;
      error) echo "browser bridge entered error before first MATCHSTATE" >&2; exit 1 ;;
      match_complete)
        echo "match completed before a playable browser state (hands=$hands_completed)" >&2
        exit 1
        ;;
    esac
  fi
  sleep 3
done
[ "$first_matchstate" = 1 ] || { echo "browser bridge never received a playable MATCHSTATE" >&2; exit 1; }

if ! copy_back_once; then
  echo "initial telemetry copyback failed" >&2
  exit 1
fi
FINAL_REASON="running"
write_manifest running
remaining_seconds=$(( SESSION_DEADLINE_EPOCH - $(date +%s) ))
[ "$remaining_seconds" -gt 0 ] || { echo "setup exhausted the guarded live interval" >&2; exit 1; }
printf 'PLAY_UI_READY\nPLAY_UI_URL=%s\nLOCAL_RUN_DIR=%s\nHARD_STOP_SECONDS=%s\n' \
  "$AUTHENTICATED_URL" "$LOCAL_RUN_DIR" "$remaining_seconds"

run_result=0
completion_detected_epoch=0
while [ "$(date +%s)" -lt "$SESSION_DEADLINE_EPOCH" ]; do
  if ! copy_back_once; then
    echo "warning: periodic telemetry copyback failed; final copy will retry" >&2
  fi
  write_manifest running

  snapshot="$LOCAL_RUN_DIR/controller-exists.json"
  if "$LOCAL_PYTHON" "$POD_HELPER" exists --pod-id "$POD_ID" > "$snapshot" 2>/dev/null; then
    read -r pod_exists live_name desired_status < <("$LOCAL_PYTHON" - "$snapshot" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
pod = value.get("pod") or {}
print(str(bool(value.get("exists"))).lower(), pod.get("name") or "", pod.get("desiredStatus") or "")
PY
)
    if [ "$pod_exists" = "false" ]; then
      FINAL_REASON="provider_reports_absent"
      break
    fi
    if [ "$live_name" != "$POD_NAME" ]; then
      echo "pod identity changed; refusing further remote mutation" >&2
      FINAL_REASON="pod_identity_changed"
      run_result=1
      break
    fi
    if [ "$desired_status" = "EXITED" ]; then
      FINAL_REASON="provider_stopped"
      break
    fi
  else
    echo "warning: RunPod status transport failed; watchdogs remain armed" >&2
  fi

  if read -r ui_status hands_completed < <(fetch_ui_state 2>/dev/null); then
    case "$ui_status" in
      error)
        echo "browser bridge reported an error; terminating early" >&2
        FINAL_REASON="ui_error"
        run_result=1
        break
        ;;
      match_complete)
        [ "$completion_detected_epoch" -ne 0 ] || completion_detected_epoch="$(date +%s)"
        ;;
    esac
  fi

  if process_state="$(remote_process_state 2>/dev/null)"; then
    case "$process_state" in
      *"ui=dead"*)
        echo "web UI process exited; terminating early" >&2
        FINAL_REASON="ui_process_exited"
        run_result=1
        break
        ;;
      *"dealer=running"*"bot=dead"*)
        echo "bot process exited while dealer remained active; terminating early" >&2
        FINAL_REASON="bot_process_exited"
        run_result=1
        break
        ;;
      *"dealer=dead"*)
        [ "$completion_detected_epoch" -ne 0 ] || completion_detected_epoch="$(date +%s)"
        ;;
    esac
  fi

  if [ "$completion_detected_epoch" -ne 0 ] && \
      [ $(( $(date +%s) - completion_detected_epoch )) -ge "$MATCH_COMPLETE_GRACE_SECONDS" ]; then
    FINAL_REASON="match_complete"
    break
  fi
  sleep 12
done

[ -n "$FINAL_REASON" ] && [ "$FINAL_REASON" != "running" ] || FINAL_REASON="guard_finalize_margin"
trap - EXIT INT TERM
release_launch_lock
if ! finalize_session "$FINAL_REASON"; then
  run_result=1
fi
exit "$run_result"
