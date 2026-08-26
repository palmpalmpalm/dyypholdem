#!/usr/bin/env bash
# Guarded live RTX 4090 session with periodic telemetry copyback.
# Never allow inherited or command-line xtrace to echo credentials or session URLs.
set +x
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
CONTROLLER_LOG="$SESSION_ROOT/controller.log"
CONTROLLER_PID_FILE="$SESSION_ROOT/controller.pid"
CONTROLLER_READY_WAIT_SECONDS="${DYYPHOLDEM_UI_CONTROLLER_READY_WAIT_SECONDS:-420}"
GUARD_SECONDS="${DYYPHOLDEM_UI_GUARD_SECONDS:-3600}"
GPU_TYPE="${DYYPHOLDEM_GPU_TYPE:-NVIDIA GeForce RTX 4090}"
CLOUD_TYPE="${DYYPHOLDEM_GPU_CLOUD_TYPE:-SECURE}"
HANDS="${DYYPHOLDEM_UI_HANDS:-100}"
SEED="${DYYPHOLDEM_UI_SEED:-20260823}"
OPPONENT="${DYYPHOLDEM_UI_OPPONENT:-human}"
OPPONENT_SEED="${DYYPHOLDEM_UI_OPPONENT_SEED:-20260824}"
GPU_REGRESSION="${DYYPHOLDEM_UI_GPU_REGRESSION:-1}"
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
  validate_uint DYYPHOLDEM_UI_OPPONENT_SEED "$OPPONENT_SEED" 0 2147483647
  [ "$OPPONENT" = "human" ] || [ "$OPPONENT" = "random" ] || {
    echo "DYYPHOLDEM_UI_OPPONENT must be human or random" >&2
    return 1
  }
  [ "$GPU_REGRESSION" = 0 ] || [ "$GPU_REGRESSION" = 1 ] || {
    echo "DYYPHOLDEM_UI_GPU_REGRESSION must be 0 or 1" >&2
    return 1
  }
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
  set +x 2>/dev/null || true
  [ -f "$ENV_FILE" ] || { echo "missing ignored RunPod credential file: $ENV_FILE" >&2; return 1; }
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +x 2>/dev/null || true
  set +a
  [ -n "${RUNPOD_API_KEY:-}" ] || { echo "RUNPOD_API_KEY is missing from $ENV_FILE" >&2; return 1; }
}

verify_models() {
  for relative in preflop-aux/final_compact.pt flop/final_compact.pt turn/final_compact.pt river/final_compact.pt; do
    [ -s "$MODEL_ROOT/$relative" ] || { echo "missing compact model: $MODEL_ROOT/$relative" >&2; return 1; }
  done
}

verify_play_ui_bundle() {
  [ -s "$PROJECT_DIR/requirements-play-ui.txt" ] || {
    echo "missing play UI Python requirements" >&2
    return 1
  }
  [ -s "$PROJECT_DIR/scripts/solver_regression.py" ] || {
    echo "missing strict solver regression harness" >&2
    return 1
  }
  [ -s "$PROJECT_DIR/web/dist/index.html" ] || {
    echo "missing compiled play UI; run 'make web-build' before renting" >&2
    return 1
  }
  find "$PROJECT_DIR/web/dist/assets" -maxdepth 1 -type f -name '*.js' -size +0c \
    -print -quit 2>/dev/null | grep -q . || {
      echo "compiled play UI has no JavaScript bundle" >&2
      return 1
    }
  find "$PROJECT_DIR/web/dist/assets" -maxdepth 1 -type f -name '*.css' -size +0c \
    -print -quit 2>/dev/null | grep -q . || {
      echo "compiled play UI has no stylesheet bundle" >&2
      return 1
    }
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
  set +x 2>/dev/null || true
  local authenticated_url_mode="${1:-load-authenticated-url}"
  [ -s "$CURRENT_MANIFEST" ] || { echo "no DyypHoldem UI session manifest" >&2; return 1; }
  POD_ID="$(json_field "$CURRENT_MANIFEST" pod_id)"
  POD_NAME="$(json_field "$CURRENT_MANIFEST" pod_name)"
  RUN_NAME="$(json_field "$CURRENT_MANIFEST" run_name)"
  PUBLIC_URL="$(json_field "$CURRENT_MANIFEST" public_url)"
  if [ "$authenticated_url_mode" = "load-authenticated-url" ]; then
    AUTHENTICATED_URL="$(json_field "$CURRENT_MANIFEST" authenticated_url)"
  else
    AUTHENTICATED_URL=""
  fi
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
    "for role in dealer ui bot autoplay; do file=/root/dyypholdem/runs/play-ui/$RUN_NAME/\$role.pid; pid=\$([ -s \"\$file\" ] && tr -cd '0-9' < \"\$file\"); if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then printf '%s=running ' \"\$role\"; else printf '%s=dead ' \"\$role\"; fi; done; echo"
}

validate_random_completion() {
  [ "$OPPONENT" = "random" ] || return 0
  [ "$REMOTE_READY" = 1 ] && [ -s "$SSH_CONFIG" ] || return 1
  for _ in $(seq 1 3); do
    if ssh -n -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=10 dyyui \
        "python3 /root/dyypholdem/scripts/validate_random_benchmark.py --run-dir /root/dyypholdem/runs/play-ui/$RUN_NAME --hands '$HANDS'"
    then
      return 0
    fi
    sleep 2
  done
  return 1
}

controller_state() {
  controller_pid=""
  [ -s "$CONTROLLER_PID_FILE" ] && controller_pid="$(tr -cd '0-9' < "$CONTROLLER_PID_FILE")"
  if [ -n "$controller_pid" ] && kill -0 "$controller_pid" 2>/dev/null; then
    printf '%s %s\n' "$controller_pid" running
  elif [ -n "$controller_pid" ]; then
    printf '%s %s\n' "$controller_pid" exited
  else
    printf '%s %s\n' "-" absent
  fi
}

start_detached_controller() {
  validate_uint DYYPHOLDEM_UI_CONTROLLER_READY_WAIT_SECONDS \
    "$CONTROLLER_READY_WAIT_SECONDS" 1 900
  mkdir -p "$SESSION_ROOT"

  read -r previous_pid previous_state < <(controller_state)
  if [ "$previous_state" = running ]; then
    echo "a DyypHoldem UI controller is already running (pid $previous_pid)" >&2
    return 1
  fi

  umask 077
  : > "$CONTROLLER_LOG"
  controller_pid="$("$LOCAL_PYTHON" - "$PROJECT_DIR/scripts/run_play_ui.sh" \
    "$CONTROLLER_LOG" <<'PY'
import os
import subprocess
import sys

launcher, log_path = sys.argv[1:]
with open(log_path, "ab", buffering=0) as log:
    process = subprocess.Popen(
        [launcher, "controller"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
print(process.pid)
PY
)"
  printf '%s\n' "$controller_pid" > "$CONTROLLER_PID_FILE"
  chmod 600 "$CONTROLLER_LOG" "$CONTROLLER_PID_FILE"
  printf 'PLAY_UI_CONTROLLER_PID=%s\nPLAY_UI_CONTROLLER_LOG=%s\n' \
    "$controller_pid" "$CONTROLLER_LOG"

  deadline=$(( $(date +%s) + CONTROLLER_READY_WAIT_SECONDS ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ -s "$CURRENT_MANIFEST" ]; then
      manifest_pid="$(json_field "$CURRENT_MANIFEST" launcher_pid)"
      manifest_status="$(json_field "$CURRENT_MANIFEST" status)"
      if [ "$manifest_pid" = "$controller_pid" ]; then
        case "$manifest_status" in
          running)
            printf 'PLAY_UI_READY\nPLAY_UI_URL=%s\nLOCAL_RUN_DIR=%s\nHARD_STOP_EPOCH=%s\n' \
              "$(json_field "$CURRENT_MANIFEST" authenticated_url)" \
              "$(json_field "$CURRENT_MANIFEST" local_run_dir)" \
              "$(json_field "$CURRENT_MANIFEST" absolute_stop_epoch)"
            return 0
            ;;
          terminated|terminated_*)
            echo "detached DyypHoldem UI controller terminated during setup" >&2
            tail -80 "$CONTROLLER_LOG" >&2 || true
            return 1
            ;;
        esac
      fi
    fi
    if ! kill -0 "$controller_pid" 2>/dev/null; then
      wait "$controller_pid" 2>/dev/null || controller_rc=$?
      controller_rc="${controller_rc:-1}"
      echo "detached DyypHoldem UI controller exited during setup (status $controller_rc)" >&2
      tail -80 "$CONTROLLER_LOG" >&2 || true
      return "$controller_rc"
    fi
    sleep 2
  done

  echo "controller is still setting up; it remains detached" >&2
  echo "follow progress with: tail -f $CONTROLLER_LOG" >&2
}

if [ "$COMMAND" = "dry-run" ]; then
  validate_config
  printf '%s\n' \
    "DyypHoldem live UI dry run" \
    "  GPU: one $CLOUD_TYPE $GPU_TYPE" \
    "  public service: authenticated HTTPS proxy on port $HTTP_PORT" \
    "  solver: real ACPC dealer + ContinualResolving, 1,000 CFR iterations" \
    "  opponent: $OPPONENT${OPPONENT_SEED:+ (seed $OPPONENT_SEED)}" \
    "  models: four checksum-verified compact recovered networks" \
    "  telemetry: private JSONL plus safe live/final per-street reports" \
    "  GPU regression: $GPU_REGRESSION (strict preflop root/chance tensors before UI start)" \
    "  controller: detached locally; start waits up to $CONTROLLER_READY_WAIT_SECONDS seconds for PLAY_UI_READY" \
    "  hard guard: $GUARD_SECONDS seconds; authenticated remote retry-delete plus independent local stop/delete watchdog" \
    "  shutdown: quiesce, retry final copyback, exact-name stop/delete, six successful absence checks"
  exit 0
fi

if [ "$COMMAND" = "status" ] || [ "$COMMAND" = "logs" ] || [ "$COMMAND" = "stop" ]; then
  if [ "$COMMAND" = "stop" ]; then
    load_manifest_session load-authenticated-url
  else
    # Status and logs need only operational metadata; do not read the secret URL.
    load_manifest_session omit-authenticated-url
  fi
  read -r saved_controller_pid saved_controller_state < <(controller_state)
  printf 'controllerPid=%s\ncontrollerState=%s\ncontrollerLog=%s\n' \
    "$saved_controller_pid" "$saved_controller_state" "$CONTROLLER_LOG"
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
    printf 'browserAccess=redacted\nlocalRunDir=%s\ncopyStatus=%s\nlastSuccessfulCopyAt=%s\n' \
      "$LOCAL_RUN_DIR" "$COPY_STATUS" "$LAST_COPY_AT"
    exit 0
  fi
  if finalize_session user_stop; then
    echo "DyypHoldem UI session stopped; exact-name absence verified"
    exit 0
  fi
  exit 1
fi

if [ "$COMMAND" = "start" ]; then
  start_detached_controller
  exit $?
fi

[ "$COMMAND" = "controller" ] || { usage >&2; exit 2; }
validate_config
verify_models
verify_play_ui_bundle
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
printf '#!/usr/bin/env bash\nset -u\nsleep %s\n# Keep requesting permanent deletion until this container is terminated.\nwhile true; do\n  runpodctl pod delete %s >/dev/null 2>&1 || runpodctl remove pod %s >/dev/null 2>&1 || true\n  sleep 15\ndone\n' "$remote_guard_seconds" "$POD_ID" "$POD_ID" | \
  "${SSH_STDIN[@]}" dyyui 'command -v runpodctl >/dev/null && umask 077 && cat > /root/dyypholdem_self_stop.sh && chmod 700 /root/dyypholdem_self_stop.sh'
remote_guard_command="IFS= read -r RUNPOD_API_KEY && test -n \"\$RUNPOD_API_KEY\" && export RUNPOD_API_KEY && (runpodctl pod list >/dev/null 2>&1 || runpodctl get pod >/dev/null 2>&1) && (setsid nohup /root/dyypholdem_self_stop.sh >/root/dyypholdem_self_stop.log 2>&1 < /dev/null & guard=\$!; sleep 1; kill -0 \"\$guard\")"
printf '%s\n' "$RUNPOD_API_KEY" | "${SSH_STDIN[@]}" dyyui "$remote_guard_command"
echo "authenticated remote hard-deadline delete guard armed"

"${SSH[@]}" dyyui 'if command -v rsync >/dev/null && command -v curl >/dev/null; then :; else (apt-get update -qq && apt-get install -y -qq rsync curl) >/dev/null 2>&1 || exit 1; fi; mkdir -p /root/dyypholdem /root/logs'
rsync -az -e "ssh -F $SSH_CONFIG -o BatchMode=yes" \
  --exclude .git --exclude .DS_Store --exclude __pycache__ --exclude runs \
  --exclude node_modules --exclude coverage --exclude .vite \
  "$PROJECT_DIR/src" "$PROJECT_DIR/scripts" "$PROJECT_DIR/acpc_server" \
  "$PROJECT_DIR/web" "$PROJECT_DIR/requirements-play-ui.txt" \
  dyyui:/root/dyypholdem/
"${SSH[@]}" dyyui "python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version' && python3 -m pip install --quiet --break-system-packages -r /root/dyypholdem/requirements-play-ui.txt && python3 -c 'import gdown, loguru, pokerkit; from importlib.metadata import version; assert version(\"gdown\") == \"5.2.0\"; assert version(\"loguru\") == \"0.7.3\"; assert version(\"PokerKit\") == \"0.7.5\"' && mkdir -p /root/dyypholdem/runs/model-recovery/compact /root/dyypholdem/runs/play-ui/$RUN_NAME"
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

if [ "$GPU_REGRESSION" = 1 ]; then
  echo "running strict CUDA preflop/chance regression"
  "${SSH[@]}" dyyui "cd /root/dyypholdem && timeout 900s python3 scripts/solver_regression.py capture --device cuda --spot preflop-root --iterations 1000 --skip-iterations 500 --warmups 1 --repeats 3 --threads 1 --output runs/play-ui/$RUN_NAME/solver-regression-cuda.json > runs/play-ui/$RUN_NAME/solver-regression-cuda.log 2>&1"
  "${SSH[@]}" dyyui "cd /root/dyypholdem && python3 -c 'import json; p=json.load(open(\"runs/play-ui/$RUN_NAME/solver-regression-cuda.json\")); s=p[\"spots\"][0]; rows=[a for b in s[\"chance_action_cfvs\"][\"boards\"] for a in b[\"actions\"]]; assert s[\"timing\"][\"max_repeat_tensor_delta\"] == 0; assert len(rows) == 6; assert all(a[\"timing\"][\"solver\"].get(\"captured_flop\") is True and a[\"timing\"][\"solver\"].get(\"replayed_flop\") is False for a in rows)'"
fi

echo "starting dealer, authenticated UI, and real continual resolver"
"${SSH[@]}" dyyui "export DYYPHOLDEM_COMPACT_MODEL_PATH=/root/dyypholdem/runs/model-recovery/compact DYYPHOLDEM_SOURCE_COMMIT=$(git -C "$PROJECT_DIR" rev-parse HEAD); cd /root/dyypholdem && ./scripts/start_play_ui_remote.sh '$RUN_NAME' '$HANDS' '$SEED' /root/dyypholdem/session-token '$OPPONENT' '$OPPONENT_SEED'"

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

# Prove the bearer-token bootstrap, HttpOnly cookie, React document, and one
# hashed static asset all work through the public HTTPS proxy before sharing it.
COOKIE_JAR="$LOCAL_RUN_DIR/browser-cookie.jar"
FRONTEND_HTML="$LOCAL_RUN_DIR/frontend-index.html"
if ! curl -fsS --max-time 20 -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    "$AUTHENTICATED_URL" > "$FRONTEND_HTML"; then
  echo "authenticated browser session bootstrap failed" >&2
  exit 1
fi
chmod 600 "$COOKIE_JAR" "$FRONTEND_HTML"
grep -q '<div id="root"' "$FRONTEND_HTML" || {
  echo "public proxy did not serve the compiled React table" >&2
  exit 1
}
FRONTEND_ASSET="$($LOCAL_PYTHON - "$FRONTEND_HTML" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r'(?:src|href)="(/assets/[^"]+\.(?:js|css))"', text)
print(match.group(1) if match else "")
PY
)"
[ -n "$FRONTEND_ASSET" ] || { echo "compiled React table references no hashed asset" >&2; exit 1; }
curl -fsS --max-time 20 -b "$COOKIE_JAR" "$PUBLIC_URL$FRONTEND_ASSET" >/dev/null || {
  echo "compiled React asset was not reachable through the proxy" >&2
  exit 1
}

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
        if [ "${ui_status:-unknown}" = "match_complete" ]; then
          [ "$completion_detected_epoch" -ne 0 ] || completion_detected_epoch="$(date +%s)"
        else
          echo "bot process exited while dealer remained active; terminating early" >&2
          FINAL_REASON="bot_process_exited"
          run_result=1
          break
        fi
        ;;
      *"dealer=running"*"autoplay=dead"*)
        if [ "$OPPONENT" = "random" ] && [ "${ui_status:-unknown}" != "match_complete" ]; then
          echo "random opponent process exited before match completion; terminating early" >&2
          FINAL_REASON="random_opponent_exited"
          run_result=1
          break
        fi
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

# A final hand can land inside the 60-second process-drain grace but before the
# 90-second provider-finalization margin. Accept it immediately when both sides'
# exact artifacts already prove clean completion.
if [ "$OPPONENT" = "random" ] && [ "$FINAL_REASON" = "running" ] && \
    [ "$completion_detected_epoch" -ne 0 ] && validate_random_completion; then
  FINAL_REASON="match_complete"
fi
[ -n "$FINAL_REASON" ] && [ "$FINAL_REASON" != "running" ] || FINAL_REASON="guard_finalize_margin"
if [ "$OPPONENT" = "random" ]; then
  if [ "$FINAL_REASON" = "match_complete" ]; then
    if ! validate_random_completion; then
      echo "100-hand random benchmark failed final artifact validation" >&2
      FINAL_REASON="random_completion_validation_failed"
      run_result=1
    fi
  elif [ "$run_result" -eq 0 ]; then
    echo "random benchmark ended before all $HANDS hands completed" >&2
    run_result=1
  fi
fi
trap - EXIT INT TERM
release_launch_lock
if ! finalize_session "$FINAL_REASON"; then
  run_result=1
fi
exit "$run_result"
