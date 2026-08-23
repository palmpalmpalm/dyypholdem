#!/usr/bin/env bash
# Independent local hard-deadline guard for one exact-name DyypHoldem pod.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 WATCHDOG_CONFIG ENV_FILE" >&2
  exit 2
fi

WATCHDOG_CONFIG="$1"
ENV_FILE="$2"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"

[ -s "$WATCHDOG_CONFIG" ] || { echo "missing watchdog config" >&2; exit 2; }
[ -s "$ENV_FILE" ] || { echo "missing RunPod credential file" >&2; exit 2; }

config_field() {
  "$LOCAL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$WATCHDOG_CONFIG" "$1"
}

POD_ID="$(config_field pod_id)"
POD_NAME="$(config_field pod_name)"
DEADLINE_EPOCH="$(config_field deadline_epoch)"
POD_HELPER="$(config_field pod_helper)"
CANCEL_FILE="$(config_field cancel_file)"
STATUS_FILE="$(config_field status_file)"
WORK_DIR="$(dirname "$WATCHDOG_CONFIG")"

case "$POD_ID" in *[!A-Za-z0-9]*|'') echo "invalid watchdog pod id" >&2; exit 2 ;; esac
case "$POD_NAME" in *[!A-Za-z0-9._-]*|'') echo "invalid watchdog pod name" >&2; exit 2 ;; esac
case "$DEADLINE_EPOCH" in *[!0-9]*|'') echo "invalid watchdog deadline" >&2; exit 2 ;; esac
[ -f "$POD_HELPER" ] || { echo "missing RunPod helper" >&2; exit 2; }

write_status() {
  status="$1"
  detail="${2:-}"
  "$LOCAL_PYTHON" - "$STATUS_FILE" "$status" "$detail" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

path, status, detail = sys.argv[1:]
payload = {
    "status": status,
    "detail": detail,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
target = Path(path)
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.chmod(0o600)
temporary.replace(target)
PY
}

while [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ]; do
  if [ -e "$CANCEL_FILE" ]; then
    write_status cancelled "controller verified exact-name absence"
    exit 0
  fi
  sleep 5
done

if [ -e "$CANCEL_FILE" ]; then
  write_status cancelled "controller verified exact-name absence"
  exit 0
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a
[ -n "${RUNPOD_API_KEY:-}" ] || { write_status failed "RUNPOD_API_KEY is missing"; exit 1; }

snapshot="$WORK_DIR/watchdog-exists.json"
owned=0
for _ in $(seq 1 6); do
  if "$LOCAL_PYTHON" "$POD_HELPER" exists --pod-id "$POD_ID" > "$snapshot" 2>/dev/null; then
    read -r exists name < <("$LOCAL_PYTHON" - "$snapshot" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
pod = value.get("pod") or {}
print(str(bool(value.get("exists"))).lower(), pod.get("name") or "")
PY
)
    if [ "$exists" = "false" ]; then
      owned=0
      break
    fi
    if [ "$name" != "$POD_NAME" ]; then
      write_status refused "pod id no longer has the expected exact name"
      exit 1
    fi
    owned=1
    break
  fi
  sleep 5
done

if [ "$owned" = 1 ]; then
  "$LOCAL_PYTHON" "$POD_HELPER" stop --pod-id "$POD_ID" >/dev/null 2>&1 || true
  for _ in $(seq 1 18); do
    if "$LOCAL_PYTHON" "$POD_HELPER" exists --pod-id "$POD_ID" > "$snapshot" 2>/dev/null; then
      read -r exists desired < <("$LOCAL_PYTHON" - "$snapshot" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
pod = value.get("pod") or {}
print(str(bool(value.get("exists"))).lower(), pod.get("desiredStatus") or "")
PY
)
      [ "$exists" = "false" ] && break
      [ "$desired" = "EXITED" ] && break
    fi
    sleep 2
  done
  for _ in $(seq 1 3); do
    "$LOCAL_PYTHON" "$POD_HELPER" terminate --pod-id "$POD_ID" >/dev/null 2>&1 && break
    sleep 3
  done
fi

successful_empty=0
for _ in $(seq 1 30); do
  list_file="$WORK_DIR/watchdog-pods.json"
  if "$LOCAL_PYTHON" "$POD_HELPER" list > "$list_file" 2>/dev/null; then
    exact_count="$("$LOCAL_PYTHON" - "$list_file" "$POD_NAME" <<'PY'
import json, sys
pods = json.load(open(sys.argv[1]))
print(sum(1 for pod in pods if pod.get("name") == sys.argv[2]))
PY
)"
    if [ "$exact_count" = 0 ]; then
      successful_empty=$((successful_empty + 1))
      if [ "$successful_empty" -ge 6 ]; then
        write_status terminated "hard deadline reached; exact-name absence verified"
        exit 0
      fi
    else
      successful_empty=0
      for exact_id in $("$LOCAL_PYTHON" - "$list_file" "$POD_NAME" <<'PY'
import json, sys
for pod in json.load(open(sys.argv[1])):
    if pod.get("name") == sys.argv[2] and pod.get("id"):
        print(pod["id"])
PY
); do
        "$LOCAL_PYTHON" "$POD_HELPER" stop --pod-id "$exact_id" >/dev/null 2>&1 || true
        "$LOCAL_PYTHON" "$POD_HELPER" terminate --pod-id "$exact_id" >/dev/null 2>&1 || true
      done
    fi
  fi
  sleep 3
done

write_status failed "could not verify exact-name pod absence after hard deadline"
exit 1
