#!/usr/bin/env bash
# Rent one throwaway GPU, benchmark DyypHoldem's river resolver, copy the
# result home, and terminate the pod on every exit path.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_HELPER="${DYYPHOLDEM_POD_HELPER:-/Users/palm/Documents/poker-supremus/scripts/runpod_cfv_pod.py}"
ENV_FILE="${DYYPHOLDEM_ENV_FILE:-/Users/palm/Documents/poker-supremus/.env.local}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"
GUARD_SECONDS="${DYYPHOLDEM_GPU_GUARD_SECONDS:-3600}"
MAX_CREATE_ATTEMPTS="${DYYPHOLDEM_GPU_CREATE_ATTEMPTS:-10}"
GPU_TYPE="${DYYPHOLDEM_GPU_TYPE:-NVIDIA GeForce RTX 3090}"
GPU_CLOUD_TYPE="${DYYPHOLDEM_GPU_CLOUD_TYPE:-COMMUNITY}"
RUN_NAME="dyypholdem-river-$(date -u +%Y%m%dT%H%M%SZ)"
POD_NAME="$RUN_NAME-$PPID-$$"
SOURCE_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
SOURCE_DIFF_SHA256="$(git -C "$PROJECT_DIR" diff -- src/nn/bucketer.py src/settings/arguments.py | shasum -a 256 | awk '{print $1}')"
LOCAL_RUN_DIR="$PROJECT_DIR/runs/gpu-baseline/$RUN_NAME"
TASK_TMP="$(mktemp -d)"
SSH_CONFIG="$TASK_TMP/ssh-config"
CREATE_JSON="$TASK_TMP/create.json"
POD_LIST_JSON="$TASK_TMP/pods.json"
POD_ID=""
REMOTE_READY=0

if [ "${1:-}" = "--dry-run" ]; then
  printf '%s\n' \
    "DyypHoldem GPU baseline dry run" \
    "  source commit: $SOURCE_COMMIT" \
    "  GPU: one $GPU_CLOUD_TYPE $GPU_TYPE" \
    "  workload: river, 1,000 CFR iterations, two deterministic repeats" \
    "  assets: three hash-verified files (261.28 MiB total)" \
    "  hard guard: $GUARD_SECONDS seconds" \
    "  cleanup: stop, terminate, and verify exact-name pod absence"
  rm -rf "$TASK_TMP"
  exit 0
fi

copy_back() {
  if [ "$REMOTE_READY" = 1 ] && [ -s "$SSH_CONFIG" ]; then
    mkdir -p "$LOCAL_RUN_DIR"
    rsync -az -e "ssh -F $SSH_CONFIG -o BatchMode=yes -o ConnectTimeout=10" \
      "cfvpod:/root/dyypholdem/runs/gpu-baseline/$RUN_NAME/" \
      "$LOCAL_RUN_DIR/" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  set +e
  copy_back
  matching_ids=""
  pod_absent=0
  for _ in $(seq 1 12); do
    matching_ids=""
    if "$LOCAL_PYTHON" "$POD_HELPER" list >"$POD_LIST_JSON" 2>/dev/null; then
      matching_ids="$($LOCAL_PYTHON - "$POD_NAME" "$POD_LIST_JSON" <<'PY'
import json
import sys

pod_name, path = sys.argv[1:]
print("\n".join(str(pod["id"]) for pod in json.load(open(path)) if pod.get("name") == pod_name))
PY
      )"
      if [ -z "$matching_ids" ]; then
        pod_absent=1
        break
      fi
    fi
    while IFS= read -r matching_id; do
      [ -n "$matching_id" ] || continue
      echo "terminating throwaway DyypHoldem GPU pod $matching_id"
      "$LOCAL_PYTHON" "$POD_HELPER" stop --pod-id "$matching_id" >/dev/null 2>&1
      "$LOCAL_PYTHON" "$POD_HELPER" terminate --pod-id "$matching_id" >/dev/null 2>&1
    done <<EOF
$matching_ids
EOF
    sleep 5
  done
  if [ "$pod_absent" = 1 ]; then
    echo "verified no matching DyypHoldem pod remains"
  else
    echo "ERROR: could not verify DyypHoldem pod termination" >&2
    exit_code=1
  fi
  rm -rf "$TASK_TMP"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

[ -f "$POD_HELPER" ] || { echo "missing safe RunPod helper: $POD_HELPER" >&2; exit 1; }
[ -f "$ENV_FILE" ] || { echo "missing ignored credential file: $ENV_FILE" >&2; exit 1; }
set -a
source "$ENV_FILE"
set +a

"$LOCAL_PYTHON" "$POD_HELPER" list >"$POD_LIST_JSON"
echo "RunPod preflight complete"

recover_pods_by_name() {
  "$LOCAL_PYTHON" "$POD_HELPER" list >"$POD_LIST_JSON" 2>/dev/null || return 1
  "$LOCAL_PYTHON" - "$POD_NAME" "$POD_LIST_JSON" <<'PY'
import json
import sys

pod_name, path = sys.argv[1:]
for pod in json.load(open(path)):
    if pod.get("name") == pod_name:
        print(pod.get("id") or "")
PY
}

reconcile_created_pods() {
  recovered_ids="$(recover_pods_by_name || true)"
  if [ -n "$recovered_ids" ] && [ -z "$POD_ID" ]; then
    POD_ID="$(printf '%s\n' "$recovered_ids" | head -1)"
  fi
  while IFS= read -r recovered_id; do
    [ -n "$recovered_id" ] || continue
    [ "$recovered_id" = "$POD_ID" ] && continue
    "$LOCAL_PYTHON" "$POD_HELPER" stop --pod-id "$recovered_id" >/dev/null 2>&1
    "$LOCAL_PYTHON" "$POD_HELPER" terminate --pod-id "$recovered_id" >/dev/null 2>&1
  done <<EOF
$recovered_ids
EOF
}

for attempt in $(seq 1 "$MAX_CREATE_ATTEMPTS"); do
  reconcile_created_pods
  [ -z "$POD_ID" ] || break
  if "$LOCAL_PYTHON" "$POD_HELPER" create-gpu \
      --name "$POD_NAME" \
      --gpu-types "$GPU_TYPE" \
      --cloud-type "$GPU_CLOUD_TYPE" >"$CREATE_JSON" 2>/dev/null; then
    POD_ID="$($LOCAL_PYTHON - "$CREATE_JSON" <<'PY'
import json
import sys

try:
    print(json.load(open(sys.argv[1])).get("id") or "")
except (OSError, ValueError, AttributeError):
    print("")
PY
)"
  fi
  if [ -z "$POD_ID" ]; then
    for _ in $(seq 1 3); do
      reconcile_created_pods
      [ -z "$POD_ID" ] || break
      sleep 2
    done
  fi
  reconcile_created_pods
  [ -z "$POD_ID" ] || break
  echo "no $GPU_CLOUD_TYPE $GPU_TYPE capacity (attempt $attempt/$MAX_CREATE_ATTEMPTS)"
  sleep 20
done
[ -n "$POD_ID" ] || { echo "unable to acquire a $GPU_CLOUD_TYPE $GPU_TYPE" >&2; exit 1; }

cost_per_hour="$($LOCAL_PYTHON - "$CREATE_JSON" <<'PY'
import json
import sys

try:
    print(json.load(open(sys.argv[1])).get("costPerHr") or "unknown")
except (OSError, ValueError, AttributeError):
    print("unknown")
PY
)"
echo "acquired isolated GPU pod at \$$cost_per_hour/hour"

ssh_ready=0
for _ in $(seq 1 36); do
  if "$LOCAL_PYTHON" "$POD_HELPER" ssh-config \
      --pod-id "$POD_ID" --out "$SSH_CONFIG" >/dev/null 2>&1; then
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
  if "${SSH[@]}" cfvpod true 2>/dev/null; then
    connected=1
    break
  fi
  sleep 5
done
[ "$connected" = 1 ] || { echo "pod SSH never became ready" >&2; exit 1; }

printf '#!/usr/bin/env bash\nsleep %s\nrunpodctl stop pod %s\n' \
  "$GUARD_SECONDS" "$POD_ID" | "${SSH_STDIN[@]}" cfvpod \
  'command -v runpodctl >/dev/null && cat > /root/dyypholdem_self_stop.sh && chmod +x /root/dyypholdem_self_stop.sh && (setsid nohup /root/dyypholdem_self_stop.sh >/root/dyypholdem_self_stop.log 2>&1 < /dev/null &)'
echo "remote self-stop guard armed"

"${SSH[@]}" cfvpod \
  'command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync) >/dev/null 2>&1; mkdir -p /root/dyypholdem /root/logs'

rsync -az -e "ssh -F $SSH_CONFIG -o BatchMode=yes" \
  --exclude .git \
  --exclude .DS_Store \
  --exclude __pycache__ \
  --exclude runs \
  "$PROJECT_DIR/src" "$PROJECT_DIR/scripts" \
  cfvpod:/root/dyypholdem/

"${SSH[@]}" cfvpod 'python3 -m pip install --quiet gdown loguru'

gpu_healthy=0
for _ in $(seq 1 18); do
  if "${SSH[@]}" cfvpod \
      'timeout 20s nvidia-smi -L >/dev/null 2>&1 && timeout 20s python3 -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1'; then
    gpu_healthy=1
    break
  fi
  sleep 10
done
[ "$gpu_healthy" = 1 ] || { echo "unhealthy GPU host" >&2; exit 1; }
"${SSH[@]}" cfvpod \
  'python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"'

echo "materializing and verifying minimal river assets"
"${SSH[@]}" cfvpod \
  'cd /root/dyypholdem && python3 scripts/materialize_assets.py --profile river'

REMOTE_READY=1
echo "running DyypHoldem river CUDA baseline"
"${SSH[@]}" cfvpod \
  "set -o pipefail; cd /root/dyypholdem && mkdir -p runs/gpu-baseline/$RUN_NAME && timeout 1800s python3 scripts/gpu_baseline.py --street river --iterations 1000 --repeats 2 --seed 0 --source-commit $SOURCE_COMMIT --source-diff-sha256 $SOURCE_DIFF_SHA256 --expected-root-sha256 820d911cad2f15416b19b02e0e61de4b5740d8d6eeffaf15c1746d58361c5ca6 --expected-strategy-sha256 16529973d061d3c594a067fa251af7a44569377958b5f8255c748ab889610346 --output runs/gpu-baseline/$RUN_NAME/summary.json 2>&1 | tee runs/gpu-baseline/$RUN_NAME/run.log"

copy_back
[ -s "$LOCAL_RUN_DIR/summary.json" ] || { echo "benchmark summary was not copied back" >&2; exit 1; }
echo "DyypHoldem CUDA baseline copied to $LOCAL_RUN_DIR"
