import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "run_play_ui.sh"


class RunPlayUiLauncherTests(unittest.TestCase):
    def run_spend_gate(self, rate, guard_seconds="1800", cap="0.50"):
        source = LAUNCHER.read_text()
        marker = "validate_config() {\n"
        prefix, found, _ = source.partition(marker)
        self.assertEqual(found, marker)
        script = (
            prefix
            + f"COST_PER_HOUR={rate!r}\n"
            + f"GUARD_SECONDS={guard_seconds!r}\n"
            + f"MAX_TOTAL_COST_USD={cap!r}\n"
            + "enforce_projected_spend_cap\n"
        )
        return subprocess.run(
            ["bash"],
            input=script,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_dry_run_reports_detached_controller(self):
        result = subprocess.run(
            [str(LAUNCHER), "dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("controller: detached locally", result.stdout)
        self.assertIn("hard guard: 3600 seconds", result.stdout)
        self.assertIn("GPU regression: 1", result.stdout)

    def test_dry_run_can_disable_gpu_regression_explicitly(self):
        env = os.environ.copy()
        env["DYYPHOLDEM_UI_GPU_REGRESSION"] = "0"
        result = subprocess.run(
            [str(LAUNCHER), "dry-run"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertIn("GPU regression: 0", result.stdout)

    def test_dry_run_reports_projected_spend_cap(self):
        env = os.environ.copy()
        env["DYYPHOLDEM_UI_MAX_TOTAL_COST_USD"] = "0.50"
        result = subprocess.run(
            [str(LAUNCHER), "dry-run"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertIn(
            "spend cap: 0.50 USD projected maximum compute cost",
            result.stdout,
        )

    def test_invalid_spend_cap_fails_before_launch(self):
        env = os.environ.copy()
        env["DYYPHOLDEM_UI_MAX_TOTAL_COST_USD"] = "not-a-price"
        result = subprocess.run(
            [str(LAUNCHER), "dry-run"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be a positive decimal", result.stderr)

    def test_spend_gate_accepts_exact_authorized_boundary(self):
        result = self.run_spend_gate("1.00")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("$0.5000 is within authorized $0.5000", result.stdout)

    def test_spend_gate_rejects_overpriced_or_invalid_quotes(self):
        for rate in ("1.0001", "0", "unknown", "NaN", "Infinity"):
            with self.subTest(rate=rate):
                result = self.run_spend_gate(rate)
                self.assertNotEqual(result.returncode, 0)

    def test_start_detaches_controller_and_waits_for_its_manifest(self):
        source = LAUNCHER.read_text()
        marker = '[ "$COMMAND" = "controller" ] || { usage >&2; exit 2; }\n'
        prefix, found, _ = source.partition(marker)
        self.assertEqual(found, marker)

        fake_controller = prefix + marker + r'''
mkdir -p "$SESSION_ROOT"
"$LOCAL_PYTHON" - "$CURRENT_MANIFEST" "$$" <<'PY'
import json
from pathlib import Path
import sys

path, launcher_pid = sys.argv[1:]
Path(path).write_text(json.dumps({
    "status": "running",
    "launcher_pid": int(launcher_pid),
    "authenticated_url": "https://example.invalid/?token=test",
    "local_run_dir": "/tmp/fake-play-ui-run",
    "absolute_stop_epoch": 4102444800,
}) + "\n")
PY
sleep 30
'''

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script_dir = root / "scripts"
            script_dir.mkdir()
            script = script_dir / "run_play_ui.sh"
            script.write_text(fake_controller)
            script.chmod(0o755)
            env = os.environ.copy()
            env["DYYPHOLDEM_UI_CONTROLLER_READY_WAIT_SECONDS"] = "10"
            result = subprocess.run(
                [str(script), "start"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )

            session_root = root / "runs" / "play-ui"
            pid_file = session_root / "controller.pid"
            log_file = session_root / "controller.log"
            controller_pid = int(pid_file.read_text().strip())
            try:
                os.kill(controller_pid, 0)
                manifest = json.loads((session_root / "current.json").read_text())
                self.assertEqual(controller_pid, manifest["launcher_pid"])
                self.assertEqual(controller_pid, os.getsid(controller_pid))
                self.assertIn(f"PLAY_UI_CONTROLLER_PID={controller_pid}", result.stdout)
                self.assertIn("PLAY_UI_READY", result.stdout)
                self.assertEqual(stat.S_IMODE(pid_file.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(log_file.stat().st_mode), 0o600)
            finally:
                try:
                    os.kill(controller_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                for _ in range(50):
                    try:
                        os.kill(controller_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    try:
                        os.kill(controller_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_status_redacts_authenticated_url_and_credentials_under_xtrace(self):
        secret_token = "status-secret-token-must-not-appear"
        secret_api_key = "runpod-secret-key-must-not-appear"
        authenticated_url = f"https://example.invalid/?token={secret_token}"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script_dir = root / "scripts"
            session_root = root / "runs" / "play-ui"
            local_run_dir = session_root / "status-test-run"
            script_dir.mkdir()
            local_run_dir.mkdir(parents=True)

            launcher = script_dir / "run_play_ui.sh"
            launcher.write_text(LAUNCHER.read_text())
            launcher.chmod(0o755)

            pod_helper = script_dir / "runpod_ui_pod.py"
            pod_helper.write_text(
                "import json\n"
                "print(json.dumps({'exists': False, 'id': 'statuspod123'}))\n"
            )

            credential_file = root / ".env.local"
            credential_file.write_text(f"RUNPOD_API_KEY={secret_api_key}\n")

            python_wrapper = root / "python-wrapper"
            python_wrapper.write_text(
                f"#!{sys.executable}\n"
                "import os\n"
                "import sys\n"
                "if sys.argv[-1:] == ['authenticated_url']:\n"
                "    raise SystemExit('status attempted to read authenticated_url')\n"
                f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
            )
            python_wrapper.chmod(0o755)

            manifest = {
                "status": "terminated",
                "pod_id": "statuspod123",
                "pod_name": "dyypholdem-ui-status-test",
                "run_name": "status-test-run",
                "public_url": "https://example.invalid",
                "authenticated_url": authenticated_url,
                "cost_per_hour": "0.74",
                "local_run_dir": str(local_run_dir),
                "ssh_config": str(local_run_dir / "missing-ssh-config"),
                "absolute_stop_epoch": 4102444800,
                "session_deadline_epoch": 4102444700,
                "watchdog_pid": None,
                "copy_status": "succeeded",
                "last_successful_copy_at": "2026-08-23T21:22:15+00:00",
            }
            (session_root / "current.json").write_text(json.dumps(manifest) + "\n")

            env = os.environ.copy()
            env["DYYPHOLDEM_ENV_FILE"] = str(credential_file)
            env["LOCAL_PYTHON"] = str(python_wrapper)
            result = subprocess.run(
                ["bash", "-x", str(launcher), "status"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            combined_output = result.stdout + result.stderr
            self.assertIn("browserAccess=redacted", result.stdout)
            self.assertIn(f"localRunDir={local_run_dir}", result.stdout)
            self.assertNotIn("authenticatedUrl=", combined_output)
            self.assertNotIn(authenticated_url, combined_output)
            self.assertNotIn(secret_token, combined_output)
            self.assertNotIn(secret_api_key, combined_output)


if __name__ == "__main__":
    unittest.main()
