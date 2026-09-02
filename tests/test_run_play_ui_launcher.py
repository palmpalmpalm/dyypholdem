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

    def test_dry_run_slumbot_mode_reports_outbound_api_and_long_guard(self):
        env = os.environ.copy()
        env["DYYPHOLDEM_UI_OPPONENT"] = "slumbot"
        env["DYYPHOLDEM_UI_HANDS"] = "1000"
        env["DYYPHOLDEM_UI_SEED"] = "20260902"
        env["DYYPHOLDEM_UI_GUARD_SECONDS"] = "19800"
        result = subprocess.run(
            [str(LAUNCHER), "dry-run"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertIn("opponent: slumbot (1000 hands over 1 concurrent session(s), bot seed 20260902", result.stdout)
        self.assertIn("public service: none; the bot connects out to https://slumbot.com/api", result.stdout)
        self.assertIn("hard guard: 19800 seconds", result.stdout)

    def test_slumbot_sessions_must_divide_hands_and_only_apply_to_slumbot(self):
        cases = (
            ({"DYYPHOLDEM_UI_OPPONENT": "slumbot", "DYYPHOLDEM_UI_HANDS": "5000", "DYYPHOLDEM_UI_SESSIONS": "3"}, "divide evenly"),
            ({"DYYPHOLDEM_UI_OPPONENT": "random", "DYYPHOLDEM_UI_HANDS": "100", "DYYPHOLDEM_UI_SESSIONS": "2"}, "applies only to the slumbot"),
            ({"DYYPHOLDEM_UI_OPPONENT": "slumbot", "DYYPHOLDEM_UI_HANDS": "20016", "DYYPHOLDEM_UI_SESSIONS": "16"}, "1 through 20000"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                env = os.environ.copy()
                env.update(overrides)
                result = subprocess.run([str(LAUNCHER), "dry-run"], check=False, capture_output=True, text=True, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
        env = os.environ.copy()
        env.update({"DYYPHOLDEM_UI_OPPONENT": "slumbot", "DYYPHOLDEM_UI_HANDS": "5000", "DYYPHOLDEM_UI_SESSIONS": "4", "DYYPHOLDEM_UI_GUARD_SECONDS": "21600"})
        result = subprocess.run([str(LAUNCHER), "dry-run"], check=True, capture_output=True, text=True, env=env)
        self.assertIn("5000 hands over 4 concurrent session(s)", result.stdout)

    def test_graph_gate_and_bet_sizing_flags_are_reported_and_validated(self):
        env = os.environ.copy()
        env.update({"DYYPHOLDEM_UI_OPPONENT": "slumbot", "DYYPHOLDEM_UI_GRAPH_GATE": "1", "DYYPHOLDEM_OPPONENT_BET_SIZING": "0.5,1,2"})
        result = subprocess.run([str(LAUNCHER), "dry-run"], check=True, capture_output=True, text=True, env=env)
        self.assertIn("CUDA Graph gate: 1", result.stdout)
        self.assertIn("opponent bet sizing: 0.5,1,2", result.stdout)
        default = subprocess.run([str(LAUNCHER), "dry-run"], check=True, capture_output=True, text=True)
        self.assertIn("CUDA Graph gate: 0", default.stdout)
        self.assertIn("opponent bet sizing: default pot-only tree", default.stdout)
        for overrides, message in (
            ({"DYYPHOLDEM_UI_GRAPH_GATE": "yes"}, "must be 0 or 1"),
            ({"DYYPHOLDEM_OPPONENT_BET_SIZING": "half;pot"}, "comma-separated list of pot fractions"),
        ):
            with self.subTest(overrides=overrides):
                env = os.environ.copy()
                env.update(overrides)
                result = subprocess.run([str(LAUNCHER), "dry-run"], check=False, capture_output=True, text=True, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_guard_above_six_hours_or_unknown_opponent_is_rejected(self):
        for overrides, message in (
            ({"DYYPHOLDEM_UI_GUARD_SECONDS": "21601"}, "900 through 21600"),
            ({"DYYPHOLDEM_UI_OPPONENT": "pluribus"}, "human, random, or slumbot"),
        ):
            with self.subTest(overrides=overrides):
                env = os.environ.copy()
                env.update(overrides)
                result = subprocess.run(
                    [str(LAUNCHER), "dry-run"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def run_bundle_check(self, project_dir, opponent):
        source = LAUNCHER.read_text()
        marker = "acquire_launch_lock() {\n"
        prefix, found, _ = source.partition(marker)
        self.assertEqual(found, marker)
        script = (
            prefix
            + f"PROJECT_DIR={str(project_dir)!r}\n"
            + f"OPPONENT={opponent!r}\n"
            + "SLUMBOT_MODE=0\n"
            + '[ "$OPPONENT" = slumbot ] && SLUMBOT_MODE=1\n'
            + "verify_play_ui_bundle\n"
        )
        return subprocess.run(["bash"], input=script, check=False, capture_output=True, text=True)

    def test_bundle_check_needs_no_web_build_in_slumbot_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "requirements-play-ui.txt",
                "scripts/solver_regression.py",
                "scripts/start_slumbot_remote.sh",
                "scripts/validate_slumbot_benchmark.py",
                "scripts/slumbot_session_status.py",
                "scripts/slumbot_run_report.py",
                "src/player/dyypholdem_slumbot_player.py",
                "src/player/slumbot_match.py",
                "src/server/slumbot_game.py",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("present\n")

            slumbot = self.run_bundle_check(root, "slumbot")
            self.assertEqual(slumbot.returncode, 0, slumbot.stderr)

            human = self.run_bundle_check(root, "human")
            self.assertNotEqual(human.returncode, 0)
            self.assertIn("missing compiled play UI", human.stderr)

            (root / "src/server/slumbot_game.py").unlink()
            broken = self.run_bundle_check(root, "slumbot")
            self.assertNotEqual(broken.returncode, 0)
            self.assertIn("missing Slumbot benchmark component", broken.stderr)

    def test_code_sync_excludes_every_pod_downloaded_play_asset(self):
        import fnmatch
        import re

        source = LAUNCHER.read_text()
        sync_block = re.search(
            r'rsync -az -e "ssh -F \$SSH_CONFIG -o BatchMode=yes" \\\n(?:.*\\\n)*?  "\$\{sync_sources\[@\]\}"',
            source,
        )
        self.assertIsNotNone(sync_block, "code sync rsync block not found")
        excludes = re.findall(r"--exclude '([^']+)'", sync_block.group(0))
        self.assertIn("*.pt", excludes)
        self.assertIn("*.pkl", excludes)

        assets = PROJECT_ROOT / "scripts" / "materialize_assets.py"
        asset_paths = re.findall(r'"(src/[^"]+)"', assets.read_text())
        self.assertGreaterEqual(len(asset_paths), 8)
        for relative in asset_paths:
            self.assertTrue(
                any(fnmatch.fnmatch(Path(relative).name, pattern) for pattern in excludes),
                f"{relative} would be uploaded from the local checkout instead of downloaded on the pod",
            )

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
