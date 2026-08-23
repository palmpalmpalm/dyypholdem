import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import tempfile
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "run_play_ui.sh"


class RunPlayUiLauncherTests(unittest.TestCase):
    def test_dry_run_reports_detached_controller(self):
        result = subprocess.run(
            [str(LAUNCHER), "dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("controller: detached locally", result.stdout)
        self.assertIn("hard guard: 3600 seconds", result.stdout)

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


if __name__ == "__main__":
    unittest.main()
