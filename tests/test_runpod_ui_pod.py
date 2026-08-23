#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from runpod_ui_pod import PodNotFound, build_create_payload, cmd_exists, safe_status  # noqa: E402


class RunPodUiPodTest(unittest.TestCase):
    def test_payload_exposes_only_ssh_and_ui(self):
        payload = build_create_payload("test", "ssh-key")
        self.assertEqual(payload["gpuTypeIds"], ["NVIDIA GeForce RTX 4090"])
        self.assertEqual(payload["cloudType"], "SECURE")
        self.assertEqual(payload["ports"], ["22/tcp", "8000/http"])
        self.assertEqual(payload["gpuCount"], 1)

    def test_safe_status_redacts_network_endpoint(self):
        result = safe_status(
            {
                "id": "pod",
                "name": "test",
                "publicIp": "192.0.2.1",
                "ports": ["22/tcp", "8000/http"],
                "portMappings": {"22": 12345},
            }
        )
        self.assertTrue(result["publicIpPresent"])
        self.assertEqual(result["httpPorts"], [8000])
        self.assertNotIn("publicIp", result)

    def test_invalid_http_port_is_rejected(self):
        with self.assertRaises(ValueError):
            build_create_payload("test", "ssh-key", http_port=22)

    def test_exists_reports_only_explicit_not_found_as_absent(self):
        args = mock.Mock(pod_id="missing")
        with mock.patch("runpod_ui_pod.request", side_effect=PodNotFound("/pods/missing")):
            with mock.patch("builtins.print") as output:
                cmd_exists(args)
        self.assertIn('"exists": false', output.call_args.args[0])

    def test_exists_does_not_convert_transport_failure_to_absence(self):
        args = mock.Mock(pod_id="unknown")
        with mock.patch("runpod_ui_pod.request", side_effect=SystemExit("transport error")):
            with self.assertRaisesRegex(SystemExit, "transport error"):
                cmd_exists(args)


if __name__ == "__main__":
    unittest.main()
