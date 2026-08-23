#!/usr/bin/env python3
"""Safe RunPod lifecycle helper for an authenticated DyypHoldem web session."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


REST_BASE_URL = "https://rest.runpod.io/v1"
# Current image behind RunPod's official ``runpod-torch-v280`` template.
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"


class PodNotFound(RuntimeError):
    """The RunPod API explicitly returned HTTP 404 for a pod."""


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return key


def request(method: str, path: str, payload: dict | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        REST_BASE_URL + path,
        data=body,
        method=method,
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        if error.code == 404:
            raise PodNotFound(path) from error
        raise SystemExit(f"RunPod API HTTP {error.code}: {detail}")
    except urllib.error.URLError as error:
        raise SystemExit(f"RunPod API transport error: {error.reason}") from error
    return json.loads(raw) if raw.strip() else {}


def build_create_payload(
    name: str,
    public_key: str,
    gpu_type: str = "NVIDIA GeForce RTX 4090",
    cloud_type: str = "SECURE",
    image: str = DEFAULT_IMAGE,
    disk_gb: int = 30,
    http_port: int = 8000,
) -> dict:
    if cloud_type not in ("COMMUNITY", "SECURE"):
        raise ValueError("cloud_type must be COMMUNITY or SECURE")
    if disk_gb < 5:
        raise ValueError("disk_gb must be at least 5")
    if not 1 <= http_port <= 65535 or http_port == 22:
        raise ValueError("http_port must be a valid non-SSH port")
    return {
        "name": name,
        "imageName": image,
        "computeType": "GPU",
        "cloudType": cloud_type,
        "gpuTypeIds": [gpu_type],
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "ports": ["22/tcp", f"{http_port}/http"],
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": public_key},
    }


def safe_status(pod: dict) -> dict:
    ports = [str(value) for value in (pod.get("ports") or [])]
    mappings = {str(key): value for key, value in (pod.get("portMappings") or {}).items()}
    return {
        "id": pod.get("id"),
        "name": pod.get("name"),
        "desiredStatus": pod.get("desiredStatus"),
        "costPerHr": pod.get("costPerHr"),
        "gpuCount": pod.get("gpuCount"),
        "memoryInGb": pod.get("memoryInGb"),
        "vcpuCount": pod.get("vcpuCount"),
        "cloudType": pod.get("cloudType"),
        "gpuType": (pod.get("machine") or {}).get("gpuTypeId") or pod.get("gpuTypeId"),
        "httpPorts": sorted(
            int(value.split("/", 1)[0]) for value in ports if value.endswith("/http")
        ),
        "publicIpPresent": bool(pod.get("publicIp")),
        "sshPortMapped": "22" in mappings,
    }


def ssh_config(alias: str, ip: str, port: int) -> str:
    return (
        f"Host {alias}\n"
        f"  HostName {ip}\n"
        f"  Port {port}\n"
        "  User root\n"
        "  IdentityFile ~/.ssh/id_ed25519\n"
        "  StrictHostKeyChecking accept-new\n"
    )


def cmd_create(args):
    key = Path(args.public_key_file).expanduser().read_text(encoding="utf-8").strip()
    pod = request(
        "POST",
        "/pods",
        build_create_payload(
            args.name,
            key,
            gpu_type=args.gpu_type,
            cloud_type=args.cloud_type,
            image=args.image,
            disk_gb=args.disk_gb,
            http_port=args.http_port,
        ),
    )
    print(json.dumps(safe_status(pod), indent=2, sort_keys=True))


def cmd_list(_args):
    pods = request("GET", "/pods")
    if not isinstance(pods, list):
        raise SystemExit("RunPod returned an invalid pod list")
    print(json.dumps([safe_status(pod) for pod in pods], indent=2, sort_keys=True))


def cmd_status(args):
    try:
        pod = request("GET", f"/pods/{args.pod_id}")
    except PodNotFound:
        raise SystemExit(f"RunPod pod not found: {args.pod_id}") from None
    print(json.dumps(safe_status(pod), indent=2, sort_keys=True))


def cmd_exists(args):
    """Report explicit presence/absence without treating transport errors as absence."""
    try:
        pod = request("GET", f"/pods/{args.pod_id}")
    except PodNotFound:
        result = {"exists": False, "id": args.pod_id}
    else:
        result = {"exists": True, "pod": safe_status(pod)}
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_ssh_config(args):
    try:
        pod = request("GET", f"/pods/{args.pod_id}")
    except PodNotFound:
        raise SystemExit(f"RunPod pod not found: {args.pod_id}") from None
    mappings = {str(key): value for key, value in (pod.get("portMappings") or {}).items()}
    if not pod.get("publicIp") or not mappings.get("22"):
        raise SystemExit("pod has no SSH mapping yet")
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ssh_config(args.alias, str(pod["publicIp"]), int(mappings["22"])), encoding="utf-8")
    output.chmod(0o600)
    print(json.dumps({"id": args.pod_id, "sshConfigWritten": True}))


def cmd_public_url(args):
    try:
        pod = request("GET", f"/pods/{args.pod_id}")
    except PodNotFound:
        raise SystemExit(f"RunPod pod not found: {args.pod_id}") from None
    ports = {str(value) for value in (pod.get("ports") or [])}
    if f"{args.http_port}/http" not in ports:
        raise SystemExit(f"pod does not expose {args.http_port}/http")
    print(json.dumps({"id": args.pod_id, "url": f"https://{args.pod_id}-{args.http_port}.proxy.runpod.net"}))


def cmd_stop(args):
    try:
        request("POST", f"/pods/{args.pod_id}/stop")
    except PodNotFound:
        print(json.dumps({"exists": False, "id": args.pod_id}, indent=2, sort_keys=True))
        return
    cmd_status(args)


def cmd_terminate(args):
    try:
        request("DELETE", f"/pods/{args.pod_id}")
    except PodNotFound:
        print(json.dumps({"exists": False, "id": args.pod_id}, indent=2, sort_keys=True))
        return
    print(json.dumps({"id": args.pod_id, "terminated": True}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    create.add_argument("--cloud-type", choices=("COMMUNITY", "SECURE"), default="SECURE")
    create.add_argument("--image", default=DEFAULT_IMAGE)
    create.add_argument("--disk-gb", type=int, default=30)
    create.add_argument("--http-port", type=int, default=8000)
    create.add_argument("--public-key-file", default="~/.ssh/id_ed25519.pub")
    create.set_defaults(func=cmd_create)

    listing = commands.add_parser("list")
    listing.set_defaults(func=cmd_list)
    for name, func in (
        ("status", cmd_status),
        ("exists", cmd_exists),
        ("stop", cmd_stop),
        ("terminate", cmd_terminate),
    ):
        command = commands.add_parser(name)
        command.add_argument("--pod-id", required=True)
        command.set_defaults(func=func)
    config = commands.add_parser("ssh-config")
    config.add_argument("--pod-id", required=True)
    config.add_argument("--out", required=True)
    config.add_argument("--alias", default="dyyui")
    config.set_defaults(func=cmd_ssh_config)
    public_url = commands.add_parser("public-url")
    public_url.add_argument("--pod-id", required=True)
    public_url.add_argument("--http-port", type=int, default=8000)
    public_url.set_defaults(func=cmd_public_url)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
