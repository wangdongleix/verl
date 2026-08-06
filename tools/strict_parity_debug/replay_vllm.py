"""Call the monkey-patched vLLM Ray server replay RPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _select_actor_entry(entries: list[dict[str, str]], requested_name: str | None) -> dict[str, str]:
    rollout_entries = [
        entry
        for entry in entries
        if entry.get("name", "").startswith("vllm_server_")
        and "server_decode_" not in entry.get("name", "")
        and "server_reward_" not in entry.get("name", "")
        and "server_teacher_" not in entry.get("name", "")
    ]
    exact = [entry for entry in rollout_entries if entry.get("name") == requested_name] if requested_name else []
    candidates = exact or rollout_entries
    if not candidates:
        available = ", ".join(
            f"{entry.get('namespace', '')}/{entry.get('name', '')}" for entry in entries
        ) or "<none>"
        raise RuntimeError(f"no live vLLM server actor found; available named actors: {available}")
    if len(candidates) > 1:
        rank_zero = [entry for entry in candidates if entry.get("name", "").startswith("vllm_server_0_0")]
        if len(rank_zero) == 1:
            return rank_zero[0]
        choices = ", ".join(f"{entry.get('namespace', '')}/{entry.get('name', '')}" for entry in candidates)
        raise RuntimeError(
            "multiple vLLM server actors found; pass --actor-name and --actor-namespace explicitly: " + choices
        )
    return candidates[0]


def _resolve_actor(ray: Any, actor_name: str | None, actor_namespace: str | None):
    if actor_namespace:
        return ray.get_actor(actor_name or "vllm_server_0_0", namespace=actor_namespace)
    try:
        entries = ray.util.list_named_actors(all_namespaces=True)
    except Exception as exc:
        if actor_name:
            try:
                return ray.get_actor(actor_name)
            except ValueError:
                pass
        raise RuntimeError("failed to list Ray actors across namespaces; pass --actor-namespace explicitly") from exc
    normalized = [
        {"name": str(entry.get("name") or ""), "namespace": str(entry.get("namespace") or "")}
        for entry in entries
        if isinstance(entry, dict)
    ]
    selected = _select_actor_entry(normalized, actor_name)
    print(
        "[STRICT-PARITY] using Ray actor "
        f"name={selected['name']!r} namespace={selected['namespace']!r}",
        flush=True,
    )
    return ray.get_actor(selected["name"], namespace=selected["namespace"])


def _invoke_replay(ray: Any, actor: Any) -> dict:
    """Trigger replay without RPC arguments; actor startup settings are authoritative."""
    return ray.get(actor.strict_parity_replay.remote())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-name", default="vllm_server_0_0")
    parser.add_argument("--actor-namespace", help="Ray namespace; auto-detected across namespaces by default")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--msprobe-config", help="enable a fresh rollout msprobe debugger only for this replay")
    parser.add_argument("--msprobe-dump-path", help="output root for the replay-only msprobe debugger")
    args = parser.parse_args()
    if bool(args.msprobe_config) != bool(args.msprobe_dump_path):
        parser.error("--msprobe-config and --msprobe-dump-path must be provided together")

    import ray

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    actor = _resolve_actor(ray, args.actor_name, args.actor_namespace)
    result = _invoke_replay(ray, actor)
    expected_replay = str(Path(args.replay).expanduser().resolve())
    actual_replay = result.get("replay_path")
    if actual_replay is None:
        raise RuntimeError(
            "vLLM actor is using an outdated strict-parity server patch: replay path cannot be verified. "
            "Sync the entire tools/strict_parity_debug directory and restart the training job."
        )
    if str(Path(actual_replay).expanduser().resolve()) != expected_replay:
        raise RuntimeError(f"vLLM actor replay path mismatch: expected={expected_replay}, actual={actual_replay}")
    if args.sample_index is not None and result.get("sample_index") != args.sample_index:
        raise RuntimeError(
            f"vLLM actor sample index mismatch: expected={args.sample_index}, actual={result.get('sample_index')}"
        )
    if args.msprobe_config:
        expected_config = str(Path(args.msprobe_config).expanduser().resolve())
        actual_config = result.get("msprobe_config_path")
        if actual_config is None:
            raise RuntimeError(
                "vLLM actor is using an outdated strict-parity server patch: replay ran without replay-only msprobe. "
                "Sync the entire tools/strict_parity_debug directory and restart the training job."
            )
        if str(Path(actual_config).expanduser().resolve()) != expected_config:
            raise RuntimeError(
                f"vLLM actor msprobe config mismatch: expected={expected_config}, actual={actual_config}"
            )
    if args.msprobe_dump_path:
        expected_dump_path = str(Path(args.msprobe_dump_path).expanduser().resolve())
        actual_dump_path = result.get("msprobe_dump_path")
        if actual_dump_path is None:
            raise RuntimeError("vLLM actor replay ran without an explicit msprobe dump path")
        if str(Path(actual_dump_path).expanduser().resolve()) != expected_dump_path:
            raise RuntimeError(
                f"vLLM actor msprobe dump path mismatch: expected={expected_dump_path}, actual={actual_dump_path}"
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
