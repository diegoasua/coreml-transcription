#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_KEYS = ["chunk_index", "t", "token_id", "duration_idx", "skip", "prev_token"]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "step":
            continue
        out.append(item)
    return out


def _signature(event: dict[str, Any], keys: list[str]) -> tuple[Any, ...]:
    return tuple(event.get(key) for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Swift vs Python decoder per-step traces.")
    parser.add_argument("--swift-trace", required=True, type=Path, help="Swift decoder trace JSONL path.")
    parser.add_argument("--python-trace", required=True, type=Path, help="Python decoder trace JSONL path.")
    parser.add_argument(
        "--keys",
        default=",".join(DEFAULT_KEYS),
        help=f"Comma-separated comparison keys (default: {','.join(DEFAULT_KEYS)}).",
    )
    parser.add_argument("--max-events", type=int, default=0, help="Cap compared events (0 = all).")
    parser.add_argument("--show-context", type=int, default=2, help="Number of neighboring events to print.")
    args = parser.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not keys:
        raise SystemExit("No comparison keys provided.")

    swift_events = _load_jsonl(args.swift_trace)
    python_events = _load_jsonl(args.python_trace)

    if args.max_events > 0:
        limit = args.max_events
        swift_events = swift_events[:limit]
        python_events = python_events[:limit]

    n_swift = len(swift_events)
    n_python = len(python_events)
    n_compare = min(n_swift, n_python)

    mismatch_index: int | None = None
    for idx in range(n_compare):
        if _signature(swift_events[idx], keys) != _signature(python_events[idx], keys):
            mismatch_index = idx
            break

    length_mismatch = n_swift != n_python
    if mismatch_index is None and length_mismatch:
        mismatch_index = n_compare

    result: dict[str, Any] = {
        "keys": keys,
        "swift_events": n_swift,
        "python_events": n_python,
        "compared_events": n_compare,
        "length_mismatch": length_mismatch,
        "first_divergence_index": mismatch_index,
        "match": mismatch_index is None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if mismatch_index is not None:
        ctx = max(0, args.show_context)
        lo = max(0, mismatch_index - ctx)
        hi = mismatch_index + ctx + 1
        print("\n[first divergence context]")
        for idx in range(lo, hi):
            swift_event = swift_events[idx] if idx < n_swift else None
            python_event = python_events[idx] if idx < n_python else None
            row = {
                "index": idx,
                "swift": swift_event,
                "python": python_event,
                "swift_sig": _signature(swift_event, keys) if swift_event is not None else None,
                "python_sig": _signature(python_event, keys) if python_event is not None else None,
            }
            print(json.dumps(row, ensure_ascii=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
