#!/usr/bin/env python3
"""Repeat realtime-bench stride sweeps and pick the best exact-match candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STRIDES = ["0.60", "0.55", "0.50", "0.48", "0.45", "0.40"]
SUMMARY_KEYS = [
    "infer_rtfx",
    "corrections",
    "corrections_per_min",
    "draft_word_vis_ms_avg",
    "draft_word_vis_ms_p95",
    "confirmed_word_vis_ms_avg",
    "confirmed_word_vis_ms_p95",
    "draft_onset_latency_ms_avg",
    "draft_onset_latency_ms_p95",
    "confirmed_onset_latency_ms_avg",
    "confirmed_onset_latency_ms_p95",
]
PRIMARY_TO_P95 = {
    "draft_word_vis_ms_avg": "draft_word_vis_ms_p95",
    "confirmed_word_vis_ms_avg": "confirmed_word_vis_ms_p95",
}


@dataclass
class RunResult:
    stride: str
    repetition: int
    run_name: str
    out_dir: Path
    log_path: Path
    summary_path: Path
    confirmed_path: Path
    summary: dict[str, Any]
    confirmed_sha1: str
    exact_match: bool

    def row(self) -> dict[str, Any]:
        row = {
            "stride": self.stride,
            "repetition": self.repetition,
            "run_name": self.run_name,
            "out_dir": str(self.out_dir),
            "log_path": str(self.log_path),
            "summary_path": str(self.summary_path),
            "confirmed_path": str(self.confirmed_path),
            "confirmed_sha1": self.confirmed_sha1,
            "exact_match": self.exact_match,
        }
        for key in SUMMARY_KEYS:
            row[key] = self.summary.get(key)
        return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Specialize rewrite-prefix decode stride for a single device."
    )
    parser.add_argument("--audio", type=Path, required=True, help="Audio clip used for specialization.")
    parser.add_argument(
        "--run-name",
        default=f"rt-prefix-specialize-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        help="Artifact run name prefix.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to artifacts/realtime-specialization-runs/<run-name>.",
    )
    parser.add_argument(
        "--stride",
        dest="strides",
        action="append",
        default=[],
        help="Candidate stride in seconds. Repeat for multiple candidates.",
    )
    parser.add_argument(
        "--baseline-stride",
        default=None,
        help="Baseline stride used for exact-match gating. Defaults to the first candidate or 0.60.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=2,
        help="Number of repeated runs per candidate.",
    )
    parser.add_argument(
        "--min-infer-rtfx",
        type=float,
        default=1.2,
        help="Minimum infer RTFx allowed across repetitions.",
    )
    parser.add_argument(
        "--max-corrections",
        type=int,
        default=0,
        help="Maximum allowed corrections per run.",
    )
    parser.add_argument(
        "--primary-metric",
        choices=sorted(PRIMARY_TO_P95.keys()),
        default="draft_word_vis_ms_avg",
        help="Metric used to define the latency frontier for passing candidates.",
    )
    parser.add_argument(
        "--latency-equivalence-ms",
        type=float,
        default=5.0,
        help=(
            "Treat candidates within this many milliseconds of the best primary-metric mean "
            "as latency-equivalent, then prefer the one with more infer headroom."
        ),
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        help="Extra shared env override in KEY=VALUE form. Repeatable.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_realtime_bench_cli.sh"),
        help="Bench runner script path.",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    return path.resolve()


def _normalize_stride(raw: str) -> str:
    value = float(raw)
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _candidate_id(stride: str) -> str:
    return f"stride_{stride.replace('.', 'p')}"


def _parse_set_env(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in values:
        if "=" not in entry:
            raise SystemExit(f"--set-env must be KEY=VALUE, got: {entry}")
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--set-env key must be non-empty: {entry}")
        result[key] = value
    return result


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.pstdev(values)


def _series(runs: list[RunResult], key: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        value = _safe_float(run.summary.get(key))
        if value is not None:
            values.append(value)
    return values


def _aggregate_candidate(
    stride: str,
    runs: list[RunResult],
    canonical_sha1: str,
    primary_metric: str,
    min_infer_rtfx: float,
    max_corrections: int,
) -> dict[str, Any]:
    primary_p95 = PRIMARY_TO_P95[primary_metric]
    infer_values = _series(runs, "infer_rtfx")
    correction_values = [int(run.summary.get("corrections") or 0) for run in runs]
    exact_match_all = all(run.exact_match and run.confirmed_sha1 == canonical_sha1 for run in runs)

    aggregate = {
        "stride": stride,
        "run_count": len(runs),
        "exact_match_all": exact_match_all,
        "infer_rtfx_mean": _mean(infer_values),
        "infer_rtfx_min": min(infer_values) if infer_values else None,
        "corrections_max": max(correction_values) if correction_values else None,
        "corrections_total": sum(correction_values),
        "draft_word_vis_ms_avg_mean": _mean(_series(runs, "draft_word_vis_ms_avg")),
        "draft_word_vis_ms_avg_stdev": _stdev(_series(runs, "draft_word_vis_ms_avg")),
        "draft_word_vis_ms_p95_mean": _mean(_series(runs, "draft_word_vis_ms_p95")),
        "confirmed_word_vis_ms_avg_mean": _mean(_series(runs, "confirmed_word_vis_ms_avg")),
        "confirmed_word_vis_ms_avg_stdev": _stdev(_series(runs, "confirmed_word_vis_ms_avg")),
        "confirmed_word_vis_ms_p95_mean": _mean(_series(runs, "confirmed_word_vis_ms_p95")),
        "draft_onset_latency_ms_avg_mean": _mean(_series(runs, "draft_onset_latency_ms_avg")),
        "confirmed_onset_latency_ms_avg_mean": _mean(_series(runs, "confirmed_onset_latency_ms_avg")),
        "primary_metric": primary_metric,
        "primary_metric_mean": _mean(_series(runs, primary_metric)),
        "primary_metric_stdev": _stdev(_series(runs, primary_metric)),
        "primary_metric_p95_name": primary_p95,
        "primary_metric_p95_mean": _mean(_series(runs, primary_p95)),
    }

    fail_reasons: list[str] = []
    if not exact_match_all:
        fail_reasons.append("confirmed transcript mismatch")
    infer_min = aggregate["infer_rtfx_min"]
    if infer_min is None or infer_min < min_infer_rtfx:
        fail_reasons.append(f"infer_rtfx < {min_infer_rtfx}")
    corrections_max_seen = aggregate["corrections_max"]
    if corrections_max_seen is None or corrections_max_seen > max_corrections:
        fail_reasons.append(f"corrections > {max_corrections}")

    aggregate["pass"] = not fail_reasons
    aggregate["fail_reasons"] = fail_reasons
    return aggregate


def _latency_rank_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    primary = candidate.get("primary_metric_mean")
    primary_p95 = candidate.get("primary_metric_p95_mean")
    infer_min = candidate.get("infer_rtfx_min")
    return (
        float(primary if primary is not None else float("inf")),
        float(primary_p95 if primary_p95 is not None else float("inf")),
        -float(infer_min if infer_min is not None else 0.0),
    )


def _headroom_rank_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    infer_min = candidate.get("infer_rtfx_min")
    primary_p95 = candidate.get("primary_metric_p95_mean")
    primary = candidate.get("primary_metric_mean")
    return (
        -float(infer_min if infer_min is not None else 0.0),
        float(primary_p95 if primary_p95 is not None else float("inf")),
        float(primary if primary is not None else float("inf")),
    )


def _select_recommended(
    passing: list[dict[str, Any]],
    latency_equivalence_ms: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float | None]:
    if not passing:
        return None, [], None
    sorted_by_latency = sorted(passing, key=lambda entry: _latency_rank_key(entry["aggregate"]))
    best_primary = _safe_float(sorted_by_latency[0]["aggregate"].get("primary_metric_mean"))
    if best_primary is None:
        return sorted_by_latency[0]["aggregate"], sorted_by_latency, None
    cutoff = best_primary + max(0.0, latency_equivalence_ms)
    frontier = [
        entry for entry in sorted_by_latency
        if (_safe_float(entry["aggregate"].get("primary_metric_mean")) or float("inf")) <= cutoff
    ]
    frontier.sort(key=lambda entry: _headroom_rank_key(entry["aggregate"]))
    return frontier[0]["aggregate"], frontier, cutoff


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stride",
        "run_count",
        "pass",
        "exact_match_all",
        "infer_rtfx_mean",
        "infer_rtfx_min",
        "corrections_max",
        "corrections_total",
        "draft_word_vis_ms_avg_mean",
        "draft_word_vis_ms_avg_stdev",
        "draft_word_vis_ms_p95_mean",
        "confirmed_word_vis_ms_avg_mean",
        "confirmed_word_vis_ms_avg_stdev",
        "confirmed_word_vis_ms_p95_mean",
        "draft_onset_latency_ms_avg_mean",
        "confirmed_onset_latency_ms_avg_mean",
        "primary_metric",
        "primary_metric_mean",
        "primary_metric_stdev",
        "primary_metric_p95_name",
        "primary_metric_p95_mean",
        "fail_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_copy = dict(row)
            row_copy["fail_reasons"] = "; ".join(row_copy.get("fail_reasons") or [])
            writer.writerow(row_copy)


def _run_candidate(
    repo_root: Path,
    runner_path: Path,
    output_root: Path,
    run_name: str,
    audio_path: Path,
    stride: str,
    repetition: int,
    shared_env: dict[str, str],
) -> RunResult:
    candidate_id = _candidate_id(stride)
    repetition_name = f"{run_name}-{candidate_id}-rep{repetition:02d}"
    out_dir = output_root / candidate_id / f"rep{repetition:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    summary_path = out_dir / "summary.json"
    confirmed_path = out_dir / "summary.confirmed.txt"

    env = os.environ.copy()
    env.update(shared_env)
    env["AUDIO_PATH"] = str(audio_path)
    env["RUN_NAME"] = repetition_name
    env["OUT_DIR"] = str(out_dir)
    env["PARAKEET_STREAM_PREFIX_DECODE_STRIDE_SEC"] = stride

    print(
        f"[specialize] running stride={stride} repetition={repetition} -> {out_dir}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_fh:
        result = subprocess.run(
            ["bash", str(runner_path)],
            cwd=repo_root,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit(
            f"bench run failed for stride={stride} repetition={repetition}: see {log_path}"
        )
    if not summary_path.exists() or not confirmed_path.exists():
        raise SystemExit(
            f"bench artifacts missing for stride={stride} repetition={repetition}: {out_dir}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    confirmed_text = confirmed_path.read_text(encoding="utf-8")
    return RunResult(
        stride=stride,
        repetition=repetition,
        run_name=repetition_name,
        out_dir=out_dir,
        log_path=log_path,
        summary_path=summary_path,
        confirmed_path=confirmed_path,
        summary=summary,
        confirmed_sha1=_sha1_text(confirmed_text),
        exact_match=False,
    )


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    runner_path = _require_file((repo_root / args.runner).resolve(), "runner")
    audio_path = _require_file(args.audio, "audio").resolve()

    strides = [_normalize_stride(value) for value in (args.strides or DEFAULT_STRIDES)]
    if not strides:
        raise SystemExit("at least one stride candidate is required")
    baseline_stride = _normalize_stride(args.baseline_stride or strides[0] or "0.60")
    if baseline_stride not in strides:
        strides = [baseline_stride] + [value for value in strides if value != baseline_stride]

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (repo_root / "artifacts" / "realtime-specialization-runs" / args.run_name).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    shared_env = _parse_set_env(args.set_env)
    run_results: dict[str, list[RunResult]] = {stride: [] for stride in strides}

    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "audio_path": str(audio_path),
                "output_root": str(output_root),
                "baseline_stride": baseline_stride,
                "strides": strides,
                "repetitions": args.repetitions,
                "primary_metric": args.primary_metric,
                "gates": {
                    "min_infer_rtfx": args.min_infer_rtfx,
                    "max_corrections": args.max_corrections,
                    "exact_confirmed_match": True,
                },
                "selection_policy": {
                    "type": "latency-equivalent-then-headroom",
                    "latency_equivalence_ms": args.latency_equivalence_ms,
                },
            },
            indent=2,
        ),
        flush=True,
    )

    for stride in strides:
        for repetition in range(1, args.repetitions + 1):
            run_results[stride].append(
                _run_candidate(
                    repo_root=repo_root,
                    runner_path=runner_path,
                    output_root=output_root,
                    run_name=args.run_name,
                    audio_path=audio_path,
                    stride=stride,
                    repetition=repetition,
                    shared_env=shared_env,
                )
            )

    baseline_runs = run_results[baseline_stride]
    canonical_sha1 = baseline_runs[0].confirmed_sha1
    baseline_consistent = all(run.confirmed_sha1 == canonical_sha1 for run in baseline_runs)

    for runs in run_results.values():
        for run in runs:
            run.exact_match = run.confirmed_sha1 == canonical_sha1

    candidate_reports: list[dict[str, Any]] = []
    for stride in strides:
        runs = run_results[stride]
        aggregate = _aggregate_candidate(
            stride=stride,
            runs=runs,
            canonical_sha1=canonical_sha1,
            primary_metric=args.primary_metric,
            min_infer_rtfx=args.min_infer_rtfx,
            max_corrections=args.max_corrections,
        )
        if not baseline_consistent:
            aggregate["pass"] = False
            aggregate["fail_reasons"] = list(aggregate["fail_reasons"]) + [
                "baseline transcript unstable across repetitions"
            ]
        candidate_reports.append(
            {
                "stride": stride,
                "aggregate": aggregate,
                "runs": [run.row() for run in runs],
            }
        )

    passing = [entry for entry in candidate_reports if entry["aggregate"]["pass"]]
    recommended, latency_frontier, latency_cutoff = _select_recommended(
        passing,
        latency_equivalence_ms=args.latency_equivalence_ms,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "audio_path": str(audio_path),
        "run_name": args.run_name,
        "output_root": str(output_root),
        "baseline_stride": baseline_stride,
        "baseline_confirmed_sha1": canonical_sha1,
        "baseline_consistent": baseline_consistent,
        "primary_metric": args.primary_metric,
        "gates": {
            "min_infer_rtfx": args.min_infer_rtfx,
            "max_corrections": args.max_corrections,
            "exact_confirmed_match": True,
        },
        "selection_policy": {
            "type": "latency-equivalent-then-headroom",
            "latency_equivalence_ms": args.latency_equivalence_ms,
            "latency_equivalent_primary_metric_cutoff_ms": latency_cutoff,
            "latency_equivalent_strides": [
                entry["aggregate"]["stride"] for entry in latency_frontier
            ],
        },
        "shared_env_overrides": shared_env,
        "candidates": candidate_reports,
        "recommended": recommended,
    }

    report_path = output_root / "report.json"
    csv_path = output_root / "candidate_summary.csv"
    env_path = output_root / "recommended.env"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, [entry["aggregate"] for entry in candidate_reports])

    env_lines = [
        f"# generated by {Path(__file__).name}",
        f"# run_name={args.run_name}",
        f"# primary_metric={args.primary_metric}",
        f"# latency_equivalence_ms={args.latency_equivalence_ms}",
    ]
    if recommended is None:
        env_lines.append("# no passing candidate found")
    else:
        env_lines.append(
            f"# recommended_stride={recommended['stride']} "
            f"primary_mean={recommended['primary_metric_mean']:.3f} "
            f"primary_p95_mean={recommended['primary_metric_p95_mean']:.3f} "
            f"infer_rtfx_min={recommended['infer_rtfx_min']:.3f}"
        )
        if latency_frontier:
            env_lines.append(
                "# latency_equivalent_strides=" +
                ",".join(entry["aggregate"]["stride"] for entry in latency_frontier)
            )
        env_lines.append(
            f"export PARAKEET_STREAM_PREFIX_DECODE_STRIDE_SEC={recommended['stride']}"
        )
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    summary_payload = {
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "env_path": str(env_path),
        "recommended_stride": recommended["stride"] if recommended is not None else None,
        "baseline_consistent": baseline_consistent,
        "passing_candidates": [entry["aggregate"]["stride"] for entry in passing],
        "latency_equivalent_candidates": [
            entry["aggregate"]["stride"] for entry in latency_frontier
        ],
    }
    print(json.dumps(summary_payload, indent=2), flush=True)
    if recommended is None:
        sys.exit(2)


if __name__ == "__main__":
    main()
