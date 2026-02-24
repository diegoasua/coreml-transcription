#!/usr/bin/env python3
"""Run OpenBench with a custom local transcription command pipeline.

This script integrates with a cloned OpenBench repository and evaluates a
command-based ASR backend on standardized OpenBench datasets (for example,
`librispeech`, `earnings22`, `common-voice-en`).

Example (run from repo root):
  cd external/OpenBench
  uv run python ../../scripts/run_openbench_custom_transcription.py \
    --openbench-dir . \
    --dataset librispeech-200 \
    --transcribe-cmd "python ../../scripts/your_transcriber.py --audio {audio_path}" \
    --run-name parakeet-coreml-ls200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run OpenBench evaluation with a custom transcription command.")
    parser.add_argument("--openbench-dir", type=Path, default=Path("external/OpenBench"))
    parser.add_argument("--run-name", type=str, default="custom-transcription")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "artifacts/openbench-runs")
    parser.add_argument(
        "--command-cwd",
        type=Path,
        default=repo_root,
        help="Working directory used when running --transcribe-cmd.",
    )

    dataset_group = parser.add_mutually_exclusive_group(required=False)
    dataset_group.add_argument("--dataset", type=str, default="librispeech-200", help="OpenBench dataset alias.")
    dataset_group.add_argument("--dataset-id", type=str, default=None, help="Direct HF dataset id.")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--dataset-name", type=str, default="custom-dataset", help="Name used in OpenBench output.")

    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument(
        "--transcribe-cmd",
        type=str,
        default=None,
        help="Command template, must contain {audio_path}.",
    )
    backend_group.add_argument(
        "--python-transcriber",
        type=Path,
        default=None,
        help="Path to a python file exporting transcribe_file(audio_path, language=None, keywords=None) -> str.",
    )
    parser.add_argument("--stdout-json-key", type=str, default=None, help="If set, parse stdout as JSON and read this key.")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--normalize", choices=("none", "basic"), default="none")
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--temp-audio-dir", type=Path, default=repo_root / "artifacts/openbench-temp-audio")

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["wer"],
        help="OpenBench metric names (default: wer).",
    )
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()


def _bootstrap_openbench(openbench_dir: Path) -> None:
    src_dir = openbench_dir / "src"
    if not src_dir.exists():
        raise SystemExit(f"OpenBench src directory not found: {src_dir}")
    sys.path.insert(0, str(src_dir))


def _ensure_texterrors_importable() -> None:
    """Keep OpenBench importable when texterrors native wheel is broken on macOS.

    OpenBench imports keyword metrics at module import-time, which imports
    `texterrors` even when we only evaluate WER. If that compiled extension fails
    to load, we provide a minimal stub so non-keyword metrics still run.
    """
    try:
        import texterrors  # type: ignore  # noqa: F401

        return
    except Exception as exc:
        msg = str(exc).lower()
        if "texterrors_align" not in msg and "symbol not found" not in msg and "dlopen" not in msg:
            # Unknown failure: don't hide it.
            raise

    stub = types.ModuleType("texterrors")

    def _unsupported_align_texts(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "texterrors is unavailable in this environment. "
            "Keyword metrics are disabled for this run."
        )

    stub.align_texts = _unsupported_align_texts  # type: ignore[attr-defined]
    stub.__dict__["__version__"] = "stub"
    sys.modules["texterrors"] = stub
    print("[warning] texterrors native module failed to load; keyword metrics disabled (WER still available).")


def _normalize_text(text: str, mode: str) -> str:
    if mode == "none":
        return text.strip()
    lowered = text.lower().strip()
    stripped = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    squashed = re.sub(r"\s+", " ", stripped).strip()
    return squashed


@dataclass
class CommandInput:
    audio_path: Path
    language: str | None = None
    keywords: list[str] | None = None


def main() -> None:
    args = _parse_args()
    if args.transcribe_cmd and "{audio_path}" not in args.transcribe_cmd:
        raise SystemExit("--transcribe-cmd must contain '{audio_path}'.")

    _bootstrap_openbench(args.openbench_dir)
    _ensure_texterrors_importable()

    try:
        import soundfile as sf  # type: ignore
        from pydantic import Field

        from openbench.dataset import DatasetConfig, DatasetRegistry, TranscriptionSample
        from openbench.metric import MetricOptions
        from openbench.pipeline.base import Pipeline, register_pipeline
        from openbench.pipeline.transcription.common import TranscriptionConfig, TranscriptionOutput
        from openbench.pipeline_prediction import Transcript
        from openbench.runner import BenchmarkConfig, BenchmarkRunner, WandbConfig
        from openbench.types import PipelineType
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Failed to import OpenBench dependencies. "
            "Run this script inside OpenBench env, for example:\n"
            "  cd external/OpenBench && uv sync && "
            "uv run python ../../scripts/run_openbench_custom_transcription.py ...\n"
            f"Import error: {exc}"
        ) from exc

    class CommandTranscriptionConfig(TranscriptionConfig):
        python_transcriber: str | None = Field(
            None,
            description=(
                "Path to python transcriber module that exports "
                "transcribe_file(audio_path, language=None, keywords=None) -> str."
            ),
        )
        command_template: str = Field(..., description="Shell command template with {audio_path} placeholder.")
        stdout_json_key: str | None = Field(None, description="If set, parse stdout JSON and read this key.")
        timeout_sec: float = Field(180.0, description="Per-sample inference timeout.")
        normalize: str = Field("none", description="Optional output normalization: none|basic.")
        keep_audio: bool = Field(False, description="Keep temp audio files after inference.")
        temp_audio_dir: str = Field("artifacts/openbench-temp-audio", description="Directory for temporary audio files.")
        command_cwd: str = Field(".", description="Working directory for command execution.")

    @register_pipeline
    class CommandTranscriptionPipeline(Pipeline):
        _config_class = CommandTranscriptionConfig
        pipeline_type = PipelineType.TRANSCRIPTION

        def build_pipeline(self) -> Callable[[CommandInput], str]:
            python_transcriber = self.config.python_transcriber
            command_template = self.config.command_template
            timeout_sec = float(self.config.timeout_sec)
            stdout_json_key = self.config.stdout_json_key
            normalize_mode = self.config.normalize
            command_cwd = Path(self.config.command_cwd)
            command_cwd.mkdir(parents=True, exist_ok=True)

            if python_transcriber:
                transcriber_path = Path(python_transcriber)
                if not transcriber_path.is_absolute():
                    transcriber_path = (command_cwd / transcriber_path).resolve()
                if not transcriber_path.exists():
                    raise RuntimeError(f"Python transcriber module not found: {transcriber_path}")

                module_name = f"openbench_custom_transcriber_{abs(hash(str(transcriber_path)))}"
                spec = importlib.util.spec_from_file_location(module_name, transcriber_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"Could not load python transcriber module: {transcriber_path}")
                module = importlib.util.module_from_spec(spec)
                # Ensure decorators/types that inspect sys.modules (e.g. dataclass)
                # can resolve the module during execution.
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                transcribe_fn = getattr(module, "transcribe_file", None)
                if transcribe_fn is None or not callable(transcribe_fn):
                    raise RuntimeError(
                        f"Transcriber module '{transcriber_path}' must export callable "
                        "transcribe_file(audio_path, language=None, keywords=None) -> str"
                    )
                warmup_fn = getattr(module, "warmup", None)
                if callable(warmup_fn):
                    warmup_fn()

                def _run_python(payload: CommandInput) -> str:
                    text = transcribe_fn(
                        str(payload.audio_path),
                        language=payload.language,
                        keywords=payload.keywords,
                    )
                    return _normalize_text(str(text), normalize_mode)

                return _run_python

            def _run(payload: CommandInput) -> str:
                format_kwargs = {
                    "audio_path": shlex.quote(str(payload.audio_path)),
                    "language": shlex.quote(payload.language or ""),
                    "keywords_csv": shlex.quote(",".join(payload.keywords or [])),
                    "keywords_json": shlex.quote(json.dumps(payload.keywords or [])),
                    "command_cwd": shlex.quote(str(command_cwd)),
                }
                cmd = command_template.format(**format_kwargs)
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                    cwd=command_cwd,
                )
                if completed.returncode != 0:
                    err = completed.stderr.strip() or completed.stdout.strip()
                    raise RuntimeError(f"Command failed (code={completed.returncode}): {err[:400]}")

                raw_out = completed.stdout.strip()
                if stdout_json_key:
                    try:
                        payload_json = json.loads(raw_out)
                        text = str(payload_json.get(stdout_json_key, ""))
                    except Exception as exc:
                        raise RuntimeError("stdout was not valid JSON for configured --stdout-json-key") from exc
                else:
                    text = raw_out
                return _normalize_text(text, normalize_mode)

            return _run

        def parse_input(self, input_sample: TranscriptionSample) -> CommandInput:
            out_dir = Path(self.config.temp_audio_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            audio_path = out_dir / f"{input_sample.audio_name}.wav"
            sf.write(str(audio_path), input_sample.waveform, input_sample.sample_rate)
            return CommandInput(
                audio_path=audio_path,
                language=input_sample.language if self.config.force_language else None,
                keywords=input_sample.dictionary if self.config.use_keywords else None,
            )

        def parse_output(self, output: str) -> TranscriptionOutput:
            words = output.strip().split()
            return TranscriptionOutput(
                prediction=Transcript.from_words_info(
                    words=words,
                    start=None,
                    end=None,
                    speaker=None,
                )
            )

        def __call__(self, input_sample: TranscriptionSample) -> TranscriptionOutput:
            parsed_input = self.parse_input(input_sample)
            try:
                start = time.perf_counter()
                output = self.pipeline(parsed_input)
                end = time.perf_counter()
                parsed_output = self.parse_output(output)
                if parsed_output.prediction_time is None:
                    parsed_output.prediction_time = end - start
                return parsed_output
            finally:
                if not self.config.keep_audio:
                    parsed_input.audio_path.unlink(missing_ok=True)

    # Resolve dataset config.
    if args.dataset_id:
        dataset_cfg = DatasetConfig(
            dataset_id=args.dataset_id,
            subset=args.subset,
            split=args.split,
            num_samples=args.num_samples,
        )
        dataset_name = args.dataset_name
    else:
        if not DatasetRegistry.has_alias(args.dataset):
            available = ", ".join(DatasetRegistry.list_aliases()[:25])
            raise SystemExit(f"Unknown dataset alias: {args.dataset}. Example aliases: {available} ...")
        base_cfg = DatasetRegistry.get_alias_config(args.dataset)
        update_kwargs: dict[str, Any] = {}
        if args.subset is not None:
            update_kwargs["subset"] = args.subset
        if args.split is not None:
            update_kwargs["split"] = args.split
        if args.num_samples is not None:
            update_kwargs["num_samples"] = args.num_samples
        dataset_cfg = base_cfg.model_copy(update=update_kwargs)
        dataset_name = args.dataset

    metric_map = {}
    for metric_name in args.metrics:
        try:
            metric_option = MetricOptions(metric_name)
        except ValueError as exc:
            valid = ", ".join(m.value for m in MetricOptions)
            raise SystemExit(f"Invalid metric '{metric_name}'. Valid metrics: {valid}") from exc
        metric_map[metric_option] = {}

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline = CommandTranscriptionPipeline(
        CommandTranscriptionConfig(
            out_dir=str(run_dir),
            python_transcriber=str(args.python_transcriber) if args.python_transcriber else None,
            command_template=args.transcribe_cmd or "",
            stdout_json_key=args.stdout_json_key,
            timeout_sec=args.timeout_sec,
            normalize=args.normalize,
            keep_audio=args.keep_audio,
            temp_audio_dir=str(args.temp_audio_dir),
            command_cwd=str(args.command_cwd),
        )
    )
    benchmark_config = BenchmarkConfig(
        wandb_config=WandbConfig(
            project_name="openbench-custom-local",
            run_name=args.run_name,
            is_active=args.use_wandb,
            tags=["custom", "local-command"],
        ),
        metrics=metric_map,
        datasets={dataset_name: dataset_cfg},
    )

    runner = BenchmarkRunner(config=benchmark_config, pipelines=[pipeline])
    result = runner.run()

    total_audio = sum(sample.audio_duration for sample in result.sample_results)
    total_pred = sum(sample.prediction_time for sample in result.sample_results)
    rtf = total_pred / max(total_audio, 1e-9)
    speed_x = total_audio / max(total_pred, 1e-9)

    global_metrics = {
        g.metric_name: {
            "global_result": g.global_result,
            "avg_result": g.avg_result,
            "lower_bound": g.lower_bound,
            "upper_bound": g.upper_bound,
        }
        for g in result.global_results
    }
    summary = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "dataset_name": dataset_name,
        "dataset_config": dataset_cfg.model_dump(),
        "num_samples": len(result.sample_results),
        "global_metrics": global_metrics,
        "total_audio_sec": total_audio,
        "total_prediction_sec": total_pred,
        "rtf": rtf,
        "speed_x": speed_x,
    }

    summary_path = run_dir / "custom-openbench-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
