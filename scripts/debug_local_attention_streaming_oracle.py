#!/usr/bin/env python3
"""Compare local-attention encoder outputs across native, patched, and cached streaming paths."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from make_streaming_local_attention_encoder_torchscript import (
    _make_local_attention_export_friendly,
)


def _resolve_streaming_value(raw: int | list[int], index: int) -> int:
    if isinstance(raw, list):
        return int(raw[index])
    return int(raw)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare cached local-attention streaming encoder outputs against full-prefix references."
    )
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="NeMo model name. Example: nvidia/parakeet-tdt-0.6b-v2",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("/Users/diegoasua/Downloads/sophie_deepdive.wav"),
        help="Audio file used to generate feature chunks.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=3.0,
        help="Trim input audio before feature extraction.",
    )
    parser.add_argument("--left-context-steps", type=int, default=80)
    parser.add_argument("--right-context-steps", type=int, default=0)
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument("--shift-steps", type=int, default=4)
    parser.add_argument(
        "--left-chunks",
        type=int,
        default=None,
        help="Defaults to left_context_steps / chunk_steps, matching the exporter.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Number of streaming steps to compare.",
    )
    parser.add_argument(
        "--rel-rmse-threshold",
        type=float,
        default=0.05,
        help="Threshold used when reporting the first divergent layer.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "artifacts" / "debug-local-attention-oracle" / "oracle-summary.json",
    )
    return parser.parse_args()


def _load_audio(path: Path, max_audio_seconds: float | None) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"audio not found: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if max_audio_seconds is not None and max_audio_seconds > 0:
        limit = int(sample_rate * max_audio_seconds)
        audio = audio[:limit]
    return audio.astype(np.float32, copy=False), int(sample_rate)


def _extract_features(model: Any, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
    if sample_rate != 16000:
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("librosa is required when resampling audio.") from exc
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000).astype(np.float32, copy=False)
        sample_rate = 16000

    signal = torch.from_numpy(audio).float().unsqueeze(0)
    signal_length = torch.tensor([signal.shape[-1]], dtype=torch.int64)
    with torch.inference_mode():
        features, _ = model.preprocessor(input_signal=signal, length=signal_length)
    return features.detach().cpu().to(dtype=torch.float32)


def _make_stream_chunk(
    features: torch.Tensor,
    *,
    step_index: int,
    input_feature_frames: int,
    shift_feature_frames: int,
) -> tuple[torch.Tensor | None, int]:
    target_end = (step_index + 1) * shift_feature_frames
    total_frames = int(features.shape[-1])
    if target_end > total_frames:
        return None, target_end
    copy_frames = min(input_feature_frames, target_end)
    source_start = max(0, target_end - copy_frames)
    destination_start = input_feature_frames - copy_frames
    chunk = torch.zeros((1, int(features.shape[1]), input_feature_frames), dtype=features.dtype)
    chunk[:, :, destination_start : destination_start + copy_frames] = features[:, :, source_start:target_end]
    return chunk, target_end


def _make_reference_prefix(
    features: torch.Tensor,
    *,
    target_end: int,
    left_padding_frames: int,
) -> torch.Tensor:
    prefix = torch.zeros((1, int(features.shape[1]), left_padding_frames + target_end), dtype=features.dtype)
    if target_end > 0:
        prefix[:, :, left_padding_frames:] = features[:, :, :target_end]
    return prefix


def _metric_bundle(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.to(dtype=torch.float32)
    cand = candidate.to(dtype=torch.float32)
    if tuple(ref.shape) != tuple(cand.shape):
        return {
            "shape_match": False,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
        }
    diff = cand - ref
    ref_rms = float(torch.sqrt(torch.mean(ref * ref)).item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    denom = ref_rms if ref_rms > 1.0e-9 else 1.0
    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    cosine = float(
        torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), cand_flat.unsqueeze(0), dim=1).item()
    )
    return {
        "shape_match": True,
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "mae": float(torch.mean(torch.abs(diff)).item()),
        "max_abs": float(torch.max(torch.abs(diff)).item()),
        "rmse": rmse,
        "reference_rms": ref_rms,
        "relative_rmse": rmse / denom,
        "cosine": cosine,
    }


def _first_divergent_layer(layer_metrics: list[dict[str, Any]], threshold: float) -> int | None:
    for index, metrics in enumerate(layer_metrics):
        if not metrics.get("shape_match", False):
            return index
        if float(metrics.get("relative_rmse", 0.0)) > threshold:
            return index
    return None


def _capture_encoder_call(
    encoder: Any,
    *,
    audio_signal: torch.Tensor,
    length: torch.Tensor,
    cache_last_channel: torch.Tensor | None,
    cache_last_time: torch.Tensor | None,
    cache_last_channel_len: torch.Tensor | None,
    keep_all_outputs: bool,
) -> dict[str, Any]:
    layer_outputs: list[torch.Tensor | None] = [None] * len(encoder.layers)
    hooks = []

    def _make_hook(index: int):
        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            layer_outputs[index] = tensor.detach().cpu().to(dtype=torch.float32).clone()

        return _hook

    for index, layer in enumerate(encoder.layers):
        hooks.append(layer.register_forward_hook(_make_hook(index)))

    try:
        with torch.inference_mode():
            rets = encoder.forward_internal(
                audio_signal=audio_signal,
                length=length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                bypass_pre_encode=False,
            )
            outputs = encoder.streaming_post_process(rets, keep_all_outputs=keep_all_outputs)
    finally:
        for hook in hooks:
            hook.remove()

    encoded, encoded_len, next_cache, next_time_cache, next_cache_len = outputs
    final_btd = encoded.detach().cpu().to(dtype=torch.float32).transpose(1, 2).contiguous()
    return {
        "final": final_btd,
        "length": int(encoded_len.reshape(-1)[0].item()),
        "layers": [tensor if tensor is not None else torch.empty(0) for tensor in layer_outputs],
        "next_cache": next_cache.detach().clone() if next_cache is not None else None,
        "next_time_cache": next_time_cache.detach().clone() if next_time_cache is not None else None,
        "next_cache_len": next_cache_len.detach().clone() if next_cache_len is not None else None,
    }


def main() -> None:
    args = _parse_args()

    try:
        import nemo.collections.asr as nemo_asr  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "nemo_toolkit[asr] is not installed. Run: pip install -r requirements-conversion.txt"
        ) from exc

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
    model.eval()
    model.to("cpu")

    left_chunks = args.left_chunks
    if left_chunks is None:
        left_chunks = max(1, int(args.left_context_steps) // max(1, int(args.chunk_steps)))

    encoder_template = model.encoder
    encoder_template.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[int(args.left_context_steps), int(args.right_context_steps)],
        update_config=True,
        device=torch.device("cpu"),
    )
    encoder_template.setup_streaming_params(
        chunk_size=int(args.chunk_steps),
        shift_size=int(args.shift_steps),
        left_chunks=int(left_chunks),
        att_context_size=[int(args.left_context_steps), int(args.right_context_steps)],
    )
    native_encoder = copy.deepcopy(encoder_template)
    patched_encoder = copy.deepcopy(encoder_template)
    native_encoder.eval()
    patched_encoder.eval()

    audio, sample_rate = _load_audio(args.audio, args.max_audio_seconds)
    features = _extract_features(model, audio, sample_rate)
    streaming_cfg = encoder_template.streaming_cfg
    input_feature_frames = _resolve_streaming_value(streaming_cfg.chunk_size, 1) + _resolve_streaming_value(
        streaming_cfg.pre_encode_cache_size, 1
    )
    shift_feature_frames = _resolve_streaming_value(streaming_cfg.shift_size, 1)
    pre_encode_cache_frames = _resolve_streaming_value(streaming_cfg.pre_encode_cache_size, 1)
    reference_left_padding_frames = input_feature_frames - shift_feature_frames
    valid_out_len = int(streaming_cfg.valid_out_len)
    drop_extra_pre_encoded = int(streaming_cfg.drop_extra_pre_encoded)

    if int(features.shape[-1]) < shift_feature_frames:
        raise SystemExit(
            f"feature sequence too short: {int(features.shape[-1])} frames < required first step {shift_feature_frames}"
        )

    native_steps: list[dict[str, Any]] = []
    for step_index in range(int(args.max_steps)):
        chunk, target_end = _make_stream_chunk(
            features,
            step_index=step_index,
            input_feature_frames=input_feature_frames,
            shift_feature_frames=shift_feature_frames,
        )
        if chunk is None:
            break
        prefix = _make_reference_prefix(
            features,
            target_end=target_end,
            left_padding_frames=reference_left_padding_frames,
        )
        native = _capture_encoder_call(
            native_encoder,
            audio_signal=prefix,
            length=torch.tensor([prefix.shape[-1]], dtype=torch.int64),
            cache_last_channel=None,
            cache_last_time=None,
            cache_last_channel_len=None,
            keep_all_outputs=True,
        )
        native_steps.append(
            {
                "step_index": step_index,
                "target_end_frame": target_end,
                "full_output_length": native["length"],
                "step_start": drop_extra_pre_encoded + step_index * valid_out_len,
                "step_end": drop_extra_pre_encoded + (step_index + 1) * valid_out_len,
                "native": native,
            }
        )

    if not native_steps:
        raise SystemExit("no streaming steps could be formed from the extracted features")

    _make_local_attention_export_friendly(patched_encoder)

    channel_cache, time_cache, cache_len = patched_encoder.get_initial_cache_state(
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    summary_steps: list[dict[str, Any]] = []
    for step in native_steps:
        step_index = int(step["step_index"])
        target_end = int(step["target_end_frame"])
        prefix = _make_reference_prefix(
            features,
            target_end=target_end,
            left_padding_frames=reference_left_padding_frames,
        )
        patched_full = _capture_encoder_call(
            patched_encoder,
            audio_signal=prefix,
            length=torch.tensor([prefix.shape[-1]], dtype=torch.int64),
            cache_last_channel=None,
            cache_last_time=None,
            cache_last_channel_len=None,
            keep_all_outputs=True,
        )

        chunk, _ = _make_stream_chunk(
            features,
            step_index=step_index,
            input_feature_frames=input_feature_frames,
            shift_feature_frames=shift_feature_frames,
        )
        assert chunk is not None
        cached = _capture_encoder_call(
            patched_encoder,
            audio_signal=chunk,
            length=torch.tensor([input_feature_frames], dtype=torch.int64),
            cache_last_channel=channel_cache,
            cache_last_time=time_cache,
            cache_last_channel_len=cache_len,
            keep_all_outputs=False,
        )
        channel_cache = cached["next_cache"]
        time_cache = cached["next_time_cache"]
        cache_len = cached["next_cache_len"]

        step_start = int(step["step_start"])
        step_end = int(step["step_end"])
        native_final_slice = step["native"]["final"][:, step_start:step_end, :].contiguous()
        patched_full_final_slice = patched_full["final"][:, step_start:step_end, :].contiguous()
        cached_final = cached["final"].contiguous()

        native_vs_patched_layers = []
        patched_vs_cached_layers = []
        for layer_index in range(len(patched_encoder.layers)):
            cached_layer = cached["layers"][layer_index].contiguous()
            cached_layer_len = int(cached_layer.shape[1])
            layer_end = step_start + cached_layer_len
            native_layer = step["native"]["layers"][layer_index][:, step_start:layer_end, :].contiguous()
            patched_full_layer = patched_full["layers"][layer_index][:, step_start:layer_end, :].contiguous()
            native_vs_patched_layers.append(_metric_bundle(native_layer, patched_full_layer))
            patched_vs_cached_layers.append(_metric_bundle(patched_full_layer, cached_layer))

        native_vs_patched_final = _metric_bundle(native_final_slice, patched_full_final_slice)
        patched_vs_cached_final = _metric_bundle(patched_full_final_slice, cached_final)
        first_divergent = _first_divergent_layer(patched_vs_cached_layers, float(args.rel_rmse_threshold))

        step_summary = {
            "step_index": step_index,
            "target_end_frame": target_end,
            "reference_prefix_frames": int(prefix.shape[-1]),
            "native_full_output_length": int(step["native"]["length"]),
            "patched_full_output_length": int(patched_full["length"]),
            "cached_output_length": int(cached["length"]),
            "cache_length_after_step": int(cache_len.reshape(-1)[0].item()) if cache_len is not None else None,
            "slice": {
                "start": step_start,
                "end": step_end,
                "valid_out_len": valid_out_len,
            },
            "native_vs_patched_full_final": native_vs_patched_final,
            "patched_full_vs_cached_final": patched_vs_cached_final,
            "native_vs_patched_full_layers": native_vs_patched_layers,
            "patched_full_vs_cached_layers": patched_vs_cached_layers,
            "first_divergent_layer": first_divergent,
        }
        summary_steps.append(step_summary)

        patched_rel_rmse = native_vs_patched_final.get("relative_rmse")
        cached_rel_rmse = patched_vs_cached_final.get("relative_rmse")
        print(
            f"step={step_index} target_end={target_end} "
            f"native_vs_patched_final_rel_rmse={patched_rel_rmse:.6f} "
            f"patched_vs_cached_final_rel_rmse={cached_rel_rmse:.6f} "
            f"first_divergent_layer={first_divergent}"
        )

    summary = {
        "audio": str(args.audio),
        "max_audio_seconds": args.max_audio_seconds,
        "model": args.model,
            "streaming": {
            "left_context_steps": int(args.left_context_steps),
            "right_context_steps": int(args.right_context_steps),
            "chunk_steps": int(args.chunk_steps),
            "shift_steps": int(args.shift_steps),
            "left_chunks": int(left_chunks),
            "input_feature_frames": int(input_feature_frames),
            "shift_feature_frames": int(shift_feature_frames),
            "pre_encode_cache_frames": int(pre_encode_cache_frames),
            "reference_left_padding_frames": int(reference_left_padding_frames),
            "drop_extra_pre_encoded": int(drop_extra_pre_encoded),
            "valid_output_steps": int(valid_out_len),
            "features_total_frames": int(features.shape[-1]),
        },
        "thresholds": {
            "relative_rmse": float(args.rel_rmse_threshold),
        },
        "steps": summary_steps,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
