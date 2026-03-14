#!/usr/bin/env python3
"""Export a no-cache local-attention encoder TorchScript wrapper for CoreML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from make_streaming_local_attention_encoder_torchscript import (
    _make_local_attention_export_friendly,
)


class LocalAttentionEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.feat_in = int(encoder._feat_in)
        self.d_model = int(encoder.d_model)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        encoded, encoded_lengths = self.encoder(audio_signal=audio_signal, length=length)
        return encoded, encoded_lengths.to(dtype=torch.int64)


def _build_manifest(
    *,
    onnx_path_hint: str,
    feat_in: int,
    d_model: int,
    input_feature_frames: int,
    output_steps: int,
    attention_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "onnx_path": onnx_path_hint,
        "inputs": [
            {
                "name": "audio_signal",
                "dtype": "FLOAT",
                "shape": [1, feat_in, input_feature_frames],
            },
            {
                "name": "length",
                "dtype": "INT64",
                "shape": [1],
            },
        ],
        "outputs": [
            {
                "name": "outputs",
                "dtype": "FLOAT",
                "shape": [1, d_model, output_steps],
            },
            {
                "name": "encoded_lengths",
                "dtype": "INT64",
                "shape": [1],
            },
        ],
        "node_count": 0,
        "op_histogram": {},
        "coreml_suggestions": {
            "audio_input_name": "audio_signal",
            "length_input_name": "length",
            "logits_output_name": "outputs",
        },
        "attention_info": attention_info,
        "wrapped_local_attention_encoder": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a no-cache local-attention encoder TorchScript wrapper.")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="NeMo model name. Example: nvidia/parakeet-tdt-0.6b-v2",
    )
    parser.add_argument("--output-ts", type=Path, required=True, help="Output TorchScript path.")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Output manifest JSON path.")
    parser.add_argument(
        "--left-context-steps",
        type=int,
        required=True,
        help="Encoder-step local attention left context size.",
    )
    parser.add_argument(
        "--right-context-steps",
        type=int,
        default=0,
        help="Encoder-step local attention right context size.",
    )
    parser.add_argument(
        "--input-feature-frames",
        type=int,
        default=300,
        help="Encoder feature-frame window size baked into the manifest/example inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import nemo.collections.asr as nemo_asr  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "nemo_toolkit[asr] is not installed. Run: pip install -r requirements-conversion.txt"
        ) from exc

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
    model.eval()
    model.to("cpu")
    encoder = model.encoder

    encoder.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[int(args.left_context_steps), int(args.right_context_steps)],
        update_config=True,
        device=torch.device("cpu"),
    )
    _make_local_attention_export_friendly(encoder)

    wrapper = LocalAttentionEncoderWrapper(encoder=encoder)
    wrapper.eval()

    input_feature_frames = int(args.input_feature_frames)
    example_audio = torch.randn(1, wrapper.feat_in, input_feature_frames, dtype=torch.float32)
    example_length = torch.tensor([input_feature_frames], dtype=torch.int64)

    with torch.no_grad():
        example_outputs, example_lengths = wrapper(example_audio, example_length)

    try:
        wrapped_ts = torch.jit.script(wrapper)
    except Exception:
        wrapped_ts = torch.jit.trace(
            wrapper,
            (example_audio, example_length),
            check_trace=False,
        )

    args.output_ts.parent.mkdir(parents=True, exist_ok=True)
    wrapped_ts.save(str(args.output_ts))

    attention_info = {
        "kind": "local_attention_prefix",
        "attention_model": "rel_pos_local_attn",
        "left_context_steps": int(args.left_context_steps),
        "right_context_steps": int(args.right_context_steps),
        "input_feature_frames": input_feature_frames,
        "output_steps": int(example_outputs.shape[-1]),
        "example_encoded_length": int(example_lengths.reshape(-1)[0].item()),
    }

    manifest = _build_manifest(
        onnx_path_hint=str(args.output_ts),
        feat_in=wrapper.feat_in,
        d_model=wrapper.d_model,
        input_feature_frames=input_feature_frames,
        output_steps=int(example_outputs.shape[-1]),
        attention_info=attention_info,
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Local-attention encoder TorchScript: {args.output_ts}")
    print(f"Local-attention encoder manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
