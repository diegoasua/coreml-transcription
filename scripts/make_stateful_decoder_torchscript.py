#!/usr/bin/env python3
"""Wrap decoder TorchScript with registered recurrent-state buffers.

This enables coremltools `StateType` conversion by exposing state tensors as
named buffers in the source TorchScript model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _resolve_shape(raw: list[Any], *, decoder_frames: int) -> list[int]:
    out: list[int] = []
    for idx, dim in enumerate(raw):
        if isinstance(dim, int):
            out.append(dim)
            continue
        # Dynamic dims fallback:
        # - batch-like dims -> 1
        # - time dim for encoder_outputs -> decoder_frames
        if idx == len(raw) - 1 and len(raw) >= 3:
            out.append(decoder_frames)
        else:
            out.append(1)
    return out


class StatefulDecoderWrapper(torch.nn.Module):
    def __init__(
        self,
        base: torch.jit.ScriptModule,
        state1: torch.Tensor,
        state2: torch.Tensor,
    ):
        super().__init__()
        self.base = base
        self.register_buffer("state_1", state1.clone())
        self.register_buffer("state_2", state2.clone())

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        target_length: torch.Tensor,
        state_update_gate: torch.Tensor,
        step_index: torch.Tensor,
    ):
        outputs, prednet_lengths, out_state1, out_state2 = self.base(
            encoder_outputs,
            targets,
            target_length,
            self.state_1,
            self.state_2,
        )

        # Tensorized gate selection (TorchScript/CoreML-friendly):
        # gate<0.5 -> no commit
        # 0.5<=gate<1.5 -> always commit
        # gate>=1.5 -> auto-commit when current-step token is non-blank
        gate_scalar = torch.select(state_update_gate.reshape(-1), 0, 0).to(dtype=outputs.dtype)
        always_mask = torch.logical_and(gate_scalar >= 0.5, gate_scalar < 1.5).to(dtype=out_state1.dtype)
        auto_mode_mask = (gate_scalar >= 1.5).to(dtype=out_state1.dtype)

        # Decoder output is expected as [B, T, U, V] with U=1 for current export.
        step_raw = torch.select(step_index.reshape(-1), 0, 0).to(dtype=torch.long)
        max_t = outputs.size(1) - 1
        step_clamped = torch.clamp(step_raw, min=0, max=max_t).reshape(1)
        step_slice = torch.index_select(outputs, dim=1, index=step_clamped)
        step_slice = torch.select(step_slice, dim=0, index=0)
        step_slice = torch.select(step_slice, dim=0, index=0)
        step_slice = torch.select(step_slice, dim=0, index=0)
        duration_bins = 5
        vocab_plus_blank = step_slice.size(0) - duration_bins
        blank_id = vocab_plus_blank - 1
        idx = torch.arange(step_slice.size(0), dtype=torch.long, device=step_slice.device)
        token_mask = idx < vocab_plus_blank
        token_logits = torch.where(token_mask, step_slice, torch.full_like(step_slice, -1.0e4))
        token_id = torch.argmax(token_logits, dim=0).to(dtype=torch.long)
        auto_commit = (token_id != blank_id).to(dtype=out_state1.dtype)

        commit_scalar = torch.clamp(always_mask + auto_mode_mask * auto_commit, min=0.0, max=1.0)
        gate = commit_scalar.reshape(1, 1, 1)
        inv_gate = 1.0 - gate
        self.state_1[:, :, :] = out_state1 * gate + self.state_1 * inv_gate
        self.state_2[:, :, :] = out_state2 * gate + self.state_2 * inv_gate

        # Ensure output tensor is not aliased to input tensor.
        prednet_lengths_out = prednet_lengths + torch.zeros_like(prednet_lengths)
        return outputs, prednet_lengths_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stateful decoder TorchScript wrapper.")
    parser.add_argument("--decoder-ts", type=Path, required=True, help="Input decoder TorchScript path.")
    parser.add_argument("--manifest", type=Path, required=True, help="Input decoder manifest JSON.")
    parser.add_argument("--output-ts", type=Path, required=True, help="Output wrapped TorchScript path.")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Output manifest path for wrapped model.")
    parser.add_argument("--decoder-frames", type=int, default=300, help="Fallback decoder time frames.")
    parser.add_argument(
        "--duration-bins",
        type=int,
        default=5,
        help="Number of TDT duration bins (used to infer blank id from logits dim).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs", [])

    def _find(name: str) -> dict[str, Any]:
        for item in inputs:
            if item.get("name") == name:
                return item
        raise ValueError(f"Input '{name}' not found in manifest {args.manifest}")

    enc_shape = _resolve_shape(_find("encoder_outputs").get("shape", [1, 1024, args.decoder_frames]), decoder_frames=args.decoder_frames)
    tgt_shape = _resolve_shape(_find("targets").get("shape", [1, 1]), decoder_frames=args.decoder_frames)
    tgt_len_shape = _resolve_shape(_find("target_length").get("shape", [1]), decoder_frames=args.decoder_frames)
    st1_shape = _resolve_shape(_find("input_states_1").get("shape", [2, 1, 640]), decoder_frames=args.decoder_frames)
    st2_shape = _resolve_shape(_find("input_states_2").get("shape", [2, 1, 640]), decoder_frames=args.decoder_frames)

    enc = torch.randn(*enc_shape, dtype=torch.float32)
    targets = torch.zeros(*tgt_shape, dtype=torch.int32)
    target_length = torch.ones(*tgt_len_shape, dtype=torch.int32)
    # Keep state dtype aligned with decoder state tensors (float32).
    state1 = torch.zeros(*st1_shape, dtype=torch.float32)
    state2 = torch.zeros(*st2_shape, dtype=torch.float32)
    step_index = torch.zeros(1, dtype=torch.int32)

    base = torch.jit.load(str(args.decoder_ts), map_location="cpu")
    base.eval()

    with torch.no_grad():
        sample_outputs = base(enc, targets, target_length, state1, state2)
    if not isinstance(sample_outputs, (tuple, list)) or not sample_outputs:
        raise RuntimeError("Decoder TorchScript did not return expected tuple outputs.")
    logits = sample_outputs[0]
    if not isinstance(logits, torch.Tensor) or logits.ndim < 2:
        raise RuntimeError(f"Unexpected decoder logits tensor: type={type(logits)} rank={getattr(logits, 'ndim', '?')}")
    logits_dim = int(logits.shape[-1])
    duration_bins = max(1, int(args.duration_bins))
    token_vocab_size = logits_dim - duration_bins
    blank_id = token_vocab_size - 1
    if token_vocab_size <= 0 or blank_id < 0:
        raise RuntimeError(
            f"Invalid logits/duration sizing: logits_dim={logits_dim}, duration_bins={duration_bins}"
        )

    wrapper = StatefulDecoderWrapper(base, state1=state1, state2=state2)
    wrapper.eval()
    try:
        wrapped_ts = torch.jit.script(wrapper)
    except Exception:
        gate = torch.zeros(1, dtype=torch.float32)
        wrapped_ts = torch.jit.trace(wrapper, (enc, targets, target_length, gate, step_index), check_trace=False)

    args.output_ts.parent.mkdir(parents=True, exist_ok=True)
    wrapped_ts.save(str(args.output_ts))

    filtered_inputs = [
        item for item in inputs if item.get("name") not in {"input_states_1", "input_states_2"}
    ]
    filtered_outputs = [
        item for item in manifest.get("outputs", []) if item.get("name") not in {"output_states_1", "output_states_2"}
    ]
    gate_input = {
        "name": "state_update_gate",
        "dtype": "FLOAT",
        "shape": [1],
    }
    step_index_input = {
        "name": "step_index",
        "dtype": "INT32",
        "shape": [1],
    }
    wrapped_manifest = dict(manifest)
    wrapped_manifest["inputs"] = filtered_inputs + [gate_input, step_index_input]
    wrapped_manifest["outputs"] = filtered_outputs
    wrapped_manifest["wrapped_stateful_decoder"] = True
    wrapped_manifest["stateful_buffers"] = ["state_1", "state_2"]
    wrapped_manifest["stateful_commit_modes"] = {
        "0": "no_commit",
        "1": "always_commit",
        "2": "auto_commit_on_nonblank_at_step_index",
    }
    wrapped_manifest["tdt_duration_bins"] = duration_bins
    wrapped_manifest["tdt_blank_id"] = blank_id
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(wrapped_manifest, indent=2), encoding="utf-8")

    print(f"Stateful decoder wrapper TorchScript: {args.output_ts}")
    print(f"Stateful decoder wrapper manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
