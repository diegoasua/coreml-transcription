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
    def __init__(self, base: torch.jit.ScriptModule, state1: torch.Tensor, state2: torch.Tensor):
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
    ):
        outputs, prednet_lengths, out_state1, out_state2 = self.base(
            encoder_outputs,
            targets,
            target_length,
            self.state_1,
            self.state_2,
        )

        # Trace-safe gated state update:
        # gate=0 keeps previous state, gate=1 commits out_state.
        gate = state_update_gate.reshape(1, 1, 1)
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

    base = torch.jit.load(str(args.decoder_ts), map_location="cpu")
    base.eval()

    wrapper = StatefulDecoderWrapper(base, state1=state1, state2=state2)
    wrapper.eval()
    try:
        wrapped_ts = torch.jit.script(wrapper)
    except Exception:
        gate = torch.zeros(1, dtype=torch.float32)
        wrapped_ts = torch.jit.trace(wrapper, (enc, targets, target_length, gate), check_trace=False)

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
    wrapped_manifest = dict(manifest)
    wrapped_manifest["inputs"] = filtered_inputs + [gate_input]
    wrapped_manifest["outputs"] = filtered_outputs
    wrapped_manifest["wrapped_stateful_decoder"] = True
    wrapped_manifest["stateful_buffers"] = ["state_1", "state_2"]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(wrapped_manifest, indent=2), encoding="utf-8")

    print(f"Stateful decoder wrapper TorchScript: {args.output_ts}")
    print(f"Stateful decoder wrapper manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
