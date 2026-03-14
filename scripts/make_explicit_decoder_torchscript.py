#!/usr/bin/env python3
"""Build an explicit-cell decoder TorchScript from the exported scripted decoder.

The source decoder uses `torch.lstm`, which Core ML lowers back to LSTM ops that
remain CPU-preferred in the current runtime. This script rebuilds the predictor
with explicit gate math so the TorchScript -> Core ML conversion path sees only
primitive ops.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class ExplicitDecoderJoint(torch.nn.Module):
    def __init__(
        self,
        *,
        lstm_w_ih_l0: torch.Tensor,
        lstm_w_hh_l0: torch.Tensor,
        lstm_b_ih_l0: torch.Tensor,
        lstm_b_hh_l0: torch.Tensor,
        lstm_w_ih_l1: torch.Tensor,
        lstm_w_hh_l1: torch.Tensor,
        lstm_b_ih_l1: torch.Tensor,
        lstm_b_hh_l1: torch.Tensor,
        embed_weight: torch.Tensor,
        joint_enc_weight: torch.Tensor,
        joint_enc_bias: torch.Tensor,
        joint_pred_weight: torch.Tensor,
        joint_pred_bias: torch.Tensor,
        joint_out_weight: torch.Tensor,
        joint_out_bias: torch.Tensor,
    ):
        super().__init__()
        for name, value in (
            ("lstm_w_ih_l0", lstm_w_ih_l0),
            ("lstm_w_hh_l0", lstm_w_hh_l0),
            ("lstm_b_ih_l0", lstm_b_ih_l0),
            ("lstm_b_hh_l0", lstm_b_hh_l0),
            ("lstm_w_ih_l1", lstm_w_ih_l1),
            ("lstm_w_hh_l1", lstm_w_hh_l1),
            ("lstm_b_ih_l1", lstm_b_ih_l1),
            ("lstm_b_hh_l1", lstm_b_hh_l1),
            ("embed_weight", embed_weight),
            ("joint_enc_weight", joint_enc_weight),
            ("joint_enc_bias", joint_enc_bias),
            ("joint_pred_weight", joint_pred_weight),
            ("joint_pred_bias", joint_pred_bias),
            ("joint_out_weight", joint_out_weight),
            ("joint_out_bias", joint_out_bias),
        ):
            self.register_buffer(name, value.clone())

    def _lstm_cell(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor,
        b_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = F.linear(x, w_ih, b_ih) + F.linear(h, w_hh, b_hh)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        target_length: torch.Tensor,
        input_states_1: torch.Tensor,
        input_states_2: torch.Tensor,
    ):
        tokens = targets.to(dtype=torch.int64)
        embedded = F.embedding(tokens, self.embed_weight)
        seq_first = embedded.transpose(0, 1)

        h0 = input_states_1[0]
        h1 = input_states_1[1]
        c0 = input_states_2[0]
        c1 = input_states_2[1]

        # The runtime feeds exactly one target token per decoder step.
        x_t = seq_first[0]
        h0, c0 = self._lstm_cell(
            x_t,
            h0,
            c0,
            self.lstm_w_ih_l0,
            self.lstm_w_hh_l0,
            self.lstm_b_ih_l0,
            self.lstm_b_hh_l0,
        )
        h1, c1 = self._lstm_cell(
            h0,
            h1,
            c1,
            self.lstm_w_ih_l1,
            self.lstm_w_hh_l1,
            self.lstm_b_ih_l1,
            self.lstm_b_hh_l1,
        )

        input1 = torch.unsqueeze(h1, 0)
        g = input1.transpose(0, 1)
        decoder_outputs = g.transpose(1, 2)
        input2 = encoder_outputs.transpose(1, 2)
        input3 = decoder_outputs.transpose(1, 2)
        f = F.linear(input2, self.joint_enc_weight, self.joint_enc_bias)
        g0 = F.linear(input3, self.joint_pred_weight, self.joint_pred_bias)
        f0 = torch.unsqueeze(f, 2)
        g1 = torch.unsqueeze(g0, 1)
        input4 = torch.add(f0, g1)
        input5 = F.relu(input4)
        outputs = F.linear(input5, self.joint_out_weight, self.joint_out_bias)
        output_states_1 = torch.stack([h0, h1], dim=0)
        output_states_2 = torch.stack([c0, c1], dim=0)
        return outputs, target_length, output_states_1, output_states_2


def _extract_constants(scripted: torch.jit.ScriptModule) -> dict[str, torch.Tensor]:
    _, consts = scripted.code_with_constants
    if not hasattr(consts, "const_mapping"):
        raise RuntimeError("TorchScript constants are unavailable.")
    mapping = consts.const_mapping
    required = {f"c{i}" for i in range(15)}
    missing = sorted(required.difference(mapping))
    if missing:
        raise RuntimeError(f"Missing constants from decoder TorchScript: {missing}")
    out: dict[str, torch.Tensor] = {}
    for name in sorted(required):
        tensor = mapping[name]
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"Expected tensor constant for {name}, got {type(tensor)}")
        out[name] = tensor.detach().clone()
    return out


def _make_random_inputs(manifest: dict[str, Any]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for item in manifest["inputs"]:
        name = item["name"]
        shape = []
        for idx, dim in enumerate(item["shape"]):
            if isinstance(dim, int):
                shape.append(dim)
            elif name == "encoder_outputs" and idx == 2:
                shape.append(300)
            else:
                shape.append(1)
        dtype = (item.get("dtype") or "FLOAT").upper()
        if dtype == "INT32":
            tensor = torch.zeros(shape, dtype=torch.int32)
            if name == "target_length":
                tensor.fill_(1)
        else:
            tensor = torch.randn(shape, dtype=torch.float32) * 0.25
        tensors.append(tensor)
    return tensors


def _make_trace_inputs(manifest: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    tensors = _make_random_inputs(manifest)
    for idx, item in enumerate(manifest["inputs"]):
        if item["name"] == "targets":
            tensors[idx].zero_()
        elif item["name"] == "target_length":
            tensors[idx].fill_(1)
    return tuple(tensors)


def _verify_models(
    original: torch.jit.ScriptModule,
    rewritten: torch.jit.ScriptModule,
    *,
    manifest: dict[str, Any],
    runs: int,
    atol: float,
    rtol: float,
) -> None:
    original.eval()
    rewritten.eval()
    with torch.no_grad():
        worst_abs = 0.0
        worst_rel = 0.0
        for _ in range(runs):
            inputs = _make_random_inputs(manifest)
            ref = original(*inputs)
            test = rewritten(*inputs)
            if not isinstance(ref, (tuple, list)) or not isinstance(test, (tuple, list)):
                raise RuntimeError("Decoder outputs are not tuples.")
            for idx, (a, b) in enumerate(zip(ref, test)):
                abs_err = float((a - b).abs().max().item())
                rel_err = float(((a - b).abs() / torch.clamp(a.abs(), min=1e-8)).max().item())
                worst_abs = max(worst_abs, abs_err)
                worst_rel = max(worst_rel, rel_err)
                if not torch.allclose(a, b, atol=atol, rtol=rtol):
                    raise AssertionError(
                        f"Verification failed for output {idx}: max_abs={abs_err:.6e} max_rel={rel_err:.6e}"
                    )
    print(f"Verification passed: runs={runs} max_abs={worst_abs:.6e} max_rel={worst_rel:.6e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create explicit-cell decoder TorchScript.")
    parser.add_argument("--decoder-ts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-ts", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, help="Optional copied manifest path.")
    parser.add_argument("--verify-runs", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original = torch.jit.load(str(args.decoder_ts), map_location="cpu")
    original.eval()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    consts = _extract_constants(original)

    module = ExplicitDecoderJoint(
        lstm_w_ih_l0=consts["c0"],
        lstm_w_hh_l0=consts["c1"],
        lstm_b_ih_l0=consts["c2"],
        lstm_b_hh_l0=consts["c3"],
        lstm_w_ih_l1=consts["c4"],
        lstm_w_hh_l1=consts["c5"],
        lstm_b_ih_l1=consts["c6"],
        lstm_b_hh_l1=consts["c7"],
        embed_weight=consts["c8"],
        joint_enc_weight=consts["c9"],
        joint_enc_bias=consts["c10"],
        joint_pred_weight=consts["c11"],
        joint_pred_bias=consts["c12"],
        joint_out_weight=consts["c13"],
        joint_out_bias=consts["c14"],
    )
    module.eval()
    example_inputs = _make_trace_inputs(manifest)
    scripted = torch.jit.trace(module, example_inputs, check_trace=False)
    _verify_models(
        original,
        scripted,
        manifest=manifest,
        runs=max(1, args.verify_runs),
        atol=args.atol,
        rtol=args.rtol,
    )

    args.output_ts.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output_ts))
    print(f"Explicit decoder TorchScript: {args.output_ts}")

    if args.output_manifest:
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Explicit decoder manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
