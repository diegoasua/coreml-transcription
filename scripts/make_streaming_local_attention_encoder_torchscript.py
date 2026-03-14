#!/usr/bin/env python3
"""Export a cache-aware local-attention encoder TorchScript wrapper for CoreML.

NeMo's built-in `rel_pos_local_attn` cache path reuses the full-attention cache
concatenation logic and breaks once cached keys become longer than the current
query chunk. This exporter patches the local-attention kernel to support cached
streaming queries while preserving NeMo's native encoder streaming contract:
channel cache, convolution time cache, and cache-length tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _resolve_streaming_value(raw: int | list[int], index: int) -> int:
    if isinstance(raw, list):
        return int(raw[index])
    return int(raw)


class ExportFriendlyRelPositionMultiHeadAttentionLongformer(torch.nn.Module):
    def __init__(self, base_module: torch.nn.Module):
        super().__init__()
        self.base = base_module

    def __getattr__(self, name: str):
        if name == "base":
            return super().__getattr__(name)
        return getattr(self.base, name)

    def _chunk_overlap(self, x: torch.Tensor, w: int) -> torch.Tensor:
        chunks = []
        limit = x.size(1) // w - 1
        for chunk_index in range(limit):
            start = chunk_index * w
            chunks.append(x[:, start : start + 2 * w, :].unsqueeze(1))
        return torch.cat(chunks, dim=1)

    def sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float) -> torch.Tensor:
        bsz, num_heads, seqlen, head_dim = q.size()
        assert seqlen % (w * 2) == 0
        assert q.size() == k.size()

        chunks_count = seqlen // w - 1
        q = q.reshape(bsz * num_heads, seqlen, head_dim)
        k = k.reshape(bsz * num_heads, seqlen, head_dim)

        chunk_q = self._chunk_overlap(q, w)
        chunk_k = self._chunk_overlap(k, w)
        chunk_attn = torch.einsum("bcxd,bcyd->bcxy", (chunk_q, chunk_k))
        diagonal_chunk_attn = self.base._skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

        diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1) : -1, w + 1 :]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]
        diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1)
        self.base.mask_invalid_locations(diagonal_attn, w)
        return diagonal_attn

    def sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int) -> torch.Tensor:
        bsz, num_heads, seqlen, head_dim = v.size()
        chunks_count = seqlen // w - 1
        chunk_prob = prob.reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
        v = v.reshape(bsz * num_heads, seqlen, head_dim)
        padded_v = F.pad(v, (0, 0, w, w), value=-1)

        chunk_vs = []
        for chunk_index in range(chunks_count + 1):
            start = chunk_index * w
            chunk_vs.append(padded_v[:, start : start + 3 * w, :].unsqueeze(1))
        chunk_v = torch.cat(chunk_vs, dim=1)

        skewed_prob = self.base._skew2(chunk_prob, padding_value=0)
        context = torch.einsum("bcwd,bcdh->bcwh", (skewed_prob, chunk_v))
        return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)

    def forward(self, query, key, value, pad_mask, pos_emb, cache=None):
        original_query_len = query.size(1)
        input_cache = cache
        key, value, query, cache = self.base.update_cache(key=key, value=value, query=query, cache=cache)
        full_cache_prefix_len = key.size(1) - query.size(1)
        if input_cache is not None and full_cache_prefix_len > 0:
            cache_nonzero = torch.sum(torch.abs(input_cache[0]), dim=-1) != 0
            valid_cache_len = int(torch.sum(cache_nonzero.to(torch.int64)).item())
            valid_cache_len = max(0, min(valid_cache_len, full_cache_prefix_len))
            if valid_cache_len < full_cache_prefix_len:
                key = key[:, full_cache_prefix_len - valid_cache_len :, :]
                value = value[:, full_cache_prefix_len - valid_cache_len :, :]
        cache_prefix_len = key.size(1) - query.size(1)

        if cache_prefix_len > 0:
            query_prefix = query.new_zeros((query.size(0), cache_prefix_len, query.size(2)))
            query = torch.cat((query_prefix, query), dim=1)
            if pad_mask is None:
                pad_mask = torch.zeros((query.size(0), original_query_len), device=query.device, dtype=torch.bool)
            mask_prefix = pad_mask.new_zeros((pad_mask.size(0), cache_prefix_len))
            pad_mask = torch.cat((mask_prefix, pad_mask), dim=1)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        q, k, v = self.base.forward_qkv(query, key, value)
        n_batch, _, T, _ = q.size()

        w = max(self.base.att_context_size[0], self.base.att_context_size[1])
        if w <= 0:
            raise ValueError("When using local attention, context size must be set > 0")
        pad_len = (2 * w - T % (2 * w)) % (2 * w)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        if pad_mask is None:
            mask = torch.zeros((n_batch, T + pad_len), device=q.device, dtype=torch.bool)
        else:
            mask = F.pad(pad_mask, (0, pad_len), value=True)

        q_with_bias_u = q + self.base.pos_bias_u.unsqueeze(1)
        q_with_bias_v = q + self.base.pos_bias_v.unsqueeze(1)
        diagonal_matrix_ac = self.sliding_chunks_matmul_qk(q_with_bias_u, k, w, padding_value=0.0)

        n_batch_pos = pos_emb.size(0)
        p = self.base.linear_pos(pos_emb).view(n_batch_pos, -1, self.base.h, self.base.d_k).transpose(1, 2)
        diagonal_matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        start_pos = w - self.base.att_context_size[0]
        end_pos = w + self.base.att_context_size[1]
        diagonal_matrix_ac[:, :, :, : self.base.att_context_size[0]] += diagonal_matrix_bd[
            :, :, :, : self.base.att_context_size[0]
        ]
        diagonal_matrix_ac[:, :, :, -(self.base.att_context_size[1] + 1) :] += diagonal_matrix_bd[
            :, :, :, self.base.att_context_size[0] :
        ]
        scores = diagonal_matrix_ac / self.base.s_d_k
        scores[:, :, :, :start_pos] = -1.0e4
        scores[:, :, :, end_pos + 1 :] = -1.0e4

        mask = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
        float_mask = mask.type_as(scores).masked_fill(mask, -1.0e4)
        ones = torch.ones_like(float_mask)
        d_mask = self.sliding_chunks_matmul_qk(ones, float_mask, w, padding_value=0.0)
        scores += d_mask

        if self.base.global_tokens > 0:
            raise NotImplementedError("Export-friendly wrapper does not support global_tokens > 0.")

        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        p_attn = self.base.dropout(attn)
        out = self.sliding_chunks_matmul_pv(p_attn, v, w)
        out = out.reshape(n_batch, -1, self.base.h * self.base.d_k)[:, :T]
        ret = self.base.linear_out(out)
        ret = ret[:, cache_prefix_len : cache_prefix_len + original_query_len, :]

        if cache is None:
            return ret
        return ret, cache


def _make_local_attention_export_friendly(encoder: torch.nn.Module) -> None:
    for layer in encoder.layers:
        if getattr(layer, "self_attention_model", "") != "rel_pos_local_attn":
            continue
        layer.self_attn = ExportFriendlyRelPositionMultiHeadAttentionLongformer(layer.self_attn)


class NativeStreamingLocalAttentionEncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        *,
        input_feature_frames: int,
        valid_output_steps: int,
        channel_cache_shape: tuple[int, ...],
        time_cache_shape: tuple[int, ...],
    ):
        super().__init__()
        self.encoder = encoder
        self.input_feature_frames = int(input_feature_frames)
        self.valid_output_steps = int(valid_output_steps)
        self.feat_in = int(encoder._feat_in)
        self.d_model = int(encoder.d_model)
        self.num_stage_caches = int(channel_cache_shape[0])
        self.channel_cache_shape = tuple(int(x) for x in channel_cache_shape)
        self.time_cache_shape = tuple(int(x) for x in time_cache_shape)

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        input_cache: torch.Tensor,
        input_time_cache: torch.Tensor,
        input_cache_length: torch.Tensor,
    ):
        if audio_signal.dim() != 3:
            raise ValueError(f"audio_signal rank must be 3, got {audio_signal.dim()}")
        if input_cache.dim() != 4:
            raise ValueError(f"input_cache rank must be 4, got {input_cache.dim()}")
        if input_time_cache.dim() != 4:
            raise ValueError(f"input_time_cache rank must be 4, got {input_time_cache.dim()}")
        if tuple(int(x) for x in input_cache.shape) != self.channel_cache_shape:
            raise ValueError(
                "input_cache shape mismatch: "
                f"got {tuple(input_cache.shape)} expected {self.channel_cache_shape}"
            )
        if tuple(int(x) for x in input_time_cache.shape) != self.time_cache_shape:
            raise ValueError(
                "input_time_cache shape mismatch: "
                f"got {tuple(input_time_cache.shape)} expected {self.time_cache_shape}"
            )

        rets = self.encoder.forward_internal(
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=input_cache,
            cache_last_time=input_time_cache,
            cache_last_channel_len=input_cache_length,
            bypass_pre_encode=False,
        )
        outputs, encoded_lengths, next_cache, next_time_cache, next_cache_length = self.encoder.streaming_post_process(
            rets,
            keep_all_outputs=False,
        )
        return (
            outputs,
            encoded_lengths.to(dtype=torch.int64),
            next_cache,
            next_time_cache,
            next_cache_length.to(dtype=torch.int64),
        )


def _build_manifest(
    *,
    onnx_path_hint: str,
    feat_in: int,
    d_model: int,
    channel_cache_shape: tuple[int, ...],
    time_cache_shape: tuple[int, ...],
    input_feature_frames: int,
    valid_output_steps: int,
    streaming_info: dict[str, Any],
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
            {
                "name": "input_cache",
                "dtype": "FLOAT",
                "shape": list(channel_cache_shape),
            },
            {
                "name": "input_time_cache",
                "dtype": "FLOAT",
                "shape": list(time_cache_shape),
            },
            {
                "name": "input_cache_length",
                "dtype": "INT64",
                "shape": [1],
            },
        ],
        "outputs": [
            {
                "name": "outputs",
                "dtype": "FLOAT",
                "shape": [1, d_model, valid_output_steps],
            },
            {
                "name": "encoded_lengths",
                "dtype": "INT64",
                "shape": [1],
            },
            {
                "name": "output_cache",
                "dtype": "FLOAT",
                "shape": list(channel_cache_shape),
            },
            {
                "name": "output_time_cache",
                "dtype": "FLOAT",
                "shape": list(time_cache_shape),
            },
            {
                "name": "output_cache_length",
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
        "streaming_info": streaming_info,
        "wrapped_local_attention_streaming_encoder": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a cache-aware local-attention encoder TorchScript wrapper.")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="NeMo model name. Example: nvidia/parakeet-tdt-0.6b-v2",
    )
    parser.add_argument("--output-ts", type=Path, required=True, help="Output TorchScript path.")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Output manifest JSON path.")
    parser.add_argument(
        "--output-streaming-config",
        type=Path,
        default=None,
        help="Optional exact-name sidecar JSON for runtime streaming metadata.",
    )
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
        "--chunk-steps",
        type=int,
        required=True,
        help="Streaming chunk size in encoder steps.",
    )
    parser.add_argument(
        "--shift-steps",
        type=int,
        required=True,
        help="Streaming shift size in encoder steps.",
    )
    parser.add_argument(
        "--left-chunks",
        type=int,
        default=None,
        help="Optional NeMo left-chunks override. Defaults to left_context_steps / chunk_steps.",
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

    att_context_size = [int(args.left_context_steps), int(args.right_context_steps)]
    left_chunks = args.left_chunks
    if left_chunks is None:
        left_chunks = max(1, int(args.left_context_steps) // max(1, int(args.chunk_steps)))

    encoder.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=att_context_size,
        update_config=True,
        device=torch.device("cpu"),
    )
    _make_local_attention_export_friendly(encoder)
    encoder.setup_streaming_params(
        chunk_size=int(args.chunk_steps),
        shift_size=int(args.shift_steps),
        left_chunks=int(left_chunks),
        att_context_size=att_context_size,
    )

    streaming_cfg = encoder.streaming_cfg
    input_feature_frames = _resolve_streaming_value(streaming_cfg.chunk_size, 1) + _resolve_streaming_value(
        streaming_cfg.pre_encode_cache_size, 1
    )
    valid_output_steps = _resolve_streaming_value(streaming_cfg.valid_out_len, 0)
    drop_extra_pre_encoded = int(streaming_cfg.drop_extra_pre_encoded)
    left_context_steps = int(args.left_context_steps)

    channel_cache, time_cache, cache_len = encoder.get_initial_cache_state(
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    wrapper = NativeStreamingLocalAttentionEncoderWrapper(
        encoder=encoder,
        input_feature_frames=input_feature_frames,
        valid_output_steps=valid_output_steps,
        channel_cache_shape=tuple(int(x) for x in channel_cache.shape),
        time_cache_shape=tuple(int(x) for x in time_cache.shape),
    )
    wrapper.eval()

    example_audio = torch.randn(1, wrapper.feat_in, input_feature_frames, dtype=torch.float32)
    example_length = torch.tensor([input_feature_frames], dtype=torch.int64)
    example_cache = channel_cache.clone()
    example_time_cache = time_cache.clone()
    example_cache_length = cache_len.clone()

    with torch.no_grad():
        _ = wrapper(example_audio, example_length, example_cache, example_time_cache, example_cache_length)

    try:
        wrapped_ts = torch.jit.script(wrapper)
    except Exception:
        wrapped_ts = torch.jit.trace(
            wrapper,
            (example_audio, example_length, example_cache, example_time_cache, example_cache_length),
            check_trace=False,
        )

    args.output_ts.parent.mkdir(parents=True, exist_ok=True)
    wrapped_ts.save(str(args.output_ts))

    streaming_info = {
        "kind": "local_attention_nemo_streaming_cache",
        "attention_model": "rel_pos_local_attn",
        "left_context_steps": left_context_steps,
        "right_context_steps": int(args.right_context_steps),
        "chunk_steps": int(args.chunk_steps),
        "shift_steps": int(args.shift_steps),
        "left_chunks": int(left_chunks),
        "input_feature_frames": int(input_feature_frames),
        "shift_feature_frames": int(_resolve_streaming_value(streaming_cfg.shift_size, 1)),
        "pre_encode_cache_frames": int(_resolve_streaming_value(streaming_cfg.pre_encode_cache_size, 1)),
        "drop_extra_pre_encoded": drop_extra_pre_encoded,
        "valid_output_steps": int(valid_output_steps),
        "stage_cache_count": int(wrapper.num_stage_caches),
        "state_input_name": "input_cache",
        "time_state_input_name": "input_time_cache",
        "state_length_input_name": "input_cache_length",
        "state_output_name": "output_cache",
        "time_state_output_name": "output_time_cache",
        "state_length_output_name": "output_cache_length",
        "last_channel_cache_steps": int(channel_cache.shape[2]),
        "time_cache_steps": int(time_cache.shape[3]),
    }

    manifest = _build_manifest(
        onnx_path_hint=str(args.output_ts),
        feat_in=wrapper.feat_in,
        d_model=wrapper.d_model,
        channel_cache_shape=tuple(int(x) for x in channel_cache.shape),
        time_cache_shape=tuple(int(x) for x in time_cache.shape),
        input_feature_frames=input_feature_frames,
        valid_output_steps=valid_output_steps,
        streaming_info=streaming_info,
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.output_streaming_config is not None:
        args.output_streaming_config.parent.mkdir(parents=True, exist_ok=True)
        args.output_streaming_config.write_text(json.dumps(streaming_info, indent=2), encoding="utf-8")

    print(f"Local-attention streaming encoder TorchScript: {args.output_ts}")
    print(f"Local-attention streaming encoder manifest: {args.output_manifest}")
    if args.output_streaming_config is not None:
        print(f"Local-attention streaming encoder config: {args.output_streaming_config}")


if __name__ == "__main__":
    main()
