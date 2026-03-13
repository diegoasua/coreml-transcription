#!/usr/bin/env python3
"""NeMo local-attention streaming oracle for Parakeet TDT.

Purpose:
- prototype inference-only local-attention streaming before CoreML/Swift work
- compare short fixtures against the current realtime baseline

Notes:
- NeMo's stock local-attention cache path assumes q/k have identical time length.
- The MLX implementation in `/Users/diegoasua/Developer/parakeet-mlx` supports
  right-aligned asymmetric q/k lengths for cache-aware local attention.
- This oracle can run either:
  - `patched-cache`: monkeypatch NeMo local attention with MLX-style asymmetric
    q/k handling so encoder KV cache works for local attention.
  - `overlap-no-cache`: overlap chunks and carry only RNNT hypotheses, with no
    encoder cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_for_wer(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _parse_context(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected left,right context, got {raw!r}")
    return [int(parts[0]), int(parts[1])]


def _simple_wer(reference: list[str], hypothesis: list[str]) -> float:
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    if m == 0:
        return 0.0 if n == 0 else 1.0
    return dp[m][n] / m


def _load_model(model_name: str):
    import torch
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    model = EncDecRNNTBPEModel.from_pretrained(model_name=model_name)
    model.eval()
    model.to(torch.device("cpu"))
    return model


def _maybe_extract_text(result: Any) -> str:
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        first = result[0]
        if hasattr(first, "text"):
            return _normalize_text(str(first.text))
        return _normalize_text(str(first))
    if hasattr(result, "text"):
        return _normalize_text(str(result.text))
    return _normalize_text(str(result))


def _run_offline(model, audio_path: str) -> tuple[str, float]:
    started = time.perf_counter()
    out = model.transcribe([audio_path], batch_size=1, return_hypotheses=False)
    elapsed = time.perf_counter() - started
    return _maybe_extract_text(out), elapsed


def _patch_local_attention_cache_support(model) -> None:
    from nemo.collections.asr.parts.submodules.multi_head_attention import (
        RelPositionMultiHeadAttentionLongformer,
    )

    def qk(self, q, k, w, padding_value):
        import torch

        bsz, num_heads, seq_q, head_dim = q.shape
        seq_k = k.shape[2]
        device = q.device

        rel = torch.arange(2 * w + 1, device=device) - w
        base = (seq_k - seq_q) + torch.arange(seq_q, device=device)
        idx = base[:, None] + rel[None, :]
        valid = (idx >= 0) & (idx < seq_k)
        idx = idx.clamp(0, seq_k - 1)

        idx = idx.view(1, 1, seq_q, 2 * w + 1, 1).expand(bsz, num_heads, seq_q, 2 * w + 1, head_dim)
        k_expand = k.unsqueeze(3).expand(bsz, num_heads, seq_k, 2 * w + 1, head_dim)
        gathered = torch.gather(k_expand, 2, idx)
        out = (q.unsqueeze(3) * gathered).sum(-1)
        return out.masked_fill(~valid.view(1, 1, seq_q, 2 * w + 1), padding_value)

    def pv(self, prob, v, w):
        import torch

        bsz, num_heads, seq_q, width = prob.shape
        seq_k = v.shape[2]
        head_dim = v.shape[3]
        device = prob.device

        rel = torch.arange(width, device=device) - w
        base = (seq_k - seq_q) + torch.arange(seq_q, device=device)
        idx = base[:, None] + rel[None, :]
        valid = (idx >= 0) & (idx < seq_k)
        idx = idx.clamp(0, seq_k - 1)

        idx = idx.view(1, 1, seq_q, width, 1).expand(bsz, num_heads, seq_q, width, head_dim)
        v_expand = v.unsqueeze(3).expand(bsz, num_heads, seq_k, width, head_dim)
        gathered = torch.gather(v_expand, 2, idx)

        weights = prob.masked_fill(~valid.view(1, 1, seq_q, width), 0.0).unsqueeze(-1)
        out = (weights * gathered).sum(3)
        return out.transpose(1, 2)

    for layer in model.encoder.layers:
        if isinstance(layer.self_attn, RelPositionMultiHeadAttentionLongformer):
            layer.self_attn.sliding_chunks_matmul_qk = types.MethodType(qk, layer.self_attn)
            layer.self_attn.sliding_chunks_matmul_pv = types.MethodType(pv, layer.self_attn)


def _configure_custom_wrapper_streaming(model, att_context_size: list[int], shift_steps: int, depth: int) -> None:
    keep_size = att_context_size[0]
    drop_size = att_context_size[1] * depth
    chunk_steps = shift_steps + drop_size

    _configure_local_attention(model, att_context_size, chunk_steps, shift_steps)
    _patch_local_attention_cache_support(model)

    model.encoder.streaming_cfg.last_channel_cache_size = keep_size
    model.encoder.streaming_cfg.cache_drop_size = drop_size
    for module in model.encoder.layers.modules():
        if hasattr(module, "_max_cache_len") and hasattr(module, "cache_drop_size"):
            module.cache_drop_size = drop_size


def _decode_hypotheses(model, encoded, encoded_len, partial_hypotheses):
    if int(encoded_len.item()) <= 0:
        return partial_hypotheses
    return model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoded,
        encoded_lengths=encoded_len,
        return_hypotheses=True,
        partial_hypotheses=partial_hypotheses,
    )


@dataclass
class _CustomDecodeState:
    token_ids: list[int]
    last_token: int | None
    hidden_state: Any


@dataclass
class _TensorFrameCache:
    capacity: int
    value: Any = None

    def clear(self) -> None:
        self.value = None

    def length(self) -> int:
        if self.value is None:
            return 0
        return int(self.value.shape[1])

    def concat(self, current):
        import torch

        if self.value is None:
            return current
        return current if self.length() == 0 else torch.cat([self.value, current], dim=1)

    def append_tail(self, current, keep: int) -> None:
        import torch

        if keep <= 0 or current.size(1) <= 0 or self.capacity <= 0:
            return
        tail = current[:, -min(keep, current.size(1)) :, :].detach().clone()
        if self.value is None:
            updated = tail
        else:
            updated = torch.cat([self.value, tail], dim=1)
        if updated.size(1) > self.capacity:
            updated = updated[:, -self.capacity :, :]
        self.value = updated


@dataclass
class _RotatingSequenceCache:
    capacity: int
    buffer: Any = None
    valid: int = 0
    written: int = 0

    def clear(self) -> None:
        self.buffer = None
        self.valid = 0
        self.written = 0

    def length(self) -> int:
        return int(self.valid)

    def fetch(self):
        import torch

        if self.buffer is None or self.valid <= 0:
            return None
        if self.valid < self.capacity:
            return self.buffer[:, : self.valid, :]
        pos = self.written % self.capacity
        if pos == 0:
            return self.buffer
        return torch.roll(self.buffer, shifts=-pos, dims=1)

    def append(self, x) -> None:
        import torch

        if self.capacity <= 0 or x is None or x.size(1) <= 0:
            return
        steps = min(int(x.size(1)), self.capacity)
        chunk = x[:, -steps:, :].detach().clone()
        if self.buffer is None:
            self.buffer = torch.zeros(
                chunk.size(0),
                self.capacity,
                chunk.size(2),
                dtype=chunk.dtype,
                device=chunk.device,
            )
        pos = self.written % self.capacity
        first = min(steps, self.capacity - pos)
        self.buffer[:, pos : pos + first, :] = chunk[:, :first, :]
        if steps > first:
            self.buffer[:, : steps - first, :] = chunk[:, first:, :]
        self.written += steps
        self.valid = min(self.capacity, self.valid + steps)


@dataclass
class _RotatingConvCache:
    capacity: int
    buffer: Any = None
    valid: int = 0
    written: int = 0

    def clear(self) -> None:
        self.buffer = None
        self.valid = 0
        self.written = 0

    def fetch(self):
        import torch

        if self.capacity <= 0:
            return None
        if self.buffer is None:
            return None
        if self.valid < self.capacity:
            return self.buffer[:, :, : self.valid]
        pos = self.written % self.capacity
        if pos == 0:
            return self.buffer
        return torch.roll(self.buffer, shifts=-pos, dims=2)

    def append(self, x) -> None:
        import torch

        if self.capacity <= 0 or x is None or x.size(2) <= 0:
            return
        steps = min(int(x.size(2)), self.capacity)
        chunk = x[:, :, -steps:].detach().clone()
        if self.buffer is None:
            self.buffer = torch.zeros(
                chunk.size(0),
                chunk.size(1),
                self.capacity,
                dtype=chunk.dtype,
                device=chunk.device,
            )
        pos = self.written % self.capacity
        first = min(steps, self.capacity - pos)
        self.buffer[:, :, pos : pos + first] = chunk[:, :, :first]
        if steps > first:
            self.buffer[:, :, : steps - first] = chunk[:, :, first:]
        self.written += steps
        self.valid = min(self.capacity, self.valid + steps)


@dataclass
class _ManualLayerCacheState:
    att: _RotatingSequenceCache
    conv: _RotatingConvCache

    def clear(self) -> None:
        self.att.clear()
        self.conv.clear()


def _clone_hidden_state(hidden_state):
    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):
        return tuple(item.clone() for item in hidden_state)
    if isinstance(hidden_state, list):
        return [item.clone() for item in hidden_state]
    return hidden_state


def _hypothesis_text(hypotheses) -> str:
    if not hypotheses:
        return ""
    return _normalize_text(getattr(hypotheses[0], "text", str(hypotheses[0])))


def _decode_token_ids(model, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    tokens = model.decoding.decode_ids_to_tokens(token_ids)
    return _normalize_text(model.decoding.decode_tokens_to_str(tokens))


def _decode_token_piece(model, token_id: int, *, blank_id: int) -> str:
    if token_id == blank_id:
        return "<blank>"
    pieces = model.decoding.decode_ids_to_tokens([token_id])
    if not pieces:
        return f"<tok:{token_id}>"
    return str(pieces[0])


def _merge_segment_text(existing: str, new: str, max_overlap_words: int = 12) -> str:
    existing = _normalize_text(existing)
    new = _normalize_text(new)
    if not existing:
        return new
    if not new:
        return existing

    existing_words = existing.split()
    new_words = new.split()
    max_overlap = min(max_overlap_words, len(existing_words), len(new_words))
    for overlap in range(max_overlap, 0, -1):
        if existing_words[-overlap:] == new_words[:overlap]:
            return " ".join(existing_words + new_words[overlap:])
    return f"{existing} {new}"


def _stream_cfg_scalar(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return int(value[-1])
    return int(value)


def _install_manual_layer_cache_hooks(model) -> None:
    import torch
    import torch.nn.functional as F

    def make_att_update(orig_update):
        def update_cache(self, key, value, query, cache):
            if isinstance(cache, _RotatingSequenceCache):
                history = cache.fetch()
                if history is not None:
                    history = history.to(device=key.device, dtype=key.dtype)
                    key = torch.cat([history, key], dim=1)
                    value = torch.cat([history, value], dim=1)
                keep = max(0, query.shape[1] - int(getattr(self, "cache_drop_size", 0) or 0))
                if keep > 0:
                    cache.append(query[:, :keep, :])
                return key, value, query, cache
            return orig_update(key, value, query, cache)

        return update_cache

    def make_conv_update(orig_update):
        def update_cache(self, x, cache=None):
            if isinstance(cache, _RotatingConvCache):
                right_padded = F.pad(x, pad=(0, self._right_padding))
                history = cache.fetch()
                if history is None and self.capacity_hint > 0:
                    history = right_padded.new_zeros((x.size(0), x.size(1), self.capacity_hint))
                if history is not None:
                    history = history.to(device=right_padded.device, dtype=right_padded.dtype)
                    new_x = torch.cat([history, right_padded], dim=-1)
                else:
                    new_x = right_padded
                drop = int(getattr(self, "cache_drop_size", 0) or 0)
                cache_source = new_x[:, :, :-drop] if drop > 0 else new_x
                cache.append(cache_source)
                return new_x, cache
            return orig_update(x, cache)

        return update_cache

    for layer in model.encoder.layers:
        att = layer.self_attn
        if not getattr(att, "_manual_layer_cache_patch", False):
            att._orig_update_cache = att.update_cache
            att.update_cache = types.MethodType(make_att_update(att.update_cache), att)
            att._manual_layer_cache_patch = True

        conv = layer.conv.depthwise_conv
        if not getattr(conv, "_manual_layer_cache_patch", False):
            conv._orig_update_cache = conv.update_cache
            conv.capacity_hint = conv._left_padding
            conv.update_cache = types.MethodType(make_conv_update(conv.update_cache), conv)
            conv._manual_layer_cache_patch = True


def _make_manual_layer_states(model, att_capacity: int) -> list[_ManualLayerCacheState]:
    states: list[_ManualLayerCacheState] = []
    for layer in model.encoder.layers:
        states.append(
            _ManualLayerCacheState(
                att=_RotatingSequenceCache(capacity=att_capacity),
                conv=_RotatingConvCache(capacity=int(layer.conv.depthwise_conv._left_padding)),
            )
        )
    return states


def _pre_encode_chunk(model, chunk_audio, chunk_lengths):
    import torch
    from torch import nn

    audio_signal = torch.transpose(chunk_audio, 1, 2)
    if isinstance(model.encoder.pre_encode, nn.Linear):
        encoded = model.encoder.pre_encode(audio_signal)
        lengths = chunk_lengths.to(torch.int64)
        return encoded, lengths

    encoded, lengths = model.encoder.pre_encode(x=audio_signal, lengths=chunk_lengths)
    return encoded, lengths.to(torch.int64)


def _tdt_greedy_decode(
    model,
    encoded,
    encoded_len: int,
    state: _CustomDecodeState | None,
    *,
    blank_id: int,
    durations: list[int],
    max_symbols: int | None,
    trace_events: list[dict[str, Any]] | None = None,
    trace_meta: dict[str, Any] | None = None,
):
    import torch

    if encoded_len <= 0:
        if state is None:
            return _CustomDecodeState(token_ids=[], last_token=None, hidden_state=None)
        return _CustomDecodeState(
            token_ids=list(state.token_ids),
            last_token=state.last_token,
            hidden_state=_clone_hidden_state(state.hidden_state),
        )

    if state is None:
        token_ids: list[int] = []
        last_token = None
        hidden_state = None
    else:
        token_ids = list(state.token_ids)
        last_token = state.last_token
        hidden_state = _clone_hidden_state(state.hidden_state)

    step = 0
    new_symbols = 0
    device = encoded.device

    while step < encoded_len:
        frame_index = step
        token_count_before = len(token_ids)
        last_token_before = last_token
        with torch.no_grad():
            if last_token is None:
                decoder_out, decoder_hidden = model.decoder.predict(
                    None, hidden_state, add_sos=False, batch_size=1
                )
            else:
                label = torch.tensor([[last_token]], dtype=torch.long, device=device)
                decoder_out, decoder_hidden = model.decoder.predict(
                    label, hidden_state, add_sos=False, batch_size=1
                )

            enc_frame = encoded[:, :, step : step + 1].transpose(1, 2)
            joint_out = model.joint.joint(enc_frame, decoder_out)
            logits = joint_out[0, 0, 0]

        token_logits = logits[: blank_id + 1]
        duration_logits = logits[blank_id + 1 :]
        pred_token = int(torch.argmax(token_logits).item())
        duration_index = int(torch.argmax(duration_logits).item())
        duration_value = int(durations[duration_index])

        if pred_token != blank_id:
            token_ids.append(pred_token)
            last_token = pred_token
            hidden_state = decoder_hidden

        step += duration_value
        new_symbols += 1
        if duration_value != 0:
            new_symbols = 0
        elif max_symbols is not None and new_symbols >= max_symbols:
            step += 1
            new_symbols = 0

        if trace_events is not None:
            event = {
                "frame_index": frame_index,
                "step_after": step,
                "encoded_len": encoded_len,
                "pred_token_id": pred_token,
                "pred_token_piece": _decode_token_piece(model, pred_token, blank_id=blank_id),
                "duration_index": duration_index,
                "duration_value": duration_value,
                "emitted": pred_token != blank_id,
                "token_count_before": token_count_before,
                "token_count_after": len(token_ids),
                "last_token_before": last_token_before,
                "last_token_after": last_token,
                "last_token_before_piece": None
                if last_token_before is None
                else _decode_token_piece(model, last_token_before, blank_id=blank_id),
                "last_token_after_piece": None
                if last_token is None
                else _decode_token_piece(model, last_token, blank_id=blank_id),
            }
            if trace_meta:
                event.update(trace_meta)
            trace_events.append(event)

    return _CustomDecodeState(token_ids=token_ids, last_token=last_token, hidden_state=hidden_state)


def _run_custom_wrapper_streaming(
    model,
    audio_path: str,
    att_context_size: list[int],
    shift_steps: int,
    depth: int,
    max_segment_steps: int,
    min_reset_words: int,
    trace_limit: int,
    extra_draft_frames: int,
) -> tuple[str, float, dict[str, Any]]:
    import torch

    _configure_custom_wrapper_streaming(model, att_context_size, shift_steps, depth)

    drop_size = att_context_size[1] * depth
    effective_drop_size = drop_size + max(0, extra_draft_frames)
    cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(batch_size=1)
    blank_id = int(getattr(model.decoding, "blank_id", len(model.joint.vocabulary)))
    durations = list(model.decoding.cfg.get("durations", []))
    greedy_cfg = model.decoding.cfg.get("greedy", {})
    max_symbols = greedy_cfg.get("max_symbols")
    committed_state: _CustomDecodeState | None = None
    draft_state: _CustomDecodeState | None = None
    segment_text = ""
    completed_text = ""
    segment_history: list[dict[str, Any]] = []
    step_trace: list[dict[str, Any]] = []
    decode_trace: list[dict[str, Any]] = []
    decode_trace: list[dict[str, Any]] = []
    decode_trace: list[dict[str, Any]] = []
    step_count = 0
    segment_step_count = 0
    segment_reset_count = 0
    started = time.perf_counter()

    buffer_iterable = model._setup_streaming_transcribe_dataloader(
        [audio_path], batch_size=1, online_normalization=False
    )
    streaming_buffer = next(iter(buffer_iterable))

    for chunk_audio, chunk_lengths in iter(streaming_buffer):
        current_segment_text = (
            _decode_token_ids(model, draft_state.token_ids if draft_state is not None else [])
            or _decode_token_ids(model, committed_state.token_ids if committed_state is not None else [])
        )
        current_segment_words = len([w for w in current_segment_text.split() if w])
        if (
            max_segment_steps > 0
            and segment_step_count >= max_segment_steps
            and current_segment_words >= min_reset_words
        ):
            segment_final_text = current_segment_text
            segment_history.append(
                {
                    "segment_index": segment_reset_count,
                    "steps": segment_step_count,
                    "text": segment_final_text,
                }
            )
            completed_text = _merge_segment_text(
                completed_text,
                segment_final_text,
            )
            cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(
                batch_size=1
            )
            committed_state = None
            draft_state = None
            segment_text = ""
            segment_step_count = 0
            segment_reset_count += 1

        with torch.inference_mode():
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = model.encoder(
                audio_signal=chunk_audio,
                length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )

        encoded_total = int(encoded_len[0].item())
        finalized_len = max(0, encoded_total - effective_drop_size)
        draft_len = max(0, encoded_total - finalized_len)

        trace_current_step = trace_limit < 0 or step_count < trace_limit or streaming_buffer.is_buffer_empty()

        if finalized_len > 0:
            committed_state = _tdt_greedy_decode(
                model,
                encoded[:, :, :finalized_len],
                finalized_len,
                committed_state,
                blank_id=blank_id,
                durations=durations,
                max_symbols=max_symbols,
                trace_events=decode_trace if trace_current_step else None,
                trace_meta=(
                    {
                        "stream_step": step_count,
                        "segment_step": segment_step_count,
                        "phase": "committed",
                        "mode": "custom-wrapper",
                        "finalized_len": finalized_len,
                        "draft_len": draft_len,
                    }
                    if trace_current_step
                    else None
                ),
            )

        if draft_len > 0:
            draft_state = _tdt_greedy_decode(
                model,
                encoded[:, :, finalized_len:encoded_total],
                draft_len,
                committed_state,
                blank_id=blank_id,
                durations=durations,
                max_symbols=max_symbols,
                trace_events=decode_trace if trace_current_step else None,
                trace_meta=(
                    {
                        "stream_step": step_count,
                        "segment_step": segment_step_count,
                        "phase": "draft",
                        "mode": "custom-wrapper",
                        "finalized_len": finalized_len,
                        "draft_len": draft_len,
                    }
                    if trace_current_step
                    else None
                ),
            )
        else:
            draft_state = committed_state

        committed_preview = _decode_token_ids(model, committed_state.token_ids if committed_state is not None else [])
        draft_preview = _decode_token_ids(model, draft_state.token_ids if draft_state is not None else [])
        preview = draft_preview or committed_preview
        segment_text = preview

        if trace_limit < 0 or step_count < trace_limit or streaming_buffer.is_buffer_empty():
            step_trace.append(
                {
                    "step": step_count,
                    "segment_step": segment_step_count,
                    "chunk_shape": list(chunk_audio.shape),
                    "chunk_length": chunk_lengths.tolist(),
                    "encoded_total": encoded_total,
                    "finalized_len": finalized_len,
                    "draft_len": draft_len,
                    "committed_preview": committed_preview[:160],
                    "draft_preview": draft_preview[:160],
                    "cache_last_channel_len": cache_last_channel_len.tolist(),
                }
            )
        step_count += 1
        segment_step_count += 1

    elapsed = time.perf_counter() - started

    final_segment_text = (
        _decode_token_ids(model, draft_state.token_ids if draft_state is not None else [])
        or _decode_token_ids(model, committed_state.token_ids if committed_state is not None else [])
    )
    segment_history.append(
        {
            "segment_index": segment_reset_count,
            "steps": segment_step_count,
            "text": final_segment_text,
        }
    )
    final_text = _merge_segment_text(completed_text, final_segment_text)

    streaming_cfg = model.encoder.streaming_cfg
    metrics = {
        "steps": step_count,
        "segment_reset_count": segment_reset_count,
        "streaming_elapsed_sec": elapsed,
        "mode": "custom-wrapper",
        "segment_history": segment_history,
        "streaming_cfg": {
            "chunk_size": streaming_cfg.chunk_size,
            "shift_size": streaming_cfg.shift_size,
            "cache_drop_size": streaming_cfg.cache_drop_size,
            "last_channel_cache_size": streaming_cfg.last_channel_cache_size,
            "valid_out_len": streaming_cfg.valid_out_len,
            "pre_encode_cache_size": streaming_cfg.pre_encode_cache_size,
            "drop_extra_pre_encoded": streaming_cfg.drop_extra_pre_encoded,
            "wrapper_max_segment_steps": max_segment_steps,
            "wrapper_min_reset_words": min_reset_words,
            "wrapper_depth": depth,
            "wrapper_extra_draft_frames": extra_draft_frames,
        },
        "step_trace": step_trace,
        "decode_trace": decode_trace,
    }
    return _normalize_text(final_text), elapsed, metrics


def _run_manual_preencode_cache_streaming(
    model,
    audio_path: str,
    att_context_size: list[int],
    shift_steps: int,
    depth: int,
    max_segment_steps: int,
    min_reset_words: int,
    trace_limit: int,
    extra_draft_frames: int,
) -> tuple[str, float, dict[str, Any]]:
    import torch

    drop_size = att_context_size[1] * depth
    effective_drop_size = drop_size + max(0, extra_draft_frames)
    chunk_steps = shift_steps + drop_size
    _configure_local_attention(model, att_context_size, chunk_steps, shift_steps)

    blank_id = int(getattr(model.decoding, "blank_id", len(model.joint.vocabulary)))
    durations = list(model.decoding.cfg.get("durations", []))
    greedy_cfg = model.decoding.cfg.get("greedy", {})
    max_symbols = greedy_cfg.get("max_symbols")
    drop_extra_pre_encoded = _stream_cfg_scalar(model.encoder.streaming_cfg.drop_extra_pre_encoded)
    stable_cache = _TensorFrameCache(capacity=att_context_size[0])
    completed_text = ""
    segment_history: list[dict[str, Any]] = []
    step_trace: list[dict[str, Any]] = []
    decode_trace: list[dict[str, Any]] = []
    step_count = 0
    segment_step_count = 0
    segment_reset_count = 0
    current_segment_text = ""
    started = time.perf_counter()

    buffer_iterable = model._setup_streaming_transcribe_dataloader(
        [audio_path], batch_size=1, online_normalization=False
    )
    streaming_buffer = next(iter(buffer_iterable))

    for chunk_audio, chunk_lengths in iter(streaming_buffer):
        current_segment_words = len([w for w in current_segment_text.split() if w])
        if (
            max_segment_steps > 0
            and segment_step_count >= max_segment_steps
            and current_segment_words >= min_reset_words
        ):
            segment_history.append(
                {
                    "segment_index": segment_reset_count,
                    "steps": segment_step_count,
                    "text": current_segment_text,
                }
            )
            completed_text = _merge_segment_text(completed_text, current_segment_text)
            stable_cache.clear()
            current_segment_text = ""
            segment_step_count = 0
            segment_reset_count += 1

        with torch.inference_mode():
            preencoded_chunk, preencoded_len = _pre_encode_chunk(model, chunk_audio, chunk_lengths)
            if step_count != 0 and drop_extra_pre_encoded > 0:
                drop_count = min(drop_extra_pre_encoded, preencoded_chunk.size(1))
                preencoded_chunk = preencoded_chunk[:, drop_count:, :]
                preencoded_len = (preencoded_len - drop_count).clamp(min=0)

        preencoded_total = int(preencoded_len[0].item())
        preencoded_stable_len = max(0, preencoded_total - effective_drop_size)
        stable_prefix = stable_cache.concat(preencoded_chunk)
        stable_prefix_len = stable_cache.length() + preencoded_total
        stable_prefix_len_tensor = preencoded_len.new_tensor([stable_prefix_len], dtype=torch.int64)

        with torch.inference_mode():
            encoded, encoded_len = model.encoder(
                audio_signal=stable_prefix,
                length=stable_prefix_len_tensor,
                bypass_pre_encode=True,
            )

        encoded_total = int(encoded_len[0].item())
        finalized_len = max(0, encoded_total - effective_drop_size)

        trace_current_step = trace_limit < 0 or step_count < trace_limit or streaming_buffer.is_buffer_empty()

        finalized_state = _tdt_greedy_decode(
            model,
            encoded[:, :, :finalized_len],
            finalized_len,
            None,
            blank_id=blank_id,
            durations=durations,
            max_symbols=max_symbols,
            trace_events=decode_trace if trace_current_step else None,
            trace_meta=(
                {
                    "stream_step": step_count,
                    "segment_step": segment_step_count,
                    "phase": "committed",
                    "mode": "manual-preencode-cache",
                    "finalized_len": finalized_len,
                    "draft_len": max(0, encoded_total - finalized_len),
                }
                if trace_current_step
                else None
            ),
        )
        draft_state = _tdt_greedy_decode(
            model,
            encoded,
            encoded_total,
            None,
            blank_id=blank_id,
            durations=durations,
            max_symbols=max_symbols,
            trace_events=decode_trace if trace_current_step else None,
            trace_meta=(
                {
                    "stream_step": step_count,
                    "segment_step": segment_step_count,
                    "phase": "draft",
                    "mode": "manual-preencode-cache",
                    "finalized_len": finalized_len,
                    "draft_len": max(0, encoded_total - finalized_len),
                }
                if trace_current_step
                else None
            ),
        )

        finalized_text = _decode_token_ids(model, finalized_state.token_ids)
        draft_text = _decode_token_ids(model, draft_state.token_ids)
        current_segment_text = draft_text or finalized_text

        stable_cache.append_tail(preencoded_chunk[:, :preencoded_stable_len, :], preencoded_stable_len)

        if trace_limit < 0 or step_count < trace_limit or streaming_buffer.is_buffer_empty():
            step_trace.append(
                {
                    "step": step_count,
                    "segment_step": segment_step_count,
                    "chunk_shape": list(chunk_audio.shape),
                    "chunk_length": chunk_lengths.tolist(),
                    "preencoded_total": preencoded_total,
                    "preencoded_stable_len": preencoded_stable_len,
                    "stable_cache_len": stable_cache.length(),
                    "encoded_total": encoded_total,
                    "finalized_len": finalized_len,
                    "finalized_preview": finalized_text[:160],
                    "draft_preview": draft_text[:160],
                }
            )

        step_count += 1
        segment_step_count += 1

    elapsed = time.perf_counter() - started
    segment_history.append(
        {
            "segment_index": segment_reset_count,
            "steps": segment_step_count,
            "text": current_segment_text,
        }
    )
    final_text = _merge_segment_text(completed_text, current_segment_text)

    streaming_cfg = model.encoder.streaming_cfg
    metrics = {
        "steps": step_count,
        "segment_reset_count": segment_reset_count,
        "streaming_elapsed_sec": elapsed,
        "mode": "manual-preencode-cache",
        "segment_history": segment_history,
        "streaming_cfg": {
            "chunk_size": streaming_cfg.chunk_size,
            "shift_size": streaming_cfg.shift_size,
            "cache_drop_size": streaming_cfg.cache_drop_size,
            "last_channel_cache_size": streaming_cfg.last_channel_cache_size,
            "valid_out_len": streaming_cfg.valid_out_len,
            "pre_encode_cache_size": streaming_cfg.pre_encode_cache_size,
            "drop_extra_pre_encoded": streaming_cfg.drop_extra_pre_encoded,
            "wrapper_max_segment_steps": max_segment_steps,
            "wrapper_min_reset_words": min_reset_words,
            "wrapper_depth": depth,
            "wrapper_extra_draft_frames": extra_draft_frames,
        },
        "step_trace": step_trace,
        "decode_trace": decode_trace,
    }
    return _normalize_text(final_text), elapsed, metrics


def _run_manual_layer_cache_streaming(
    model,
    audio_path: str,
    att_context_size: list[int],
    shift_steps: int,
    depth: int,
    max_segment_steps: int,
    min_reset_words: int,
    trace_limit: int,
    extra_draft_frames: int,
) -> tuple[str, float, dict[str, Any]]:
    import torch

    drop_size = att_context_size[1] * depth
    effective_drop_size = drop_size + max(0, extra_draft_frames)
    chunk_steps = shift_steps + drop_size
    _configure_local_attention(model, att_context_size, chunk_steps, shift_steps)
    _patch_local_attention_cache_support(model)
    _install_manual_layer_cache_hooks(model)

    blank_id = int(getattr(model.decoding, "blank_id", len(model.joint.vocabulary)))
    durations = list(model.decoding.cfg.get("durations", []))
    greedy_cfg = model.decoding.cfg.get("greedy", {})
    max_symbols = greedy_cfg.get("max_symbols")
    drop_extra_pre_encoded = _stream_cfg_scalar(model.encoder.streaming_cfg.drop_extra_pre_encoded)
    layer_states = _make_manual_layer_states(model, att_capacity=att_context_size[0])

    completed_text = ""
    segment_history: list[dict[str, Any]] = []
    step_trace: list[dict[str, Any]] = []
    decode_trace: list[dict[str, Any]] = []
    step_count = 0
    segment_step_count = 0
    segment_reset_count = 0
    committed_state: _CustomDecodeState | None = None
    draft_state: _CustomDecodeState | None = None
    started = time.perf_counter()

    buffer_iterable = model._setup_streaming_transcribe_dataloader(
        [audio_path], batch_size=1, online_normalization=False
    )
    streaming_buffer = next(iter(buffer_iterable))

    for chunk_audio, chunk_lengths in iter(streaming_buffer):
        current_segment_text = (
            _decode_token_ids(model, draft_state.token_ids if draft_state is not None else [])
            or _decode_token_ids(model, committed_state.token_ids if committed_state is not None else [])
        )
        current_segment_words = len([w for w in current_segment_text.split() if w])
        if (
            max_segment_steps > 0
            and segment_step_count >= max_segment_steps
            and current_segment_words >= min_reset_words
        ):
            segment_history.append(
                {
                    "segment_index": segment_reset_count,
                    "steps": segment_step_count,
                    "text": current_segment_text,
                }
            )
            completed_text = _merge_segment_text(completed_text, current_segment_text)
            for state in layer_states:
                state.clear()
            committed_state = None
            draft_state = None
            segment_step_count = 0
            segment_reset_count += 1

        with torch.inference_mode():
            preencoded_chunk, preencoded_len = _pre_encode_chunk(model, chunk_audio, chunk_lengths)
            if step_count != 0 and drop_extra_pre_encoded > 0:
                drop_count = min(drop_extra_pre_encoded, preencoded_chunk.size(1))
                preencoded_chunk = preencoded_chunk[:, drop_count:, :]
                preencoded_len = (preencoded_len - drop_count).clamp(min=0)

            cache_len = layer_states[0].att.length() if layer_states else 0
            x, pos_emb = model.encoder.pos_enc(x=preencoded_chunk, cache_len=cache_len)
            padding_length = preencoded_len + cache_len
            max_audio_length = x.size(1) + cache_len
            pad_mask, att_mask = model.encoder._create_masks(
                att_context_size=model.encoder.att_context_size,
                padding_length=padding_length,
                max_audio_length=max_audio_length,
                offset=None,
                device=x.device,
            )
            if cache_len > 0:
                pad_mask = pad_mask[:, cache_len:]
                if att_mask is not None:
                    att_mask = att_mask[:, cache_len:]

            for layer, state in zip(model.encoder.layers, layer_states):
                layer_out = layer(
                    x=x,
                    att_mask=att_mask,
                    pos_emb=pos_emb,
                    pad_mask=pad_mask,
                    cache_last_channel=state.att,
                    cache_last_time=state.conv,
                )
                x, _, _ = layer_out

            if model.encoder.out_proj is not None:
                x = model.encoder.out_proj(x)

            encoded = x.transpose(1, 2)

        encoded_total = int(preencoded_len[0].item())
        finalized_len = max(0, encoded_total - effective_drop_size)
        draft_len = max(0, encoded_total - finalized_len)

        trace_current_step = trace_limit < 0 or step_count < trace_limit or streaming_buffer.is_buffer_empty()

        if finalized_len > 0:
            committed_state = _tdt_greedy_decode(
                model,
                encoded[:, :, :finalized_len],
                finalized_len,
                committed_state,
                blank_id=blank_id,
                durations=durations,
                max_symbols=max_symbols,
                trace_events=decode_trace if trace_current_step else None,
                trace_meta=(
                    {
                        "stream_step": step_count,
                        "segment_step": segment_step_count,
                        "phase": "committed",
                        "mode": "manual-layer-cache",
                        "finalized_len": finalized_len,
                        "draft_len": draft_len,
                        "cache_len": cache_len,
                    }
                    if trace_current_step
                    else None
                ),
            )

        if draft_len > 0:
            draft_state = _tdt_greedy_decode(
                model,
                encoded[:, :, finalized_len:encoded_total],
                draft_len,
                committed_state,
                blank_id=blank_id,
                durations=durations,
                max_symbols=max_symbols,
                trace_events=decode_trace if trace_current_step else None,
                trace_meta=(
                    {
                        "stream_step": step_count,
                        "segment_step": segment_step_count,
                        "phase": "draft",
                        "mode": "manual-layer-cache",
                        "finalized_len": finalized_len,
                        "draft_len": draft_len,
                        "cache_len": cache_len,
                    }
                    if trace_current_step
                    else None
                ),
            )
        else:
            draft_state = committed_state

        committed_preview = _decode_token_ids(model, committed_state.token_ids if committed_state is not None else [])
        draft_preview = _decode_token_ids(model, draft_state.token_ids if draft_state is not None else [])

        if trace_limit < 0 or step_count < trace_limit or streaming_buffer.is_buffer_empty():
            step_trace.append(
                {
                    "step": step_count,
                    "segment_step": segment_step_count,
                    "chunk_shape": list(chunk_audio.shape),
                    "chunk_length": chunk_lengths.tolist(),
                    "preencoded_total": int(preencoded_len[0].item()),
                    "cache_len": cache_len,
                    "encoded_total": encoded_total,
                    "finalized_len": finalized_len,
                    "draft_len": draft_len,
                    "committed_preview": committed_preview[:160],
                    "draft_preview": draft_preview[:160],
                    "layer0_conv_cache_len": layer_states[0].conv.valid if layer_states else 0,
                }
            )

        step_count += 1
        segment_step_count += 1

    elapsed = time.perf_counter() - started
    final_segment_text = (
        _decode_token_ids(model, draft_state.token_ids if draft_state is not None else [])
        or _decode_token_ids(model, committed_state.token_ids if committed_state is not None else [])
    )
    segment_history.append(
        {
            "segment_index": segment_reset_count,
            "steps": segment_step_count,
            "text": final_segment_text,
        }
    )
    final_text = _merge_segment_text(completed_text, final_segment_text)

    streaming_cfg = model.encoder.streaming_cfg
    metrics = {
        "steps": step_count,
        "segment_reset_count": segment_reset_count,
        "streaming_elapsed_sec": elapsed,
        "mode": "manual-layer-cache",
        "segment_history": segment_history,
        "streaming_cfg": {
            "chunk_size": streaming_cfg.chunk_size,
            "shift_size": streaming_cfg.shift_size,
            "cache_drop_size": streaming_cfg.cache_drop_size,
            "last_channel_cache_size": streaming_cfg.last_channel_cache_size,
            "valid_out_len": streaming_cfg.valid_out_len,
            "pre_encode_cache_size": streaming_cfg.pre_encode_cache_size,
            "drop_extra_pre_encoded": streaming_cfg.drop_extra_pre_encoded,
            "wrapper_max_segment_steps": max_segment_steps,
            "wrapper_min_reset_words": min_reset_words,
            "wrapper_depth": depth,
            "wrapper_extra_draft_frames": extra_draft_frames,
        },
        "step_trace": step_trace,
        "decode_trace": decode_trace,
    }
    return _normalize_text(final_text), elapsed, metrics


def _configure_local_attention(model, att_context_size: list[int], chunk_steps: int, shift_steps: int) -> None:
    model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=att_context_size,
        update_config=False,
    )
    model.encoder.setup_streaming_params(
        chunk_size=chunk_steps,
        shift_size=shift_steps,
        left_chunks=None,
        att_context_size=att_context_size,
    )


def _run_local_streaming(
    model,
    audio_path: str,
    att_context_size: list[int],
    chunk_steps: int,
    shift_steps: int,
    online_normalization: bool,
    mode: str,
    depth: int,
    max_segment_steps: int,
    min_reset_words: int,
    trace_limit: int,
    extra_draft_frames: int,
) -> tuple[str, float, dict[str, Any]]:
    if mode == "custom-wrapper":
        return _run_custom_wrapper_streaming(
            model,
            audio_path,
            att_context_size=att_context_size,
            shift_steps=shift_steps,
            depth=depth,
            max_segment_steps=max_segment_steps,
            min_reset_words=min_reset_words,
            trace_limit=trace_limit,
            extra_draft_frames=extra_draft_frames,
        )
    if mode == "manual-preencode-cache":
        return _run_manual_preencode_cache_streaming(
            model,
            audio_path,
            att_context_size=att_context_size,
            shift_steps=shift_steps,
            depth=depth,
            max_segment_steps=max_segment_steps,
            min_reset_words=min_reset_words,
            trace_limit=trace_limit,
            extra_draft_frames=extra_draft_frames,
        )
    if mode == "manual-layer-cache":
        return _run_manual_layer_cache_streaming(
            model,
            audio_path,
            att_context_size=att_context_size,
            shift_steps=shift_steps,
            depth=depth,
            max_segment_steps=max_segment_steps,
            min_reset_words=min_reset_words,
            trace_limit=trace_limit,
            extra_draft_frames=extra_draft_frames,
        )

    _configure_local_attention(model, att_context_size, chunk_steps, shift_steps)
    if mode == "patched-cache":
        _patch_local_attention_cache_support(model)

    started = time.perf_counter()
    buffer_iterable = model._setup_streaming_transcribe_dataloader(
        [audio_path], batch_size=1, online_normalization=online_normalization
    )
    streaming_buffer = next(iter(buffer_iterable))

    if mode == "patched-cache":
        cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(
            batch_size=1
        )
    else:
        cache_last_channel = None
        cache_last_time = None
        cache_last_channel_len = None

    previous_hypotheses = None
    pred_out_stream = None
    transcribed_texts = None
    step_count = 0
    step_trace: list[dict[str, Any]] = []

    for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
        drop_extra_pre_encoded = model.encoder.streaming_cfg.drop_extra_pre_encoded if step_num != 0 else 0
        result = model.conformer_stream_step(
            processed_signal=chunk_audio,
            processed_signal_length=chunk_lengths,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=streaming_buffer.is_buffer_empty(),
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=pred_out_stream,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            return_transcription=True,
            return_log_probs=False,
        )
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = result
        step_count += 1
        if trace_limit < 0 or step_num < trace_limit or streaming_buffer.is_buffer_empty():
            preview = ""
            if transcribed_texts:
                preview = getattr(transcribed_texts[0], "text", str(transcribed_texts[0]))
            step_trace.append(
                {
                    "step": step_num,
                    "chunk_shape": list(chunk_audio.shape),
                    "chunk_length": chunk_lengths.tolist(),
                    "preview": preview[:160],
                    "cache_last_channel_len": None if cache_last_channel_len is None else cache_last_channel_len.tolist(),
                }
            )

    elapsed = time.perf_counter() - started
    text = ""
    if transcribed_texts:
        text = getattr(transcribed_texts[0], "text", str(transcribed_texts[0]))

    streaming_cfg = model.encoder.streaming_cfg
    metrics = {
        "steps": step_count,
        "streaming_elapsed_sec": elapsed,
        "mode": mode,
        "streaming_cfg": {
            "chunk_size": streaming_cfg.chunk_size,
            "shift_size": streaming_cfg.shift_size,
            "cache_drop_size": streaming_cfg.cache_drop_size,
            "last_channel_cache_size": streaming_cfg.last_channel_cache_size,
            "valid_out_len": streaming_cfg.valid_out_len,
            "pre_encode_cache_size": streaming_cfg.pre_encode_cache_size,
            "drop_extra_pre_encoded": streaming_cfg.drop_extra_pre_encoded,
        },
        "step_trace": step_trace,
    }
    return _normalize_text(text), elapsed, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="NeMo local-attention streaming oracle")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, required=True)
    parser.add_argument("--reference-txt", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=os.environ.get("PARAKEET_NEMO_MODEL", "nvidia/parakeet-tdt-0.6b-v2"),
    )
    parser.add_argument("--att-context", default=os.environ.get("PARAKEET_NEMO_LOCAL_ATT_CONTEXT", "20,0"))
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=int(os.environ.get("PARAKEET_NEMO_STREAM_CHUNK_STEPS", "3")),
    )
    parser.add_argument(
        "--shift-steps",
        type=int,
        default=int(os.environ.get("PARAKEET_NEMO_STREAM_SHIFT_STEPS", "3")),
    )
    parser.add_argument(
        "--mode",
        choices=["patched-cache", "overlap-no-cache", "custom-wrapper", "manual-preencode-cache", "manual-layer-cache"],
        default=os.environ.get("PARAKEET_NEMO_STREAM_MODE", "custom-wrapper"),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=int(os.environ.get("PARAKEET_NEMO_STREAM_DEPTH", "1")),
    )
    parser.add_argument(
        "--max-segment-steps",
        type=int,
        default=int(os.environ.get("PARAKEET_NEMO_STREAM_MAX_SEGMENT_STEPS", "64")),
    )
    parser.add_argument(
        "--min-reset-words",
        type=int,
        default=int(os.environ.get("PARAKEET_NEMO_STREAM_MIN_RESET_WORDS", "4")),
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=int(os.environ.get("PARAKEET_NEMO_STREAM_TRACE_LIMIT", "6")),
    )
    parser.add_argument(
        "--extra-draft-frames",
        type=int,
        default=int(os.environ.get("PARAKEET_NEMO_STREAM_EXTRA_DRAFT_FRAMES", "0")),
    )
    parser.add_argument("--online-normalization", action="store_true")
    parser.add_argument("--also-offline", action="store_true")
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)

    att_context = _parse_context(args.att_context)
    model = _load_model(args.model)

    summary: dict[str, Any] = {
        "audio": str(args.audio),
        "model": args.model,
        "self_attention_model": "rel_pos_local_attn",
        "att_context_size": att_context,
        "chunk_steps": args.chunk_steps,
        "shift_steps": args.shift_steps,
        "depth": args.depth,
        "mode": args.mode,
        "online_normalization": args.online_normalization,
        "extra_draft_frames": args.extra_draft_frames,
    }

    if args.also_offline:
        try:
            offline_text, offline_sec = _run_offline(model, str(args.audio))
            summary["offline_elapsed_sec"] = offline_sec
            summary["offline_preview"] = offline_text[:240]
        except Exception as exc:
            summary["offline_error"] = f"{type(exc).__name__}: {exc}"

    streaming_text, streaming_sec, streaming_metrics = _run_local_streaming(
        model,
        str(args.audio),
        att_context_size=att_context,
        chunk_steps=args.chunk_steps,
        shift_steps=args.shift_steps,
        online_normalization=args.online_normalization,
        mode=args.mode,
        depth=args.depth,
        max_segment_steps=args.max_segment_steps,
        min_reset_words=args.min_reset_words,
        trace_limit=args.trace_limit,
        extra_draft_frames=args.extra_draft_frames,
    )
    summary["streaming_elapsed_sec"] = streaming_sec
    summary["streaming_preview"] = streaming_text[:240]
    summary.update(streaming_metrics)

    if args.reference_txt is not None and args.reference_txt.exists():
        ref_text = _normalize_for_wer(args.reference_txt.read_text(encoding="utf-8", errors="replace"))
        hyp_text = _normalize_for_wer(streaming_text)
        ref_words = [w for w in ref_text.split(" ") if w]
        hyp_words = [w for w in hyp_text.split(" ") if w]
        summary["reference_word_count"] = len(ref_words)
        summary["hypothesis_word_count"] = len(hyp_words)
        summary["simple_wer"] = _simple_wer(ref_words, hyp_words)
        summary["reference_preview"] = ref_text[:240]

    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    args.output_txt.write_text(streaming_text + "\n", encoding="utf-8")
    print(args.output_json)
    print(args.output_txt)


if __name__ == "__main__":
    main()
