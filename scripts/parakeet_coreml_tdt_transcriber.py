#!/usr/bin/env python3
"""Local Parakeet v2 CoreML TDT transcriber.

Exports:
  transcribe_file(audio_path, language=None, keywords=None) -> str
  stream_transcribe_file(audio_path, language=None, keywords=None) -> dict

Can also run as CLI:
  python scripts/parakeet_coreml_tdt_transcriber.py --audio path.wav
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_tmpdir(anchor: Path) -> None:
    def _is_writable_dir(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    current = os.environ.get("TMPDIR")
    if current:
        if _is_writable_dir(Path(current).expanduser()):
            return
    tmp_dir = (anchor.parent / ".tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)


def _text_from_sentencepiece_pieces(pieces: list[str]) -> str:
    text = "".join(pieces).replace("▁", " ")
    text = " ".join(text.split())
    return text.strip()


def _longest_common_prefix_tokens(token_lists: list[list[str]]) -> list[str]:
    if not token_lists:
        return []
    prefix = list(token_lists[0])
    for tokens in token_lists[1:]:
        i = 0
        limit = min(len(prefix), len(tokens))
        while i < limit and prefix[i] == tokens[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def _merge_text_by_overlap(chunks: list[str]) -> str:
    merged: list[str] = []
    for text in chunks:
        tokens = text.split()
        if not tokens:
            continue
        if not merged:
            merged.extend(tokens)
            continue
        max_overlap = min(len(merged), len(tokens), 64)
        overlap = 0
        for candidate in range(max_overlap, 0, -1):
            if merged[-candidate:] == tokens[:candidate]:
                overlap = candidate
                break
        merged.extend(tokens[overlap:])
    return " ".join(merged).strip()


@dataclass
class _InputSpec:
    name: str
    shape: list[int]
    dtype: Any


@dataclass
class _BeamHypothesis:
    score: float
    tokens: list[int]
    state1: np.ndarray
    state2: np.ndarray
    prev_token: int
    last_frame: int


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    max_v = np.max(values)
    shifted = values - max_v
    denom = np.log(np.sum(np.exp(shifted)))
    return (shifted - denom).astype(np.float32, copy=False)


def _array_dtype_to_numpy(data_type_code: int):
    from coremltools.proto import FeatureTypes_pb2  # type: ignore

    mapping = {
        FeatureTypes_pb2.ArrayFeatureType.FLOAT32: np.float32,
        FeatureTypes_pb2.ArrayFeatureType.FLOAT16: np.float16,
        FeatureTypes_pb2.ArrayFeatureType.DOUBLE: np.float64,
        FeatureTypes_pb2.ArrayFeatureType.INT32: np.int32,
    }
    return mapping.get(data_type_code, np.float32)


def _get_input_specs(model) -> list[_InputSpec]:
    proto = model.get_spec()
    specs: list[_InputSpec] = []
    for entry in proto.description.input:
        if not entry.type.HasField("multiArrayType"):
            continue
        arr = entry.type.multiArrayType
        specs.append(
            _InputSpec(
                name=entry.name,
                shape=[int(dim) for dim in arr.shape],
                dtype=_array_dtype_to_numpy(arr.dataType),
            )
        )
    return specs


def _get_state_names(model) -> list[str]:
    proto = model.get_spec()
    return [entry.name for entry in proto.description.state]


class ParakeetCoreMLTDT:
    def __init__(
        self,
        model_dir: Path,
        encoder_model_name: str = "encoder-model-int4.mlpackage",
        decoder_model_name: str = "decoder_joint-model-int4.mlpackage",
        encoder_manifest_name: str = "encoder-model-manifest.json",
        decoder_manifest_name: str = "decoder_joint-model-manifest.json",
        vocab_name: str = "vocab.txt",
        sample_rate: int = 16_000,
    ) -> None:
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        _ensure_tmpdir(model_dir)

        required_paths = [
            model_dir / encoder_model_name,
            model_dir / decoder_model_name,
            model_dir / encoder_manifest_name,
            model_dir / decoder_manifest_name,
            model_dir / vocab_name,
        ]
        missing = [str(p) for p in required_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Parakeet CoreML artifacts. "
                "Set PARAKEET_COREML_MODEL_DIR to the correct artifact directory. "
                f"Missing: {missing}"
            )

        import coremltools as ct  # type: ignore

        compute = ct.ComputeUnit.CPU_AND_NE
        self.encoder = ct.models.MLModel(str(model_dir / encoder_model_name), compute_units=compute)
        self.decoder = ct.models.MLModel(str(model_dir / decoder_model_name), compute_units=compute)

        self.encoder_manifest = json.loads((model_dir / encoder_manifest_name).read_text(encoding="utf-8"))
        self.decoder_manifest = json.loads((model_dir / decoder_manifest_name).read_text(encoding="utf-8"))
        self.vocab = (model_dir / vocab_name).read_text(encoding="utf-8").splitlines()

        self.encoder_input_specs = _get_input_specs(self.encoder)
        self.decoder_input_specs = _get_input_specs(self.decoder)
        self.encoder_inputs_by_name = {s.name: s for s in self.encoder_input_specs}
        self.decoder_inputs_by_name = {s.name: s for s in self.decoder_input_specs}

        # TDT setup: vocab + blank (+ duration bins in TDT head)
        self.blank_id = len(self.vocab)
        durations_env = os.environ.get("PARAKEET_TDT_DURATIONS", "").strip()
        if durations_env:
            parsed_durations = [int(piece.strip()) for piece in durations_env.split(",") if piece.strip()]
            if not parsed_durations:
                raise ValueError("PARAKEET_TDT_DURATIONS is set but empty after parsing.")
            self.duration_values = parsed_durations
        else:
            self.duration_values = [0, 1, 2, 3, 4]

        self.max_symbols_per_step = max(1, int(os.environ.get("PARAKEET_TDT_MAX_SYMBOLS_PER_STEP", "10")))
        self.beam_width = max(1, int(os.environ.get("PARAKEET_TDT_BEAM_WIDTH", "1")))
        self.duration_beam_width = max(
            1,
            int(os.environ.get("PARAKEET_TDT_DURATION_BEAM_WIDTH", str(self.beam_width))),
        )
        self.max_tokens_per_chunk = max(0, int(os.environ.get("PARAKEET_TDT_MAX_TOKENS_PER_CHUNK", "0")))
        self.encoder_left_context_frames = max(
            0,
            int(os.environ.get("PARAKEET_ENCODER_LEFT_CONTEXT_FRAMES", "0")),
        )
        self.encoder_right_context_frames = max(
            0,
            int(os.environ.get("PARAKEET_ENCODER_RIGHT_CONTEXT_FRAMES", "0")),
        )

        self._zero_duration_idx: int | None = None
        self._min_non_zero_duration_idx: int | None = None
        for idx, duration in enumerate(self.duration_values):
            if duration == 0 and self._zero_duration_idx is None:
                self._zero_duration_idx = idx
            if duration > 0 and (
                self._min_non_zero_duration_idx is None
                or duration < self.duration_values[self._min_non_zero_duration_idx]
            ):
                self._min_non_zero_duration_idx = idx

        self._decoder_encoder_input_name = self._pick_decoder_encoder_input_name()
        self._decoder_logits_output_name = None
        self._decoder_state_map: dict[str, str] = {}
        self._state_input_names = sorted([name for name in self.decoder_inputs_by_name if name.startswith("input_states")])
        self._decoder_state_names = _get_state_names(self.decoder)
        self._decoder_has_runtime_state = len(self._decoder_state_names) > 0
        self._decoder_uses_explicit_state_inputs = len(self._state_input_names) >= 2
        self._decoder_runtime_state = None
        if not self._decoder_uses_explicit_state_inputs and not self._decoder_has_runtime_state:
            raise RuntimeError(
                "Decoder model must expose explicit state inputs (input_states_*) "
                "or CoreML runtime states (description.state)."
            )
        if self._decoder_has_runtime_state and not self._decoder_uses_explicit_state_inputs and self.beam_width > 1:
            print(
                "warning: stateful decoder runtime does not support beam branching in this script; "
                "forcing PARAKEET_TDT_BEAM_WIDTH=1",
            )
            self.beam_width = 1
            self.duration_beam_width = 1

        self._encoder_audio_input_name = "audio_signal" if "audio_signal" in self.encoder_inputs_by_name else next(
            iter(self.encoder_inputs_by_name.keys())
        )
        self._encoder_length_input_name = "length" if "length" in self.encoder_inputs_by_name else None
        self._encoder_frame_count = int(self.encoder_inputs_by_name[self._encoder_audio_input_name].shape[-1])
        self._nemo_preprocessor = self._build_nemo_preprocessor()
        self._trace_path = os.environ.get("PARAKEET_PY_DECODER_TRACE_PATH", "").strip()
        self._trace_max_events = max(1, int(os.environ.get("PARAKEET_PY_DECODER_TRACE_MAX_EVENTS", "200000")))
        self._trace_events_written = 0
        self._trace_chunk_index = 0
        self._trace_handle = None
        if self._trace_path:
            trace_file = Path(self._trace_path).expanduser()
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_reset = os.environ.get("PARAKEET_PY_DECODER_TRACE_RESET", "1") != "0"
            if trace_reset and trace_file.exists():
                trace_file.unlink()
            self._trace_handle = trace_file.open("a", encoding="utf-8")

    def _trace_decoder_event(self, event: dict[str, Any]) -> None:
        if self._trace_handle is None:
            return
        if self._trace_events_written >= self._trace_max_events:
            return
        try:
            self._trace_handle.write(json.dumps(event, ensure_ascii=True) + "\n")
            self._trace_handle.flush()
            self._trace_events_written += 1
        except Exception:
            pass

    def _build_nemo_preprocessor(self):
        try:
            import torch
            from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

            pre = AudioToMelSpectrogramPreprocessor(
                sample_rate=self.sample_rate,
                normalize="per_feature",
                window_size=0.025,
                window_stride=0.01,
                window="hann",
                features=128,
                n_fft=512,
                log=True,
                frame_splicing=1,
                dither=1e-5,
                pad_to=0,
                pad_value=0.0,
            )
            pre.eval()
            # keep this fully CPU-side to avoid extra runtime dependencies.
            pre.to(torch.device("cpu"))
            return pre
        except Exception:
            return None

    def _pick_decoder_encoder_input_name(self) -> str:
        if "encoder_outputs" in self.decoder_inputs_by_name:
            return "encoder_outputs"
        suggested = self.decoder_manifest.get("coreml_suggestions", {}).get("audio_input_name")
        if suggested in self.decoder_inputs_by_name:
            return suggested
        for name in self.decoder_inputs_by_name:
            lname = name.lower()
            if "encoder" in lname and "output" in lname:
                return name
        raise RuntimeError("Could not determine decoder encoder-output input name.")

    def _pick_encoder_outputs(self, outputs: dict[str, Any]) -> tuple[np.ndarray, int | None]:
        best_name = None
        best_score = None
        length_value: int | None = None
        for name, value in outputs.items():
            arr = np.asarray(value)
            if arr.dtype.kind in ("i", "u") and arr.ndim == 1 and arr.size == 1:
                length_value = int(arr.reshape(-1)[0])
                continue
            if arr.dtype.kind not in ("f", "i"):
                continue
            if arr.ndim < 2:
                continue
            score = (arr.ndim, arr.shape[-1], arr.size)
            if best_score is None or score > best_score:
                best_score = score
                best_name = name
        if best_name is None:
            raise RuntimeError("Could not infer encoder output tensor.")
        return np.asarray(outputs[best_name]), length_value

    def _prepare_decoder_encoder_tensor(self, encoder_tensor: np.ndarray) -> np.ndarray:
        target = self.decoder_inputs_by_name[self._decoder_encoder_input_name]
        target_shape = list(target.shape)
        tensor = np.asarray(encoder_tensor, dtype=target.dtype)
        if list(tensor.shape) == target_shape:
            return tensor
        if tensor.ndim != len(target_shape):
            raise RuntimeError(f"Encoder rank mismatch: got {list(tensor.shape)} expected {target_shape}")
        adjusted = np.zeros(target_shape, dtype=target.dtype)
        slices = tuple(slice(0, min(tensor.shape[i], target_shape[i])) for i in range(tensor.ndim))
        adjusted[slices] = tensor[slices]
        return adjusted

    def _extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = self._prepare_audio(audio, sr)
        return self._extract_features_prepared(audio)

    def _prepare_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        arr = np.asarray(audio)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        arr = np.asarray(arr, dtype=np.float32)
        if sr != self.sample_rate:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sample_rate)
        if arr.size == 0:
            arr = np.zeros(1600, dtype=np.float32)
        return np.asarray(arr, dtype=np.float32)

    def _extract_features_prepared(self, audio: np.ndarray) -> np.ndarray:
        import librosa

        if self._nemo_preprocessor is not None:
            try:
                import torch

                signal = torch.from_numpy(audio).float().unsqueeze(0)
                signal_length = torch.tensor([signal.shape[-1]], dtype=torch.long)
                with torch.no_grad():
                    features, _ = self._nemo_preprocessor(input_signal=signal, input_signal_length=signal_length)
                return features.cpu().numpy().astype(np.float32, copy=False)
            except Exception:
                pass

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=128,
            power=2.0,
        ).astype(np.float32)
        log_mel = np.log(np.maximum(mel, 1e-6))
        mean = np.mean(log_mel, axis=1, keepdims=True)
        std = np.std(log_mel, axis=1, keepdims=True) + 1e-5
        norm = (log_mel - mean) / std
        return norm[np.newaxis, :, :].astype(np.float32)

    def _slice_or_pad_features(self, features: np.ndarray, start_frame: int) -> tuple[np.ndarray, int]:
        chunk = features[:, :, start_frame : start_frame + self._encoder_frame_count]
        actual_frames = int(chunk.shape[-1])
        if actual_frames == self._encoder_frame_count:
            return chunk.astype(np.float32, copy=False), actual_frames
        padded = np.zeros((features.shape[0], features.shape[1], self._encoder_frame_count), dtype=np.float32)
        if actual_frames > 0:
            padded[:, :, :actual_frames] = chunk
        return padded, actual_frames

    def _extract_step_logits(self, logits_tensor: np.ndarray, t_index: int) -> np.ndarray:
        arr = np.asarray(logits_tensor)
        if arr.ndim == 4:
            arr = arr[0]  # [T, U, V] or [U, T, V]
        if arr.ndim == 3:
            vocab_axis = int(np.argmax(arr.shape))
            if vocab_axis != 2:
                arr = np.moveaxis(arr, vocab_axis, 2)
            if arr.shape[0] == 1:
                arr = arr[0]  # [T, V]
            elif arr.shape[1] == 1:
                arr = arr[:, 0, :]  # [T, V]
            else:
                arr = arr[:, 0, :]  # fallback
        if arr.ndim == 2:
            return arr[min(t_index, arr.shape[0] - 1)]
        if arr.ndim == 1:
            return arr
        raise RuntimeError(f"Unsupported logits rank: {arr.ndim}")

    def _infer_decoder_output_roles(self, decoder_outputs: dict[str, Any], decoder_feed: dict[str, Any]) -> None:
        if self._decoder_logits_output_name is None:
            suggestions = self.decoder_manifest.get("coreml_suggestions", {})
            candidate = suggestions.get("logits_output_name")
            if candidate not in decoder_outputs:
                candidate = None
                best_vocab = -1
                for name, value in decoder_outputs.items():
                    arr = np.asarray(value)
                    if arr.dtype.kind != "f" or arr.ndim < 2:
                        continue
                    vocab = int(arr.shape[-1])
                    if vocab > best_vocab:
                        best_vocab = vocab
                        candidate = name
            if candidate is None:
                raise RuntimeError("Could not infer decoder logits output.")
            self._decoder_logits_output_name = candidate

        if not self._decoder_uses_explicit_state_inputs:
            return

        if not self._decoder_state_map:
            state_inputs = sorted([name for name in decoder_feed if name.startswith("input_states")])
            candidates = []
            for out_name, out_value in decoder_outputs.items():
                if out_name == self._decoder_logits_output_name:
                    continue
                out_arr = np.asarray(out_value)
                if out_arr.dtype.kind != "f" or out_arr.ndim != 3:
                    continue
                candidates.append((out_name, out_arr.shape))

            used: set[str] = set()
            for in_name in state_inputs:
                in_shape = tuple(np.asarray(decoder_feed[in_name]).shape)
                match_name = None
                for out_name, out_shape in candidates:
                    if out_name in used:
                        continue
                    if tuple(out_shape) == in_shape:
                        match_name = out_name
                        break
                if match_name is None:
                    raise RuntimeError(f"Could not map decoder state input '{in_name}' to any decoder output.")
                used.add(match_name)
                self._decoder_state_map[in_name] = match_name

    def _chunk_max_tokens(self, enc_steps: int) -> int:
        if self.max_tokens_per_chunk > 0:
            return self.max_tokens_per_chunk
        return max(256, enc_steps * 4)

    def _topk_indices(self, values: np.ndarray, k: int) -> list[int]:
        arr = np.asarray(values).reshape(-1)
        if arr.size == 0 or k <= 0:
            return []
        count = min(k, int(arr.size))
        if count == int(arr.size):
            order = np.argsort(arr)[::-1]
        else:
            part = np.argpartition(arr, -count)[-count:]
            order = part[np.argsort(arr[part])[::-1]]
        return [int(idx) for idx in order]

    def _duration_from_index(self, duration_idx: int) -> int:
        if 0 <= duration_idx < len(self.duration_values):
            return int(self.duration_values[duration_idx])
        return 1

    def _decoder_step(
        self,
        decoder_encoder_input: np.ndarray,
        t_idx: int,
        prev_token: int,
        state1: np.ndarray | None,
        state2: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        decoder_feed: dict[str, Any] = {
            self._decoder_encoder_input_name: decoder_encoder_input,
            "targets": np.array(
                [[prev_token]],
                dtype=self.decoder_inputs_by_name.get("targets", _InputSpec("targets", [1, 1], np.int32)).dtype,
            ),
        }
        if "target_length" in self.decoder_inputs_by_name:
            decoder_feed["target_length"] = np.array(
                [1],
                dtype=self.decoder_inputs_by_name["target_length"].dtype,
            )
        if "state_update_gate" in self.decoder_inputs_by_name:
            decoder_feed["state_update_gate"] = np.array(
                [1.0],
                dtype=self.decoder_inputs_by_name["state_update_gate"].dtype,
            )

        if self._decoder_uses_explicit_state_inputs:
            if state1 is None or state2 is None:
                raise RuntimeError("Explicit decoder state inputs require non-null state tensors.")
            state_name_1, state_name_2 = self._state_input_names[:2]
            decoder_feed[state_name_1] = np.asarray(state1, dtype=np.float32)
            decoder_feed[state_name_2] = np.asarray(state2, dtype=np.float32)

        if self._decoder_has_runtime_state and not self._decoder_uses_explicit_state_inputs:
            if self._decoder_runtime_state is None:
                self._decoder_runtime_state = self.decoder.make_state()
            decoder_outputs = self.decoder.predict(decoder_feed, state=self._decoder_runtime_state)
        else:
            decoder_outputs = self.decoder.predict(decoder_feed)
        self._infer_decoder_output_roles(decoder_outputs, decoder_feed)

        logits = np.asarray(decoder_outputs[self._decoder_logits_output_name])  # type: ignore[index]
        step_logits = self._extract_step_logits(logits, t_idx)

        token_vocab_size = self.blank_id + 1
        token_logits = np.asarray(step_logits[:token_vocab_size], dtype=np.float32)
        duration_logits = np.asarray(step_logits[token_vocab_size:], dtype=np.float32)

        token_logp = _log_softmax(token_logits)
        duration_logp = _log_softmax(duration_logits) if duration_logits.size > 0 else np.zeros(1, dtype=np.float32)

        if not self._decoder_uses_explicit_state_inputs:
            return token_logp, duration_logp, state1, state2

        state_name_1, state_name_2 = self._state_input_names[:2]
        out_state_name_1 = self._decoder_state_map.get(state_name_1)
        out_state_name_2 = self._decoder_state_map.get(state_name_2)
        if out_state_name_1 is None or out_state_name_2 is None:
            raise RuntimeError("Failed to resolve decoder output state mappings.")

        next_state1 = np.asarray(decoder_outputs[out_state_name_1], dtype=np.float32)
        next_state2 = np.asarray(decoder_outputs[out_state_name_2], dtype=np.float32)
        return token_logp, duration_logp, next_state1, next_state2

    def _merge_beam_duplicates(self, hyps: list[_BeamHypothesis], beam: int) -> list[_BeamHypothesis]:
        dedup: dict[tuple[tuple[int, ...], int, int], _BeamHypothesis] = {}
        for hyp in hyps:
            key = (tuple(hyp.tokens), hyp.last_frame, hyp.prev_token)
            best = dedup.get(key)
            if best is None or hyp.score > best.score:
                dedup[key] = hyp
        merged = sorted(dedup.values(), key=lambda h: h.score, reverse=True)
        return merged[: max(beam * 2, beam)]

    def _decode_chunk_greedy(
        self,
        decoder_encoder_input: np.ndarray,
        enc_steps: int,
        state1: np.ndarray | None,
        state2: np.ndarray | None,
        prev_token: int,
        *,
        chunk_index: int,
    ) -> tuple[list[int], np.ndarray | None, np.ndarray | None, int]:
        t = 0
        max_tokens = self._chunk_max_tokens(enc_steps)
        output_token_ids: list[int] = []
        step_index = 0

        while t < enc_steps and len(output_token_ids) < max_tokens:
            symbols_added = 0
            need_loop = True
            skip = 1

            while need_loop and symbols_added < self.max_symbols_per_step and len(output_token_ids) < max_tokens:
                t_before = int(t)
                prev_before = int(prev_token)
                token_logp, duration_logp, next_state1, next_state2 = self._decoder_step(
                    decoder_encoder_input=decoder_encoder_input,
                    t_idx=t,
                    prev_token=prev_token,
                    state1=state1,
                    state2=state2,
                )

                token_id = int(np.argmax(token_logp))
                duration_idx = int(np.argmax(duration_logp))
                skip = self._duration_from_index(duration_idx)

                # Match NeMo loop-label behavior for TDT:
                # blank tokens should not advance by zero duration.
                if token_id == self.blank_id and skip == 0:
                    if self._min_non_zero_duration_idx is not None:
                        duration_idx = self._min_non_zero_duration_idx
                        skip = self._duration_from_index(duration_idx)
                    if skip == 0:
                        skip = 1

                if token_id != self.blank_id:
                    output_token_ids.append(token_id)
                    state1 = next_state1
                    state2 = next_state2
                    prev_token = token_id

                self._trace_decoder_event(
                    {
                        "source": "python",
                        "kind": "step",
                        "chunk_index": int(chunk_index),
                        "step_index": int(step_index),
                        "encoder_steps": int(enc_steps),
                        "t": int(t_before),
                        "token_id": int(token_id),
                        "duration_idx": int(duration_idx),
                        "skip": int(skip),
                        "prev_token": int(prev_before),
                        "emitted": bool(token_id != self.blank_id),
                        "commit_state": 1,
                    }
                )
                step_index += 1

                symbols_added += 1
                t += skip
                need_loop = skip == 0

            if symbols_added >= self.max_symbols_per_step:
                t += 1

        if state1 is not None:
            state1 = np.asarray(state1, dtype=np.float32)
        if state2 is not None:
            state2 = np.asarray(state2, dtype=np.float32)
        return output_token_ids, state1, state2, prev_token

    def _decode_chunk_beam(
        self,
        decoder_encoder_input: np.ndarray,
        enc_steps: int,
        state1: np.ndarray | None,
        state2: np.ndarray | None,
        prev_token: int,
        *,
        chunk_index: int,
    ) -> tuple[list[int], np.ndarray | None, np.ndarray | None, int]:
        if not self._decoder_uses_explicit_state_inputs:
            raise RuntimeError("Beam decoding is only supported with explicit decoder state inputs.")
        if state1 is None or state2 is None:
            raise RuntimeError("Beam decoding requires non-null decoder state tensors.")
        self._trace_decoder_event(
            {
                "source": "python",
                "kind": "beam_chunk",
                "chunk_index": int(chunk_index),
                "encoder_steps": int(enc_steps),
            }
        )
        beam = max(1, self.beam_width)
        token_beam_k = max(1, min(beam, self.blank_id))
        duration_beam_k = max(1, min(self.duration_beam_width, len(self.duration_values)))
        max_tokens = self._chunk_max_tokens(enc_steps)

        kept_hyps: list[_BeamHypothesis] = [
            _BeamHypothesis(
                score=0.0,
                tokens=[],
                state1=np.asarray(state1, dtype=np.float32),
                state2=np.asarray(state2, dtype=np.float32),
                prev_token=prev_token,
                last_frame=0,
            )
        ]

        for time_idx in range(enc_steps):
            hyps = [hyp for hyp in kept_hyps if hyp.last_frame == time_idx]
            kept_hyps = [hyp for hyp in kept_hyps if hyp.last_frame > time_idx]

            while hyps:
                max_hyp = max(hyps, key=lambda h: h.score)
                hyps.remove(max_hyp)

                token_logp, duration_logp, next_state1, next_state2 = self._decoder_step(
                    decoder_encoder_input=decoder_encoder_input,
                    t_idx=time_idx,
                    prev_token=max_hyp.prev_token,
                    state1=max_hyp.state1,
                    state2=max_hyp.state2,
                )

                token_topk = self._topk_indices(token_logp[: self.blank_id], token_beam_k)
                duration_topk = self._topk_indices(duration_logp, duration_beam_k)
                if not duration_topk:
                    duration_topk = [0]

                non_blank_candidates: list[tuple[float, int, int]] = []
                for d_idx in duration_topk:
                    d_score = float(duration_logp[d_idx])
                    for token_idx in token_topk:
                        score = float(token_logp[token_idx]) + d_score
                        non_blank_candidates.append((score, token_idx, d_idx))
                non_blank_candidates.sort(key=lambda item: item[0], reverse=True)

                for score, token_idx, duration_idx in non_blank_candidates[:token_beam_k]:
                    if len(max_hyp.tokens) >= max_tokens:
                        continue
                    duration = self._duration_from_index(duration_idx)
                    new_hyp = _BeamHypothesis(
                        score=max_hyp.score + score,
                        tokens=max_hyp.tokens + [token_idx],
                        state1=np.asarray(next_state1, dtype=np.float32),
                        state2=np.asarray(next_state2, dtype=np.float32),
                        prev_token=token_idx,
                        last_frame=max_hyp.last_frame + duration,
                    )
                    if duration == 0:
                        hyps.append(new_hyp)
                    else:
                        kept_hyps.append(new_hyp)

                for duration_idx in duration_topk:
                    duration = self._duration_from_index(duration_idx)
                    if duration == 0:
                        if self._zero_duration_idx is not None and duration_idx == self._zero_duration_idx:
                            if len(duration_topk) == 1 and self._min_non_zero_duration_idx is not None:
                                duration_idx = self._min_non_zero_duration_idx
                                duration = self._duration_from_index(duration_idx)
                            else:
                                continue
                        else:
                            continue

                    blank_score = float(token_logp[self.blank_id]) + float(duration_logp[duration_idx])
                    kept_hyps.append(
                        _BeamHypothesis(
                            score=max_hyp.score + blank_score,
                            tokens=list(max_hyp.tokens),
                            state1=max_hyp.state1,
                            state2=max_hyp.state2,
                            prev_token=max_hyp.prev_token,
                            last_frame=max_hyp.last_frame + duration,
                        )
                    )

                kept_hyps = self._merge_beam_duplicates(kept_hyps, beam)

                if hyps:
                    hyps_max = max(h.score for h in hyps)
                    kept_most_prob = [hyp for hyp in kept_hyps if hyp.score > hyps_max]
                    if len(kept_most_prob) >= beam:
                        kept_hyps = sorted(kept_most_prob, key=lambda h: h.score, reverse=True)[:beam]
                        break
                else:
                    kept_hyps = sorted(kept_hyps, key=lambda h: h.score, reverse=True)[:beam]

        if not kept_hyps:
            return [], np.asarray(state1, dtype=np.float32), np.asarray(state2, dtype=np.float32), prev_token

        best_hyp = max(kept_hyps, key=lambda h: h.score)
        return (
            list(best_hyp.tokens),
            np.asarray(best_hyp.state1, dtype=np.float32),
            np.asarray(best_hyp.state2, dtype=np.float32),
            int(best_hyp.prev_token),
        )

    def _decode_chunk(
        self,
        encoder_tensor: np.ndarray,
        enc_steps: int,
        state1: np.ndarray | None,
        state2: np.ndarray | None,
        prev_token: int,
    ) -> tuple[list[int], np.ndarray | None, np.ndarray | None, int]:
        enc_steps = max(0, min(enc_steps, int(self.decoder_inputs_by_name[self._decoder_encoder_input_name].shape[-1])))
        if enc_steps == 0:
            return [], state1, state2, int(prev_token)

        chunk_index = int(self._trace_chunk_index)
        self._trace_chunk_index += 1
        decoder_encoder_input = self._prepare_decoder_encoder_tensor(encoder_tensor)
        if self.beam_width > 1:
            return self._decode_chunk_beam(
                decoder_encoder_input,
                enc_steps,
                state1,
                state2,
                prev_token,
                chunk_index=chunk_index,
            )
        return self._decode_chunk_greedy(
            decoder_encoder_input,
            enc_steps,
            state1,
            state2,
            prev_token,
            chunk_index=chunk_index,
        )

    def _transcribe_features(self, features: np.ndarray, allow_right_context: bool = True) -> str:
        total_frames = int(features.shape[-1])
        if total_frames <= 0:
            return ""

        state1: np.ndarray | None
        state2: np.ndarray | None
        if self._decoder_uses_explicit_state_inputs:
            state1_shape = self.decoder_inputs_by_name[self._state_input_names[0]].shape
            state2_shape = self.decoder_inputs_by_name[self._state_input_names[1]].shape
            state1 = np.zeros(state1_shape, dtype=np.float32)
            state2 = np.zeros(state2_shape, dtype=np.float32)
            self._decoder_runtime_state = None
        else:
            state1 = None
            state2 = None
            self._decoder_runtime_state = self.decoder.make_state() if self._decoder_has_runtime_state else None
        prev_token = self.blank_id

        max_context_total = max(0, self._encoder_frame_count - 1)
        left_ctx = min(self.encoder_left_context_frames, max_context_total)
        configured_right_ctx = self.encoder_right_context_frames if allow_right_context else 0
        right_ctx = min(configured_right_ctx, max(0, max_context_total - left_ctx))
        hop_frames = max(1, self._encoder_frame_count - left_ctx - right_ctx)

        all_tokens: list[int] = []
        for center_start in range(0, total_frames, hop_frames):
            # Optional left/right context to reduce encoder boundary loss on long-form audio.
            input_start = max(0, center_start - left_ctx)
            feat_chunk, actual_frames = self._slice_or_pad_features(features, input_start)
            if actual_frames <= 0:
                continue

            encoder_feed: dict[str, Any] = {self._encoder_audio_input_name: feat_chunk.astype(np.float32, copy=False)}
            if self._encoder_length_input_name:
                length_dtype = self.encoder_inputs_by_name[self._encoder_length_input_name].dtype
                encoder_feed[self._encoder_length_input_name] = np.array([actual_frames], dtype=length_dtype)

            encoder_outputs = self.encoder.predict(encoder_feed)
            encoder_tensor, encoder_len = self._pick_encoder_outputs(encoder_outputs)
            if encoder_len is None:
                encoder_len = int(encoder_tensor.shape[-1])

            # Decode only the non-overlapping center span; keep context as encoder-only context.
            if (left_ctx > 0 or right_ctx > 0) and encoder_len > 0 and actual_frames > 0:
                left_in = center_start - input_start
                center_in = min(hop_frames, total_frames - center_start)

                scale = float(encoder_len) / float(actual_frames)
                left_out = int(round(left_in * scale))
                center_out = int(round(center_in * scale))

                left_out = max(0, min(left_out, encoder_len))
                if center_start + center_in >= total_frames:
                    end_out = encoder_len
                else:
                    end_out = max(left_out, min(encoder_len, left_out + max(1, center_out)))

                encoder_tensor = np.asarray(encoder_tensor)[..., left_out:end_out]
                encoder_len = int(max(0, end_out - left_out))

            chunk_tokens, state1, state2, prev_token = self._decode_chunk(
                encoder_tensor=encoder_tensor,
                enc_steps=encoder_len,
                state1=state1,
                state2=state2,
                prev_token=prev_token,
            )
            all_tokens.extend(chunk_tokens)

        pieces = [self.vocab[idx] for idx in all_tokens if 0 <= idx < len(self.vocab)]
        return _text_from_sentencepiece_pieces(pieces)

    def transcribe_file(self, audio_path: str) -> str:
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        prepared_audio = self._prepare_audio(np.asarray(audio, dtype=np.float32), int(sr))

        segment_sec = float(os.environ.get("PARAKEET_LONGFORM_SEGMENT_SEC", "0"))
        overlap_sec = float(os.environ.get("PARAKEET_LONGFORM_OVERLAP_SEC", "0"))
        if segment_sec <= 0:
            features = self._extract_features_prepared(prepared_audio)
            return self._transcribe_features(features, allow_right_context=True)

        segment_samples = max(1, int(round(segment_sec * self.sample_rate)))
        overlap_samples = int(round(max(0.0, overlap_sec) * self.sample_rate))
        if overlap_samples >= segment_samples:
            overlap_samples = max(0, segment_samples // 4)
        step_samples = max(1, segment_samples - overlap_samples)

        chunk_texts: list[str] = []
        start = 0
        total = int(prepared_audio.shape[0])
        while start < total:
            end = min(total, start + segment_samples)
            segment_audio = prepared_audio[start:end]
            features = self._extract_features_prepared(segment_audio)
            chunk_text = self._transcribe_features(features, allow_right_context=True)
            if chunk_text:
                chunk_texts.append(chunk_text)
            if end >= total:
                break
            start += step_samples

        return _merge_text_by_overlap(chunk_texts)

    def stream_transcribe_file(
        self,
        audio_path: str,
        stream_step_sec: float | None = None,
        agreement_window: int | None = None,
    ) -> dict[str, Any]:
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        features = self._extract_features(np.asarray(audio, dtype=np.float32), int(sr))

        total_frames = int(features.shape[-1])
        if total_frames <= 0:
            return {
                "transcript": "",
                "audio_cursor": [],
                "interim_results": [],
                "confirmed_audio_cursor": [],
                "confirmed_interim_results": [],
                "model_timestamps_hypothesis": None,
                "model_timestamps_confirmed": None,
            }

        if stream_step_sec is None:
            stream_step_sec = float(
                os.environ.get("PARAKEET_STREAM_STEP_SEC")
                or os.environ.get("PARAKEET_STREAM_REALTIME_RESOLUTION")
                or "0.25"
            )
        stream_step_sec = max(0.05, float(stream_step_sec))

        if agreement_window is None:
            agreement_window = int(os.environ.get("PARAKEET_STREAM_AGREEMENT_WINDOW", "2"))
        agreement_window = max(1, int(agreement_window))

        frame_hop_sec = 0.01
        step_frames = max(1, int(round(stream_step_sec / frame_hop_sec)))
        total_audio_sec = float(total_frames * frame_hop_sec)
        allow_right_ctx_stream = os.environ.get("PARAKEET_STREAM_ALLOW_RIGHT_CONTEXT", "0") == "1"

        interim_results: list[str] = []
        audio_cursor: list[float] = []
        confirmed_interim_results: list[str] = []
        confirmed_audio_cursor: list[float] = []
        rolling_hypotheses: list[list[str]] = []
        confirmed_tokens: list[str] = []

        end_frame = 0
        while end_frame < total_frames:
            end_frame = min(total_frames, end_frame + step_frames)
            prefix_features = features[:, :, :end_frame]
            # Streaming path stays causal by default (no right-context lookahead).
            hypothesis_text = self._transcribe_features(prefix_features, allow_right_context=allow_right_ctx_stream)
            hypothesis_tokens = hypothesis_text.split()
            rolling_hypotheses.append(hypothesis_tokens)
            if len(rolling_hypotheses) > agreement_window:
                rolling_hypotheses = rolling_hypotheses[-agreement_window:]

            cursor_sec = min(total_audio_sec, float(end_frame * frame_hop_sec))
            if hypothesis_text and (not interim_results or interim_results[-1] != hypothesis_text):
                interim_results.append(hypothesis_text)
                audio_cursor.append(cursor_sec)

            candidate_tokens = _longest_common_prefix_tokens(rolling_hypotheses)
            # Keep confirmed text monotonic.
            if len(candidate_tokens) < len(confirmed_tokens):
                candidate_tokens = confirmed_tokens
            elif confirmed_tokens and candidate_tokens[: len(confirmed_tokens)] != confirmed_tokens:
                candidate_tokens = confirmed_tokens

            candidate_text = " ".join(candidate_tokens).strip()
            current_confirmed = " ".join(confirmed_tokens).strip()
            if candidate_text and candidate_text != current_confirmed:
                confirmed_tokens = candidate_tokens
                confirmed_interim_results.append(candidate_text)
                confirmed_audio_cursor.append(cursor_sec)

        final_text = (
            interim_results[-1]
            if interim_results
            else self._transcribe_features(features, allow_right_context=allow_right_ctx_stream)
        )
        if final_text:
            if not confirmed_interim_results or confirmed_interim_results[-1] != final_text:
                confirmed_interim_results.append(final_text)
                confirmed_audio_cursor.append(total_audio_sec)

        return {
            "transcript": final_text,
            "audio_cursor": audio_cursor,
            "interim_results": interim_results,
            "confirmed_audio_cursor": confirmed_audio_cursor,
            "confirmed_interim_results": confirmed_interim_results,
            "model_timestamps_hypothesis": None,
            "model_timestamps_confirmed": None,
        }


_ENGINE: ParakeetCoreMLTDT | None = None


def _resolve_model_dir() -> Path:
    env_dir = os.environ.get("PARAKEET_COREML_MODEL_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "artifacts/parakeet-tdt-0.6b-v2").resolve()


def _resolve_model_names() -> tuple[str, str]:
    suffix = os.environ.get("PARAKEET_COREML_MODEL_SUFFIX", "int4")
    encoder_name = os.environ.get("PARAKEET_COREML_ENCODER_MODEL", f"encoder-model-{suffix}.mlpackage")
    decoder_name = os.environ.get("PARAKEET_COREML_DECODER_MODEL", f"decoder_joint-model-{suffix}.mlpackage")
    return encoder_name, decoder_name


def _get_engine() -> ParakeetCoreMLTDT:
    global _ENGINE
    if _ENGINE is None:
        model_dir = _resolve_model_dir()
        encoder_name, decoder_name = _resolve_model_names()
        _ENGINE = ParakeetCoreMLTDT(
            model_dir=model_dir,
            encoder_model_name=encoder_name,
            decoder_model_name=decoder_name,
        )
    return _ENGINE


def transcribe_file(audio_path: str, language: str | None = None, keywords: list[str] | None = None) -> str:
    # language/keywords are accepted for OpenBench interface compatibility.
    del language, keywords
    return _get_engine().transcribe_file(audio_path)


def stream_transcribe_file(
    audio_path: str,
    language: str | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    # language/keywords are accepted for OpenBench interface compatibility.
    del language, keywords
    return _get_engine().stream_transcribe_file(audio_path)


def warmup() -> None:
    engine = _get_engine()
    import soundfile as sf
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        silence = np.zeros(engine.sample_rate // 2, dtype=np.float32)
        sf.write(path, silence, engine.sample_rate)
        _ = engine.transcribe_file(path)
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parakeet CoreML TDT transcriber.")
    parser.add_argument("--audio", type=Path, required=True, help="Audio path")
    args = parser.parse_args()
    print(transcribe_file(str(args.audio)))


if __name__ == "__main__":
    main()
