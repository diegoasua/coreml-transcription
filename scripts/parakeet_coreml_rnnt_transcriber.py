#!/usr/bin/env python3
"""Local Parakeet v2 CoreML RNNT/TDT transcriber.

Exports:
  transcribe_file(audio_path, language=None, keywords=None) -> str
  stream_transcribe_file(audio_path, language=None, keywords=None) -> dict

Can also run as CLI:
  python scripts/parakeet_coreml_rnnt_transcriber.py --audio path.wav
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


@dataclass
class _InputSpec:
    name: str
    shape: list[int]
    dtype: Any


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


class ParakeetCoreMLRNNT:
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

        # RNNT setup: vocab + blank (+ duration bins in TDT head)
        self.blank_id = len(self.vocab)
        self.duration_values = [0, 1, 2, 3, 4]

        self._decoder_encoder_input_name = self._pick_decoder_encoder_input_name()
        self._decoder_logits_output_name = None
        self._decoder_state_map: dict[str, str] = {}

        self._encoder_audio_input_name = "audio_signal" if "audio_signal" in self.encoder_inputs_by_name else next(
            iter(self.encoder_inputs_by_name.keys())
        )
        self._encoder_length_input_name = "length" if "length" in self.encoder_inputs_by_name else None
        self._encoder_frame_count = int(self.encoder_inputs_by_name[self._encoder_audio_input_name].shape[-1])
        self._nemo_preprocessor = self._build_nemo_preprocessor()

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
        import librosa

        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            audio = np.zeros(1600, dtype=np.float32)

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

    def _decode_chunk(
        self,
        encoder_tensor: np.ndarray,
        enc_steps: int,
        state1: np.ndarray,
        state2: np.ndarray,
        prev_token: int,
    ) -> tuple[list[int], np.ndarray, np.ndarray, int]:
        enc_steps = max(0, min(enc_steps, int(self.decoder_inputs_by_name[self._decoder_encoder_input_name].shape[-1])))
        if enc_steps == 0:
            return [], state1, state2, prev_token

        decoder_encoder_input = self._prepare_decoder_encoder_tensor(encoder_tensor)

        decoder_feed: dict[str, Any] = {
            self._decoder_encoder_input_name: decoder_encoder_input,
            "targets": np.zeros((1, 1), dtype=np.int32),
            "target_length": np.array([1], dtype=np.int32),
            "input_states_1": np.asarray(state1, dtype=np.float32),
            "input_states_2": np.asarray(state2, dtype=np.float32),
        }

        t = 0
        symbols_at_t = 0
        max_symbols_per_t = 10
        max_tokens = max(256, enc_steps * 4)
        output_token_ids: list[int] = []

        while t < enc_steps and len(output_token_ids) < max_tokens:
            decoder_feed["targets"] = np.array([[prev_token]], dtype=np.int32)
            decoder_feed["target_length"] = np.array([1], dtype=np.int32)
            decoder_outputs = self.decoder.predict(decoder_feed)
            self._infer_decoder_output_roles(decoder_outputs, decoder_feed)

            logits = np.asarray(decoder_outputs[self._decoder_logits_output_name])  # type: ignore[index]
            step_logits = self._extract_step_logits(logits, t)

            token_vocab_size = self.blank_id + 1
            token_logits = step_logits[:token_vocab_size]
            duration_logits = step_logits[token_vocab_size:]

            token_id = int(np.argmax(token_logits))
            duration = 1
            if duration_logits.size > 0:
                d_idx = int(np.argmax(duration_logits))
                if d_idx < len(self.duration_values):
                    duration = int(self.duration_values[d_idx])

            next_states = {in_name: np.asarray(decoder_outputs[out_name]) for in_name, out_name in self._decoder_state_map.items()}

            if token_id == self.blank_id:
                # In RNNT, blank advances time without consuming a label;
                # predictor state must remain unchanged.
                t += max(1, duration)
                symbols_at_t = 0
            else:
                output_token_ids.append(token_id)
                # Update predictor state only when a non-blank symbol is consumed.
                for in_name, state_value in next_states.items():
                    decoder_feed[in_name] = state_value
                prev_token = token_id
                # TDT duration may advance encoder time on label emissions.
                if duration > 0:
                    t += duration
                    symbols_at_t = 0
                    continue
                symbols_at_t += 1
                if symbols_at_t >= max_symbols_per_t:
                    t += 1
                    symbols_at_t = 0

        out_state1 = np.asarray(decoder_feed["input_states_1"], dtype=np.float32)
        out_state2 = np.asarray(decoder_feed["input_states_2"], dtype=np.float32)
        return output_token_ids, out_state1, out_state2, prev_token

    def _transcribe_features(self, features: np.ndarray) -> str:
        total_frames = int(features.shape[-1])
        if total_frames <= 0:
            return ""

        state1_shape = self.decoder_inputs_by_name["input_states_1"].shape
        state2_shape = self.decoder_inputs_by_name["input_states_2"].shape
        state1 = np.zeros(state1_shape, dtype=np.float32)
        state2 = np.zeros(state2_shape, dtype=np.float32)
        prev_token = self.blank_id

        all_tokens: list[int] = []
        for start in range(0, total_frames, self._encoder_frame_count):
            feat_chunk, actual_frames = self._slice_or_pad_features(features, start)
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
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        features = self._extract_features(np.asarray(audio, dtype=np.float32), int(sr))
        return self._transcribe_features(features)

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
            hypothesis_text = self._transcribe_features(prefix_features)
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

        final_text = interim_results[-1] if interim_results else self._transcribe_features(features)
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


_ENGINE: ParakeetCoreMLRNNT | None = None


def _resolve_model_dir() -> Path:
    env_dir = os.environ.get("PARAKEET_COREML_MODEL_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "artifacts/parakeet-tdt-0.6b-v2").resolve()


def _get_engine() -> ParakeetCoreMLRNNT:
    global _ENGINE
    if _ENGINE is None:
        model_dir = _resolve_model_dir()
        _ENGINE = ParakeetCoreMLRNNT(model_dir=model_dir)
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
    parser = argparse.ArgumentParser(description="Parakeet CoreML RNNT/TDT transcriber.")
    parser.add_argument("--audio", type=Path, required=True, help="Audio path")
    args = parser.parse_args()
    print(transcribe_file(str(args.audio)))


if __name__ == "__main__":
    main()
