#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-nvidia/parakeet-tdt-0.6b-v2}"
ARTIFACT_DIR="${2:-artifacts/parakeet-tdt-0.6b-v2}"
TARGET="${3:-macos15}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
EXPORT_FORMATS="${EXPORT_FORMATS:-onnx,ts}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  elif command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN="python3.13"
  else
    PYTHON_BIN="python3"
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: '${PYTHON_BIN}' not found. Set PYTHON_BIN or install python3."
  exit 1
fi

missing_pydeps="$("${PYTHON_BIN}" - <<'PY'
mods = ["onnx", "coremltools", "numpy", "torch"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
print(" ".join(missing))
PY
)"
if [[ -n "${missing_pydeps}" ]]; then
  echo "error: python '${PYTHON_BIN}' is missing required modules: ${missing_pydeps}"
  echo "fix:"
  echo "  1) use your conversion env interpreter, e.g. PYTHON_BIN=.venv/bin/python"
  echo "  2) or install deps into that interpreter:"
  echo "     ${PYTHON_BIN} -m pip install -r requirements-conversion.txt"
  exit 1
fi

if [[ -z "${TMPDIR:-}" ]]; then
  ARTIFACT_DIR_ABS="$(cd "${ARTIFACT_DIR}" && pwd)"
  TMPDIR="${ARTIFACT_DIR_ABS}/.tmp"
fi
if [[ -e "${TMPDIR}" && ! -d "${TMPDIR}" ]]; then
  echo "error: TMPDIR path exists but is not a directory: ${TMPDIR}"
  echo "fix: rm -f \"${TMPDIR}\" && mkdir -p \"${TMPDIR}\""
  exit 1
fi
mkdir -p "${TMPDIR}"
export TMPDIR

if [[ "${SKIP_EXPORT}" == "1" ]]; then
  echo "[1/5] Skipping export (SKIP_EXPORT=1)"
  echo "  note: if ONNX external data is missing/corrupt, rerun with SKIP_EXPORT=0"
else
  echo "[1/5] Exporting NeMo model (${EXPORT_FORMATS}): ${MODEL_NAME}"
  "${PYTHON_BIN}" scripts/export_nemo_to_onnx.py \
    --model "${MODEL_NAME}" \
    --output-dir "${ARTIFACT_DIR}" \
    --formats "${EXPORT_FORMATS}"
fi

shopt -s nullglob
onnx_files=( "${ARTIFACT_DIR}"/*.onnx )
shopt -u nullglob

if [[ ${#onnx_files[@]} -eq 0 ]]; then
  echo "error: no ONNX files found in ${ARTIFACT_DIR}. "
  echo "ONNX manifests are required for TorchScript->CoreML input specs."
  exit 1
fi

echo "[2/5] Inspecting ONNX graph(s) and writing manifest(s)"
for onnx_file in "${onnx_files[@]}"; do
  base_name="$(basename "${onnx_file}" .onnx)"
  manifest_file="${ARTIFACT_DIR}/${base_name}-manifest.json"
  echo "  - inspect ${onnx_file}"
  if ! "${PYTHON_BIN}" scripts/inspect_onnx.py \
    --onnx "${onnx_file}" \
    --write-manifest "${manifest_file}"; then
    if [[ "${SKIP_EXPORT}" == "1" ]]; then
      echo
      echo "error: failed to inspect ONNX while SKIP_EXPORT=1."
      echo "This usually means ONNX external tensor files are missing."
      echo "fix:"
      echo "  1) unset SKIP_EXPORT"
      echo "  2) rerun export with TorchScript regeneration:"
      echo "     SKIP_EXPORT=0 EXPORT_FORMATS=onnx,ts bash scripts/run_conversion.sh"
    fi
    exit 1
  fi
done

ENABLE_LOCAL_ATTENTION_ENCODER="${ENABLE_LOCAL_ATTENTION_ENCODER:-0}"
LOCAL_ATTENTION_ENCODER_BASENAME="${LOCAL_ATTENTION_ENCODER_BASENAME:-encoder-model-local-streaming}"
LOCAL_ATTENTION_LEFT_CONTEXT_STEPS="${LOCAL_ATTENTION_LEFT_CONTEXT_STEPS:-32}"
LOCAL_ATTENTION_RIGHT_CONTEXT_STEPS="${LOCAL_ATTENTION_RIGHT_CONTEXT_STEPS:-0}"
LOCAL_ATTENTION_CHUNK_STEPS="${LOCAL_ATTENTION_CHUNK_STEPS:-10}"
LOCAL_ATTENTION_SHIFT_STEPS="${LOCAL_ATTENTION_SHIFT_STEPS:-10}"
LOCAL_ATTENTION_LEFT_CHUNKS="${LOCAL_ATTENTION_LEFT_CHUNKS:-}"
if [[ "${ENABLE_LOCAL_ATTENTION_ENCODER}" == "1" ]]; then
  echo "[2b/5] Building local-attention streaming encoder TorchScript wrapper"
  local_encoder_ts="${ARTIFACT_DIR}/${LOCAL_ATTENTION_ENCODER_BASENAME}.ts"
  local_encoder_manifest="${ARTIFACT_DIR}/${LOCAL_ATTENTION_ENCODER_BASENAME}-manifest.json"
  local_encoder_streaming="${ARTIFACT_DIR}/${LOCAL_ATTENTION_ENCODER_BASENAME}-streaming.json"
  local_encoder_cmd=(
    "${PYTHON_BIN}" scripts/make_streaming_local_attention_encoder_torchscript.py
    --model "${MODEL_NAME}"
    --output-ts "${local_encoder_ts}"
    --output-manifest "${local_encoder_manifest}"
    --output-streaming-config "${local_encoder_streaming}"
    --left-context-steps "${LOCAL_ATTENTION_LEFT_CONTEXT_STEPS}"
    --right-context-steps "${LOCAL_ATTENTION_RIGHT_CONTEXT_STEPS}"
    --chunk-steps "${LOCAL_ATTENTION_CHUNK_STEPS}"
    --shift-steps "${LOCAL_ATTENTION_SHIFT_STEPS}"
  )
  if [[ -n "${LOCAL_ATTENTION_LEFT_CHUNKS}" ]]; then
    local_encoder_cmd+=( --left-chunks "${LOCAL_ATTENTION_LEFT_CHUNKS}" )
  fi
  "${local_encoder_cmd[@]}"
fi

ENABLE_LOCAL_ATTENTION_PREFIX_ENCODER="${ENABLE_LOCAL_ATTENTION_PREFIX_ENCODER:-0}"
LOCAL_ATTENTION_PREFIX_ENCODER_BASENAME="${LOCAL_ATTENTION_PREFIX_ENCODER_BASENAME:-encoder-model-local-prefix}"
LOCAL_ATTENTION_PREFIX_LEFT_CONTEXT_STEPS="${LOCAL_ATTENTION_PREFIX_LEFT_CONTEXT_STEPS:-80}"
LOCAL_ATTENTION_PREFIX_RIGHT_CONTEXT_STEPS="${LOCAL_ATTENTION_PREFIX_RIGHT_CONTEXT_STEPS:-0}"
LOCAL_ATTENTION_PREFIX_INPUT_FEATURE_FRAMES="${LOCAL_ATTENTION_PREFIX_INPUT_FEATURE_FRAMES:-300}"
if [[ "${ENABLE_LOCAL_ATTENTION_PREFIX_ENCODER}" == "1" ]]; then
  echo "[2c/5] Building local-attention prefix encoder TorchScript wrapper"
  local_prefix_encoder_ts="${ARTIFACT_DIR}/${LOCAL_ATTENTION_PREFIX_ENCODER_BASENAME}.ts"
  local_prefix_encoder_manifest="${ARTIFACT_DIR}/${LOCAL_ATTENTION_PREFIX_ENCODER_BASENAME}-manifest.json"
  "${PYTHON_BIN}" scripts/make_local_attention_encoder_torchscript.py \
    --model "${MODEL_NAME}" \
    --output-ts "${local_prefix_encoder_ts}" \
    --output-manifest "${local_prefix_encoder_manifest}" \
    --left-context-steps "${LOCAL_ATTENTION_PREFIX_LEFT_CONTEXT_STEPS}" \
    --right-context-steps "${LOCAL_ATTENTION_PREFIX_RIGHT_CONTEXT_STEPS}" \
    --input-feature-frames "${LOCAL_ATTENTION_PREFIX_INPUT_FEATURE_FRAMES}"
fi

shopt -s nullglob
torchscript_files=( "${ARTIFACT_DIR}"/*.ts "${ARTIFACT_DIR}"/*.pt )
shopt -u nullglob
if [[ ${#torchscript_files[@]} -eq 0 ]]; then
  echo "error: no TorchScript files (*.ts/*.pt) found in ${ARTIFACT_DIR}."
  echo "Rerun export with EXPORT_FORMATS=onnx,ts and SKIP_EXPORT=0."
  exit 1
fi

echo "[3/5] Converting TorchScript -> CoreML (per component)"
ENCODER_FRAMES="${ENCODER_FRAMES:-300}"
DECODER_FRAMES="${DECODER_FRAMES:-75}"
ENABLE_STATEFUL_DECODER="${ENABLE_STATEFUL_DECODER:-0}"
DECODER_STATEFUL_INPUT_NAMES="${DECODER_STATEFUL_INPUT_NAMES:-input_states_1,input_states_2}"
DECODER_STATEFUL_WRAPPER_NAMES="${DECODER_STATEFUL_WRAPPER_NAMES:-state_1,state_2}"
ALLOW_STATEFUL_FALLBACK="${ALLOW_STATEFUL_FALLBACK:-0}"
mlpackages=()
for ts_file in "${torchscript_files[@]}"; do
  base_name="$(basename "${ts_file}")"
  base_name="${base_name%.*}"
  if [[ "${base_name}" == *-stateful-wrapper ]]; then
    echo "  - skip ${ts_file} (generated wrapper artifact)"
    continue
  fi
  manifest_file="${ARTIFACT_DIR}/${base_name}-manifest.json"
  convert_ts_file="${ts_file}"
  convert_manifest_file="${manifest_file}"
  output_mlpackage="${ARTIFACT_DIR}/${base_name}.mlpackage"
  if [[ ! -f "${manifest_file}" ]]; then
    echo "  - skip ${ts_file} (missing ${manifest_file})"
    continue
  fi
  echo "  - convert ${ts_file}"
  stateful_args=()
  if [[ "${ENABLE_STATEFUL_DECODER}" == "1" && "${base_name}" == "decoder_joint-model" ]]; then
    wrapped_ts="${ARTIFACT_DIR}/${base_name}-stateful-wrapper.ts"
    wrapped_manifest="${ARTIFACT_DIR}/${base_name}-stateful-wrapper-manifest.json"
    echo "    building stateful decoder wrapper TorchScript"
    "${PYTHON_BIN}" scripts/make_stateful_decoder_torchscript.py \
      --decoder-ts "${ts_file}" \
      --manifest "${manifest_file}" \
      --output-ts "${wrapped_ts}" \
      --output-manifest "${wrapped_manifest}" \
      --decoder-frames "${DECODER_FRAMES}"
    convert_ts_file="${wrapped_ts}"
    convert_manifest_file="${wrapped_manifest}"
    stateful_args+=( --stateful-input-names "${DECODER_STATEFUL_WRAPPER_NAMES}" )
    echo "    stateful decoder enabled (wrapper states: ${DECODER_STATEFUL_WRAPPER_NAMES})"
  fi
  convert_cmd=(
    "${PYTHON_BIN}" scripts/convert_torchscript_to_coreml.py
    --torchscript "${convert_ts_file}"
    --manifest "${convert_manifest_file}"
    --output "${output_mlpackage}"
    --target "${TARGET}"
    --compute-units all
    --encoder-frames "${ENCODER_FRAMES}"
    --decoder-frames "${DECODER_FRAMES}"
  )
  if [[ ${#stateful_args[@]} -gt 0 ]]; then
    convert_cmd+=( "${stateful_args[@]}" )
  fi
  if ! "${convert_cmd[@]}"; then
    if [[ "${ENABLE_STATEFUL_DECODER}" == "1" && "${base_name}" == "decoder_joint-model" ]]; then
      if [[ "${ALLOW_STATEFUL_FALLBACK}" == "1" ]]; then
        echo "    warning: stateful decoder conversion failed; retrying original stateless decoder conversion (ALLOW_STATEFUL_FALLBACK=1)"
        "${PYTHON_BIN}" scripts/convert_torchscript_to_coreml.py \
          --torchscript "${ts_file}" \
          --manifest "${manifest_file}" \
          --output "${output_mlpackage}" \
          --target "${TARGET}" \
          --compute-units all \
          --encoder-frames "${ENCODER_FRAMES}" \
          --decoder-frames "${DECODER_FRAMES}"
      else
        echo "error: stateful decoder conversion failed and ALLOW_STATEFUL_FALLBACK=0."
        echo "fix the stateful wrapper/conversion, or rerun with ALLOW_STATEFUL_FALLBACK=1 to force stateless fallback."
        exit 1
      fi
    else
      exit 1
    fi
  fi
  mlpackages+=( "${output_mlpackage}" )
done

if [[ ${#mlpackages[@]} -eq 0 ]]; then
  echo "error: no CoreML models were produced in step 3."
  exit 1
fi

COMPRESS_SUFFIX="${COMPRESS_SUFFIX:-int4}"
COMPRESS_ALGORITHM="${COMPRESS_ALGORITHM:-palettize}"
COMPRESS_MODE="${COMPRESS_MODE:-kmeans}"
COMPRESS_GROUP_SIZE="${COMPRESS_GROUP_SIZE:-32}"
COMPRESS_BLOCK_SIZE="${COMPRESS_BLOCK_SIZE:-32}"
COMPRESS_GRANULARITY="${COMPRESS_GRANULARITY:-auto}"
COMPRESS_ENABLE_PER_CHANNEL_SCALE="${COMPRESS_ENABLE_PER_CHANNEL_SCALE:-0}"
COMPRESS_WEIGHT_THRESHOLD="${COMPRESS_WEIGHT_THRESHOLD:-2048}"
COMPRESS_MIXED_HIGH_NBITS="${COMPRESS_MIXED_HIGH_NBITS:-8}"
COMPRESS_MIXED_HIGH_ELEMENT_RATIO="${COMPRESS_MIXED_HIGH_ELEMENT_RATIO:-0.5}"
COMPRESS_MIXED_FP16_ELEMENT_RATIO="${COMPRESS_MIXED_FP16_ELEMENT_RATIO:-0.0}"
COMPRESS_MIXED_SCORE_MODE="${COMPRESS_MIXED_SCORE_MODE:-outlier_ratio}"
COMPRESS_MIXED_SAMPLE_SIZE="${COMPRESS_MIXED_SAMPLE_SIZE:-200000}"
ENCODER_NBITS="${ENCODER_NBITS:-4}"
DECODER_NBITS="${DECODER_NBITS:-4}"
ENCODER_ALGORITHM="${ENCODER_ALGORITHM:-${COMPRESS_ALGORITHM}}"
DECODER_ALGORITHM="${DECODER_ALGORITHM:-${COMPRESS_ALGORITHM}}"
ENCODER_MODE="${ENCODER_MODE:-${COMPRESS_MODE}}"
DECODER_MODE="${DECODER_MODE:-${COMPRESS_MODE}}"
ENCODER_GROUP_SIZE="${ENCODER_GROUP_SIZE:-${COMPRESS_GROUP_SIZE}}"
DECODER_GROUP_SIZE="${DECODER_GROUP_SIZE:-${COMPRESS_GROUP_SIZE}}"
ENCODER_BLOCK_SIZE="${ENCODER_BLOCK_SIZE:-${COMPRESS_BLOCK_SIZE}}"
DECODER_BLOCK_SIZE="${DECODER_BLOCK_SIZE:-${COMPRESS_BLOCK_SIZE}}"
ENCODER_GRANULARITY="${ENCODER_GRANULARITY:-${COMPRESS_GRANULARITY}}"
DECODER_GRANULARITY="${DECODER_GRANULARITY:-${COMPRESS_GRANULARITY}}"
ENCODER_ENABLE_PER_CHANNEL_SCALE="${ENCODER_ENABLE_PER_CHANNEL_SCALE:-${COMPRESS_ENABLE_PER_CHANNEL_SCALE}}"
DECODER_ENABLE_PER_CHANNEL_SCALE="${DECODER_ENABLE_PER_CHANNEL_SCALE:-${COMPRESS_ENABLE_PER_CHANNEL_SCALE}}"
ENCODER_WEIGHT_THRESHOLD="${ENCODER_WEIGHT_THRESHOLD:-${COMPRESS_WEIGHT_THRESHOLD}}"
DECODER_WEIGHT_THRESHOLD="${DECODER_WEIGHT_THRESHOLD:-${COMPRESS_WEIGHT_THRESHOLD}}"
ENCODER_MIXED_HIGH_NBITS="${ENCODER_MIXED_HIGH_NBITS:-${COMPRESS_MIXED_HIGH_NBITS}}"
DECODER_MIXED_HIGH_NBITS="${DECODER_MIXED_HIGH_NBITS:-${COMPRESS_MIXED_HIGH_NBITS}}"
ENCODER_MIXED_HIGH_ELEMENT_RATIO="${ENCODER_MIXED_HIGH_ELEMENT_RATIO:-${COMPRESS_MIXED_HIGH_ELEMENT_RATIO}}"
DECODER_MIXED_HIGH_ELEMENT_RATIO="${DECODER_MIXED_HIGH_ELEMENT_RATIO:-${COMPRESS_MIXED_HIGH_ELEMENT_RATIO}}"
ENCODER_MIXED_FP16_ELEMENT_RATIO="${ENCODER_MIXED_FP16_ELEMENT_RATIO:-${COMPRESS_MIXED_FP16_ELEMENT_RATIO}}"
DECODER_MIXED_FP16_ELEMENT_RATIO="${DECODER_MIXED_FP16_ELEMENT_RATIO:-${COMPRESS_MIXED_FP16_ELEMENT_RATIO}}"
ENCODER_MIXED_SCORE_MODE="${ENCODER_MIXED_SCORE_MODE:-${COMPRESS_MIXED_SCORE_MODE}}"
DECODER_MIXED_SCORE_MODE="${DECODER_MIXED_SCORE_MODE:-${COMPRESS_MIXED_SCORE_MODE}}"
ENCODER_MIXED_SAMPLE_SIZE="${ENCODER_MIXED_SAMPLE_SIZE:-${COMPRESS_MIXED_SAMPLE_SIZE}}"
DECODER_MIXED_SAMPLE_SIZE="${DECODER_MIXED_SAMPLE_SIZE:-${COMPRESS_MIXED_SAMPLE_SIZE}}"

echo "[4/5] Compressing CoreML model(s) (profile=${COMPRESS_SUFFIX})"
compressed_mlpackages=()
for model_path in "${mlpackages[@]}"; do
  model_file="$(basename "${model_path}")"
  nbits="${ENCODER_NBITS}"
  algorithm="${ENCODER_ALGORITHM}"
  mode="${ENCODER_MODE}"
  group_size="${ENCODER_GROUP_SIZE}"
  block_size="${ENCODER_BLOCK_SIZE}"
  granularity="${ENCODER_GRANULARITY}"
  enable_per_channel_scale="${ENCODER_ENABLE_PER_CHANNEL_SCALE}"
  weight_threshold="${ENCODER_WEIGHT_THRESHOLD}"
  mixed_high_nbits="${ENCODER_MIXED_HIGH_NBITS}"
  mixed_high_element_ratio="${ENCODER_MIXED_HIGH_ELEMENT_RATIO}"
  mixed_fp16_element_ratio="${ENCODER_MIXED_FP16_ELEMENT_RATIO}"
  mixed_score_mode="${ENCODER_MIXED_SCORE_MODE}"
  mixed_sample_size="${ENCODER_MIXED_SAMPLE_SIZE}"
  if [[ "${model_file}" == *decoder_joint-model* ]]; then
    nbits="${DECODER_NBITS}"
    algorithm="${DECODER_ALGORITHM}"
    mode="${DECODER_MODE}"
    group_size="${DECODER_GROUP_SIZE}"
    block_size="${DECODER_BLOCK_SIZE}"
    granularity="${DECODER_GRANULARITY}"
    enable_per_channel_scale="${DECODER_ENABLE_PER_CHANNEL_SCALE}"
    weight_threshold="${DECODER_WEIGHT_THRESHOLD}"
    mixed_high_nbits="${DECODER_MIXED_HIGH_NBITS}"
    mixed_high_element_ratio="${DECODER_MIXED_HIGH_ELEMENT_RATIO}"
    mixed_fp16_element_ratio="${DECODER_MIXED_FP16_ELEMENT_RATIO}"
    mixed_score_mode="${DECODER_MIXED_SCORE_MODE}"
    mixed_sample_size="${DECODER_MIXED_SAMPLE_SIZE}"
  fi

  output_compressed="${model_path%.mlpackage}-${COMPRESS_SUFFIX}.mlpackage"
  echo "  - compress ${model_path}"
  compress_cmd=(
    "${PYTHON_BIN}" scripts/compress_coreml.py
    --model "${model_path}" \
    --output "${output_compressed}" \
    --nbits "${nbits}" \
    --algorithm "${algorithm}" \
    --mode "${mode}" \
    --group-size "${group_size}" \
    --block-size "${block_size}" \
    --granularity "${granularity}" \
    --weight-threshold "${weight_threshold}" \
    --mixed-high-nbits "${mixed_high_nbits}" \
    --mixed-high-element-ratio "${mixed_high_element_ratio}" \
    --mixed-fp16-element-ratio "${mixed_fp16_element_ratio}" \
    --mixed-score-mode "${mixed_score_mode}" \
    --mixed-sample-size "${mixed_sample_size}"
  )
  enable_per_channel_scale_norm="$(printf '%s' "${enable_per_channel_scale}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${enable_per_channel_scale}" == "1" || "${enable_per_channel_scale_norm}" == "true" ]]; then
    compress_cmd+=(--enable-per-channel-scale)
  fi
  if "${compress_cmd[@]}"; then
    compressed_mlpackages+=( "${output_compressed}" )
  else
    echo "  - warning: compression failed for ${model_path}, keeping uncompressed model"
  fi
done

benchmark_models=( "${compressed_mlpackages[@]}" )
if [[ ${#benchmark_models[@]} -eq 0 ]]; then
  benchmark_models=( "${mlpackages[@]}" )
fi

BENCH_ITERATIONS="${BENCH_ITERATIONS:-40}"
BENCH_WARMUP="${BENCH_WARMUP:-8}"
BENCH_DECODER_STEPS="${BENCH_DECODER_STEPS:-64}"

echo "[5/5] Benchmarking latency"
encoder_candidate=""
decoder_candidate=""
for candidate_model in "${benchmark_models[@]}"; do
  model_file="$(basename "${candidate_model}")"
  if [[ -z "${encoder_candidate}" && "${model_file}" == *encoder-model* ]]; then
    encoder_candidate="${candidate_model}"
  fi
  if [[ -z "${decoder_candidate}" && "${model_file}" == *decoder_joint-model* ]]; then
    decoder_candidate="${candidate_model}"
  fi
done

if [[ -n "${encoder_candidate}" && -n "${decoder_candidate}" ]]; then
  encoder_base="$(basename "${encoder_candidate}" .mlpackage)"
  encoder_base="${encoder_base%-${COMPRESS_SUFFIX}}"
  decoder_base="$(basename "${decoder_candidate}" .mlpackage)"
  decoder_base="${decoder_base%-${COMPRESS_SUFFIX}}"

  encoder_manifest="${ARTIFACT_DIR}/${encoder_base}-manifest.json"
  decoder_manifest="${ARTIFACT_DIR}/${decoder_base}-manifest.json"

  if [[ -f "${encoder_manifest}" && -f "${decoder_manifest}" ]]; then
    echo "  - benchmark TDT components"
    if "${PYTHON_BIN}" scripts/benchmark_tdt_components.py \
      --encoder-model "${encoder_candidate}" \
      --encoder-manifest "${encoder_manifest}" \
      --decoder-model "${decoder_candidate}" \
      --decoder-manifest "${decoder_manifest}" \
      --iterations "${BENCH_ITERATIONS}" \
      --warmup "${BENCH_WARMUP}" \
      --decoder-steps "${BENCH_DECODER_STEPS}" \
      --compute-units cpu_and_ne; then
      :
    else
      echo "  - warning: TDT component benchmark failed"
    fi
  else
    echo "  - warning: missing encoder/decoder manifests for TDT benchmark"
  fi
else
  echo "  - fallback: encoder-only benchmark"
  encoder_bench_done="false"
  for candidate_model in "${benchmark_models[@]}"; do
    model_file="$(basename "${candidate_model}")"
    if [[ "${model_file}" == *encoder* ]]; then
      base_name="${model_file%.mlpackage}"
      base_name="${base_name%-${COMPRESS_SUFFIX}}"
      manifest_file="${ARTIFACT_DIR}/${base_name}-manifest.json"
      if [[ -f "${manifest_file}" ]]; then
        echo "  - benchmark ${candidate_model}"
        if "${PYTHON_BIN}" scripts/benchmark_coreml_model.py \
          --model "${candidate_model}" \
          --manifest "${manifest_file}" \
          --iterations "${BENCH_ITERATIONS}" \
          --warmup "${BENCH_WARMUP}" \
          --compute-units cpu_and_ne; then
          :
        else
          echo "  - warning: benchmark failed for ${candidate_model}"
        fi
        encoder_bench_done="true"
        break
      fi
    fi
  done

  if [[ "${encoder_bench_done}" != "true" ]]; then
    echo "  - skip benchmark: no encoder* model+manifest pair found"
  fi
fi

echo "Done. Artifacts in ${ARTIFACT_DIR}"
