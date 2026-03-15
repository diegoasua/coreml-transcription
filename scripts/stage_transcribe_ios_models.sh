#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${PARAKEET_COREML_MODEL_DIR:-$repo_root/artifacts/parakeet-tdt-0.6b-v2}"
import_dest_root="${PARAKEET_IOS_IMPORT_DIR:-$repo_root/artifacts/ios-model-import}"
import_dest_dir="$import_dest_root/parakeet-tdt-0.6b-v2"
encoder_suffix="${PARAKEET_COREML_MODEL_SUFFIX:-odmbp-approx}"
decoder_suffix="${PARAKEET_COREML_DECODER_SUFFIX:-}"

if [[ ! -d "$source_dir" ]]; then
  echo "error: model directory not found: $source_dir" >&2
  exit 1
fi

if [[ -z "$decoder_suffix" ]]; then
  preferred_decoder="odmbp-approx-stateful-v2"
  if [[ -d "$source_dir/decoder_joint-model-$preferred_decoder.mlpackage" ]]; then
    decoder_suffix="$preferred_decoder"
  else
    decoder_suffix="$encoder_suffix"
  fi
fi

encoder_path="$source_dir/encoder-model-$encoder_suffix.mlpackage"
decoder_path="$source_dir/decoder_joint-model-$decoder_suffix.mlpackage"
vocab_path="$source_dir/vocab.txt"

for required_path in "$encoder_path" "$decoder_path" "$vocab_path"; do
  if [[ ! -e "$required_path" ]]; then
    echo "error: required iOS runtime asset not found: $required_path" >&2
    exit 1
  fi
done

rm -rf "$import_dest_dir"
mkdir -p "$import_dest_dir"
cp -R "$encoder_path" "$import_dest_dir/"
cp -R "$decoder_path" "$import_dest_dir/"
cp "$vocab_path" "$import_dest_dir/"

echo "Staged iOS model resources:"
echo "  source:           $source_dir"
echo "  import dest:      $import_dest_dir"
echo "  encoder suffix:   $encoder_suffix"
echo "  decoder suffix:   $decoder_suffix"
echo "  note:             import this folder from Files inside the native iOS app"
