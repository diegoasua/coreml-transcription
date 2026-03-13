# Cache-Aware Local-Attention Oracle

Current Python oracle status for the cache-aware local-attention restructure.

## Objective

Replace the current NeMo internal cache experiment with an explicit streaming encoder wrapper that:
- keeps the current TDT decoder family
- supports local attention
- makes cache behavior visible and controllable
- can be promoted to Core ML only if it matches the current 60-second baseline band

The current production baseline remains:
- [streaming-segment80-baseline.md](/Users/diegoasua/Developer/coreml-transcription/docs/baselines/streaming-segment80-baseline.md)

## Current Best NeMo-Backed Oracle

Best historical stable config so far:
- mode: `custom-wrapper`
- attention context: `80,8`
- shift steps: `4`
- depth: `1`
- max segment steps: `64`

Artifacts:
- 20s: [custom_tdt_20s_80_8_shift4_seg64.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/custom_tdt_20s_80_8_shift4_seg64.json)
- 60s: [custom_tdt_60s_ctx80_8_shift4_seg64.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/custom_tdt_60s_ctx80_8_shift4_seg64.json)

Observed results:

| clip | config | simple WER | hyp/ref words | elapsed |
| --- | --- | ---: | ---: | ---: |
| 20s | `80,8 / shift4 / depth1 / seg64` | `0.2439` | `37 / 41` | `85.8s` |
| 60s | `80,8 / shift4 / depth1 / seg64` | `0.5971` | `90 / 139` | `260.8s` |
| 20s | `64,8 / shift4 / depth1 / seg64` | `0.5122` | `27 / 41` | `69.4s` |
| 60s | `36,4 / shift4 / depth1 / seg64` | `0.7770` | `47 / 139` | `135.8s` |

Interpretation:
- the custom TDT greedy decoder was necessary
- wider local context helped materially
- long-form quality is still far from promotable
- the older `0.2439` `20s` result should be treated as a historical reference point, not the current oracle truth, because the script has changed materially since then

## Stability Boundary

The current patched NeMo local-attention cache path is only stable up to about:
- right context `<= 8`

Contexts beyond that started failing inside NeMo with encoder tensor shape mismatches.

Implication:
- the current NeMo internal cache path is useful for exploration
- it is not a clean foundation for the final architecture

## Architecture Decision

Next path:
- stop depending on NeMo internal encoder cache behavior
- keep local attention
- add an explicit rolling cache wrapper in the oracle

First explicit wrapper to build:
- manual cache at the pre-encoded frame level
- rerun full encoder on:
  - `stable pre-encoded cache`
  - `+ current chunk pre-encoded frames`
- decode from scratch per rolling window

Why this first:
- it removes the opaque NeMo cache path
- it keeps the experiment small enough to validate quickly
- it isolates encoder-wrapper behavior from decoder-state carry bugs

What it is not:
- not the final low-compute design
- not yet per-layer KV/conv rotating cache

## First Explicit Wrapper Scaffold

Implemented mode:
- `manual-preencode-cache`

Behavior:
- uses local attention
- does not use NeMo internal encoder cache tensors
- keeps a manual rolling cache of finalized pre-encoded frames
- reruns the encoder on `stable cache + current chunk`
- decodes from scratch for each rolling window

Artifacts:
- 6s smoke: [manual_preencode_6s_ctx80_8_shift4_seg64.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/manual_preencode_6s_ctx80_8_shift4_seg64.json)
- 20s smoke: [manual_preencode_20s_ctx80_8_shift4_seg64.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/manual_preencode_20s_ctx80_8_shift4_seg64.json)

Observed results:
- 6s:
  - preview: `Mm-hmm. So I have grade edit of your M V V V`
  - `simple_wer = 2.0`
- 20s:
  - preview: `And you free framework in a OS to store this six, and macro across to 26 mix...`
  - normalized simple WER against [offline_20s_odmbp_approx.reference.txt](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/offline_20s_odmbp_approx.reference.txt): `0.9149`

Interpretation:
- the explicit wrapper runs
- it removes dependence on the patched NeMo cache path
- it is currently much worse than the best NeMo-backed oracle
- it should be treated as scaffolding, not a candidate

Next technical move from this scaffold:
- replace pre-encoded frame cache with explicit per-layer rotating cache
- then move the decoder back to incremental state carry once the encoder path is trustworthy

## Second Explicit Wrapper Scaffold

Implemented mode:
- `manual-layer-cache`

Behavior:
- keeps the asymmetric local-attention `q/k` patch
- bypasses `model.encoder(...cache_last_channel/cache_last_time...)`
- keeps explicit per-layer rotating caches for:
  - attention channel state
  - convolution time state
- runs the layer stack manually
- keeps incremental TDT decode state like the better NeMo-backed wrapper

Artifacts:
- 6s smoke: [manual_layer_6s_ctx80_8_shift4_seg64.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/manual_layer_6s_ctx80_8_shift4_seg64.json)
- 20s smoke: [manual_layer_20s_ctx80_8_shift4_seg64.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/manual_layer_20s_ctx80_8_shift4_seg64.json)

Observed results:
- 6s:
  - preview: `So I have created here a MVV.`
  - `simple_wer = 0.4`
- 20s initial smoke:
  - preview: `So I have created here a M V M kind of for a bot that works written within users...`
  - `simple_wer = 0.6383`
- 20s traced rerun:
  - artifact: [manual_layer_20s_ctx80_8_shift4_seg64_decodetrace.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/manual_layer_20s_ctx80_8_shift4_seg64_decodetrace.json)
  - preview: `So I have created here a M V M um kind of for a chat at b that works written within computer...`
  - `simple_wer = 0.5106`

Current interpretation:
- this is materially better than `manual-preencode-cache`
- it proves the per-layer rotating-cache direction is executable
- under the current traced script revision, it is also better than the matching traced `custom-wrapper` rerun on `20s`
- the remaining gap is still quality, not plumbing

Current next move:
- inspect the first decoder divergence after cache saturation
- patch the decoder/cache boundary semantics before doing more broad tuning

Negative result already checked:
- replacing the ring caches with exact append-and-trim caches made the `20s` result worse
- observed regression:
  - `simple_wer: 0.4255 -> 0.6809`
  - faster runtime, but clearly worse text quality
- conclusion:
  - cache wrap arithmetic is not the dominant remaining bug

Reset-sweep result on `manual-layer-cache` (`20s`, `80,8 / shift4`):
- `seg32`: `0.5319`
- `seg48`: `0.5319`
- no-reset / `seg64` path remains better at `0.4255`

Conclusion:
- early segment resets are not the main fix for the remaining drift on `20s`

## Current Trace Findings

Paired decode-event traces:
- `custom-wrapper`: [custom_tdt_20s_80_8_shift4_seg64_decodetrace.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/custom_tdt_20s_80_8_shift4_seg64_decodetrace.json)
- `manual-layer-cache`: [manual_layer_20s_ctx80_8_shift4_seg64_decodetrace.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/manual_layer_20s_ctx80_8_shift4_seg64_decodetrace.json)

Current paired `20s` traced results:
- `custom-wrapper`: `simple_wer = 0.6170`
- `manual-layer-cache`: `simple_wer = 0.5106`

Key retained findings:
- cache saturation begins around stream step `20`
- the first post-saturation decode mismatch appears at stream step `24`
- first emitted mismatch after saturation:
  - `custom-wrapper` emits `▁U` in the draft phase
  - `manual-layer-cache` stays blank on that same step
- the first strong semantic drift appears around steps `32-34`
  - `manual-layer-cache` keeps the `chatbot` branch longer
  - `custom-wrapper` rewrites the same region into `while` / `bad`

Interpretation:
- the remaining error is now concentrated in how boundary frames move from draft to committed decode after cache saturation
- this is more consistent with decoder/cache boundary semantics than with reset cadence or ring-buffer wrap logic

Targeted negative result:
- extra draft holdback did not help
- artifacts:
  - [manual_layer_20s_ctx80_8_shift4_seg64_xd0.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/manual_layer_20s_ctx80_8_shift4_seg64_xd0.json)
  - [manual_layer_20s_ctx80_8_shift4_seg64_xd1.json](/Users/diegoasua/Developer/coreml-transcription/artifacts/tmp/manual_layer_20s_ctx80_8_shift4_seg64_xd1.json)
- results:
  - `extra_draft_frames=0`: `0.7447`
  - `extra_draft_frames=1`: `0.6383`
  - `extra_draft_frames=2`: collapsed to empty transcript (`1.0`)

Conclusion:
- “commit later” by a simple fixed extra holdback is not the fix
- the next lever should be the manual-layer decoder/cache state boundary itself

## Promotion Rule

The local-attention restructure should not move into Core ML / Swift unless it beats or matches the current 60-second baseline on:
- transcript quality
- realtime factor
- stability

The 60-second proxy is the main development gate.
`earnings22-3hours` remains a slower confirmation pass only after the 60-second gate looks good.
