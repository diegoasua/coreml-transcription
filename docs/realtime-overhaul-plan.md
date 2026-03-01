# Realtime Overhaul Plan (Low-Latency First)

Owner: Codex + Diego
Policy: prioritize low transcript latency over completeness. Dropping stale audio under overload is allowed by design.

## Goal
- Reach stable realtime behavior for Parakeet TDT CoreML on macOS/iOS modern runtime.
- Primary KPI: lips-to-screen latency and bounded queue; secondary KPI: correction rate.

## Milestones

### M1. Deterministic Realtime Harness (completed)
- Add a reproducible benchmark mode (audio file -> simulated realtime queue).
- Report every ~200ms:
  - audio cursor
  - transcript cursor
  - queue seconds
  - first-token latency (avg/p95/worst)
  - confirmed latency (avg/p95)
  - corrections per minute
  - infer RTFx / wall RTFx
- Output JSON summary with pass/fail thresholds.

Success gate:
- queue p95 <= 0.5s
- first-token p95 <= 300ms (intermediate), then <= 160ms target
- confirmed latency p95 <= 1.7s

Current status (2026-03-01):
- Harness now reports stable queue, first-token, confirmed latency, corrections, infer/wall RTFx.
- Added transcript previews and optional text-change debug logging (`PARAKEET_BENCH_DEBUG_TEXT=1`).
- Tuned realtime bench defaults to realistic VAD + speech-gated decode.
- Fixed streaming overlap bug: decoder now consumes hop-sized incremental audio instead of overlapped chunk windows.
- Changed default streaming policy to preserve model state across speech/silence boundaries (`flushOnSpeechEnd=false`) and aligned VAD defaults (`start=-50dBFS`, `end=-58dBFS`).
- On 10s slice: first-token p95 ~69ms, infer RTFx ~1.63x, queue 0s, confirmed p95 still above target.
 - After overlap fix + no-flush default: 60s slice first-token p95 ~143ms, confirmed p95 ~49ms, queue 0s, infer RTFx ~1.05x (passes latency gates).

### M2. Stateful Decoder Integration (in progress)
- Convert decoder model to use CoreML state API (iOS18+/macOS15+ target only).
- Replace explicit recurrent state IO tensors in runtime with `MLState` prediction path.
- Keep stateless path behind feature flag for A/B.

Success gate:
- decoder step latency reduced materially vs stateless baseline
- correction rate unchanged or improved

### M3. Incremental Feature + Encoder Scheduling
- Replace full recompute strategy with incremental feature updates.
- Keep fixed-size causal context window and deadline scheduler.
- Never replay stale backlog in realtime mode.

Success gate:
- sustained queue bounded under speaking continuously for >= 60s
- no progressive lag accumulation

### M4. App Runtime UX Stabilization
- Smooth hypothesis + confirmed rendering cadence.
- Add on-screen diagnostics tied to harness metrics.
- Final launch profile presets for low-latency mode.

## Current blockers
- Decoder currently has CPU-preferred op mix (including LSTM path), causing loop overhead.
- Existing live path still mixes offline-style compute decisions.

## Immediate next actions
1. Complete M1 and collect baseline JSON from 3 fixed clips.
2. Begin M2 decoder stateful conversion and runtime hook.
3. Re-benchmark with same clips and compare deltas.
