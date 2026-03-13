# Streaming Segment80 Baseline

Primary development gate for the cache-aware local-attention restructure.

## Runtime profile

```bash
PARAKEET_COREML_MODEL_SUFFIX=odmbp-approx \
PARAKEET_STREAM_MODE=rewrite-prefix \
PARAKEET_STREAM_PREFIX_LEFT_CONTEXT_FRAMES=160 \
PARAKEET_STREAM_PREFIX_RIGHT_CONTEXT_FRAMES=0 \
PARAKEET_STREAM_PREFIX_ALLOW_RIGHT_CONTEXT=0 \
PARAKEET_STREAM_PREFIX_DECODE_STRIDE_SEC=0.60 \
PARAKEET_STREAM_PREFIX_ADAPTIVE=0 \
PARAKEET_STREAM_PREFIX_ENCODER_CACHE=1 \
PARAKEET_STREAM_LATEST_FIRST=1 \
PARAKEET_STREAM_CHUNK_MS=500 \
PARAKEET_STREAM_HOP_MS=250 \
PARAKEET_STREAM_MAX_BATCH_SEC=0.5 \
PARAKEET_STREAM_BACKLOG_SOFT_SEC=5.0 \
PARAKEET_STREAM_BACKLOG_TARGET_SEC=1.5 \
PARAKEET_STREAM_MAX_SPEECH_CHUNKS=80 \
PARAKEET_STREAM_MAX_STAGNANT_CHUNKS=24 \
PARAKEET_DECODE_ONLY_WHEN_SPEECH=0 \
PARAKEET_STREAM_FLUSH_ON_SPEECH_END=0 \
bash scripts/run_transcribe_macos_release.sh
```

## Primary benchmark gate

Use the 60-second realtime proxy, not Earnings22, for fast iteration.

Artifact:
- `artifacts/realtime-bench-runs/rt_60s_segment80_post_autorelease/summary.json`
- `artifacts/realtime-bench-runs/rt_60s_segment80_post_autorelease/summary.confirmed.txt`

Expected band:
- `infer_rtfx`: about `1.83x`
- `dropped_chunks`: `0`
- `confirmed_latency_ms_avg`: about `786ms`
- `confirmed_latency_ms_p95`: about `1334ms`
- `first_token_ms_avg`: about `764ms`
- `first_token_ms_p95`: about `1334ms`

Transcript preview:
- `So I have created here a MVVM kind of style for a chatbot that works recently, sorry, locally within the computer...`

## Promotion rule for the restructure

A new cache-aware local-attention path should not replace this baseline unless on the 60-second benchmark it:
- matches or beats transcript quality
- matches or beats realtime factor materially enough to justify complexity
- does not regress stability

Earnings22 remains a slower confirmation pass only after the 60-second gate looks good.
