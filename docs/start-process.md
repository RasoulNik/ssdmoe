# Start Process

Use this process after a reboot or if the thread is interrupted.

## 1. Re-establish machine state

Run and record:

```bash
date '+%Y-%m-%d %H:%M:%S %Z'
df -h /System/Volumes/Data | sed -n '1,2p'
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
find ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec -maxdepth 1 -name 'model-*.safetensors' -type l | wc -l
```

## 2. Reconfirm artifacts

Key generated files:

- `.run/qwen35b-expert-index.json`
- `.run/stream-qwen35b-bench.json`
- `.run/stream-qwen35b-bench-profiled.json`
- `.run/stream-qwen35b-bench-nocache.json`

Core source files:

- `streaming_qwen/expert_store.py`
- `streaming_qwen/model_io.py`
- `streaming_qwen/runtime.py`
- `streaming_qwen/streamed_switch.py`
- `scripts/stream_qwen_generate.py`
- `scripts/stream_qwen_bench.py`

## 3. Re-read project notes

Read these first:

- `docs/streaming-qwen35b-assessment.md`
- `docs/streaming-experiment-log.md`

## 4. Default benchmark command

```bash
PYTHONPATH=. poetry run python scripts/stream_qwen_bench.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec \
  --index .run/qwen35b-expert-index.json \
  --prompt 'Explain what a mixture-of-experts model is in two concise sentences.' \
  --top-ks 8,6,4,3,2 \
  --max-tokens 24 \
  --output .run/stream-qwen35b-bench.json
```

## 5. Profiling command

```bash
PYTHONPATH=. poetry run python scripts/stream_qwen_bench.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec \
  --index .run/qwen35b-expert-index.json \
  --prompt 'Explain what a mixture-of-experts model is in two concise sentences.' \
  --top-ks 8,4,2 \
  --max-tokens 16 \
  --output .run/stream-qwen35b-bench-profiled.json
```

## 6. Native benchmark target

When testing Python overhead versus raw I/O, prefer the native Objective-C benchmark in:

- `native/expert_read_bench.m`

## 7. Rules for continued work

- Keep `flash-moe` as the baseline architecture.
- Use one large model download at a time.
- Update `docs/streaming-experiment-log.md` after every meaningful experiment.
- Use the internet when assumptions need validation.
