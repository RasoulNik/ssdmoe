# Streaming Qwen 3.5 35B on This M4 Air

Date: 2026-03-20

## Machine constraints

- Host: MacBook Air `Mac16,12`
- Chip: Apple `M4`
- Memory: `16 GB` unified
- CPU cores: `10`
- Internal SSD: `256 GB`
- Free disk at assessment time: about `37 GiB`

## Existing local baselines

Measured with `mlx_lm benchmark -p 128 -g 128 -n 1`:

- `Qwen3.5-4B-MLX-4bit`: `16.36 tok/s` generation, `2.69 GB` peak memory
- `Qwen3.5-9B-MLX-4bit`: `16.98 tok/s` generation, `5.34 GB` peak memory

These numbers are useful only as local compute baselines. The final prototype is not constrained to the MLX server path.

## Candidate target

Current target candidate:

- `mlx-community/Qwen3.5-35B-A3B-4bit`

Observed from the Hugging Face repository metadata:

- Snapshot size: about `19.02 GiB`
- Architecture: `Qwen3_5MoeForConditionalGeneration`
- Text config:
  - `hidden_size=2048`
  - `num_hidden_layers=40`
  - `num_experts=256`
  - `num_experts_per_tok=8`
  - `moe_intermediate_size=512`
  - `shared_expert_intermediate_size=512`

Important caveat:

- The official MLX checkpoint is the multimodal variant and includes vision weights we do not need for text inference.
- Because disk headroom is limited, the design must stream directly from the original safetensors shards.
- Full repacking into per-layer expert files would create an unacceptable duplicate on-disk copy.

## Storage-side measurements

Measured against local Hugging Face model shards:

- Sequential read throughput: about `2.50 GB/s`
- Large random `pread()` throughput (`64 x 7 MiB` reads): about `0.93 GB/s`

## Theoretical expert I/O budget

For the 35B 4-bit MoE layout, each expert is roughly `1.6875 MiB`.

Per generated token, assuming:

- `40` layers
- `8` routed experts per token

the cold-read traffic is about:

- `0.527 GiB/token`

That implies this approximate cold-read ceiling:

- at `10 tok/s`: `5.27 GiB/s`
- at `20 tok/s`: `10.55 GiB/s`
- at `30 tok/s`: `15.82 GiB/s`

## Initial conclusion

On this machine, `20-30 tok/s` for fully streamed 35B MoE inference is very unlikely if expert reads are mostly cold.

The measured SSD numbers suggest a much lower ceiling:

- around `4-5 tok/s` as an optimistic upper bound under ideal large sequential access
- closer to `1-3 tok/s` if the access pattern behaves like large random `pread()`

That does not make the project pointless. It changes the objective:

1. Build a working streamed-weight prototype that never loads the full 35B model into RAM.
2. Keep KV cache resident and bounded.
3. Avoid duplicate on-disk repacks.
4. Measure the real generation throughput and cache behavior.
5. Document where the bottleneck lands: SSD, page cache, or compute.

## Revised conclusion after implementation

After building and benchmarking the streamed prototype, the cold-read estimate was directionally right but incomplete.

What changed:

- a native batch reader improved the expert fetch path
- concurrent native component reads materially improved warmed steady-state throughput
- once the reader path improved, lower-`K` runs stopped being purely I/O-bound

What now appears true:

- cold or lightly warmed throughput is still far below `20-30 tok/s`
- long-lived warmed steady-state can be much faster than the cold estimate suggested
- the real limiter is the full pipeline between SSD bytes and routed expert compute, not raw storage bandwidth alone
- `K=1` can reach the target range on this machine, but the sample quality is obviously degraded
- `K=2` and above still look like a mixed I/O plus expert-compute problem

## Current implementation direction

Preferred first implementation:

- custom standalone prototype
- no dependency on the existing MLX HTTP backend
- direct expert streaming from original safetensors shards using an index file
- load only non-expert weights eagerly
- rely on OS page cache before adding any application-level cache

The first deliverables should be:

- a shard index for routed expert tensors
- a minimal generation loop that streams the selected experts on demand
- reproducible timing for per-token expert I/O and end-to-end generation
