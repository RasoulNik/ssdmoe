# Research Methodology

## Objective

Adapt the `flash-moe` architecture to streamed Qwen 3.5 inference on a `16 GB` M4 Air without loading the full model into RAM and without duplicating the full expert weights on disk.

## Method

1. Start from the reference implementation:
   - `danveloper/flash-moe`
2. Replace assumptions with measurements:
   - SSD throughput
   - routed expert payload size
   - resident non-expert footprint
   - end-to-end generation tok/s
3. Keep an experiment log:
   - command used
   - artifact produced
   - result
   - conclusion
4. Use the internet selectively:
   - official Apple documentation for hardware / OS behavior
   - official model cards / papers for model architecture assumptions
5. Prefer bounded changes:
   - one large model download at a time
   - one optimization variable at a time
   - keep benchmark prompts fixed while comparing runs
6. Separate cold-start from steady-state:
   - cold runs capture first-touch costs
   - warm runs use an explicit warmup generation on the same loaded model
   - do not compare a warmed multi-`K` sweep against a cold single-`K` run
7. Use external analysis only as hypothesis input:
   - validate claims about Apple silicon and compression against local measurements
   - prefer official Apple documentation when browsing the web

## Current hierarchy of evidence

Highest confidence:

- local measurements on this machine
- code paths exercised end-to-end

Medium confidence:

- microbenchmarks on isolated expert reads
- official Apple docs that describe OS or Metal behavior

Lower confidence:

- extrapolations from larger systems
- assumptions from other machines

## Current optimization order

1. Adjust `K`
2. Reduce Python/materialization overhead
3. Test hardware-specific toggles
4. Move hot path toward native code
5. Re-evaluate whether bounded caching helps or hurts
6. Shift from raw read tuning to native materialization and fused expert execution once read time is no longer dominant
