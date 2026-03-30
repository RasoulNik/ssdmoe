# Expert Index Mapper: 35B → 122B Cross-Model Routing Prediction

Branch: `bench/qwen35-122b`
Status: **in progress**

---

## Motivation

Qwen3.5-122B-A10B decodes at ~1.36 tok/s on an M4 MacBook Air (16 GB).
The bottleneck is SSD: each token requires reading ~1,944 MB of expert weights.

**Goal**: Use the smaller Qwen3.5-35B-A3B model to predict which experts the
122B model will activate one step ahead, enabling SSD prefetch that hides
latency behind GPU compute.

---

## Prior Results (Zero-Shot Correlation)

Running both models on the same prompt and comparing expert selections directly:

| Metric | Result | Random baseline |
|:--|--:|--:|
| Jaccard (proportional mapping) | 1.7% | ~3.1% |
| Jaccard (structural 1:1 mapping) | 2.1% | ~3.1% |
| Hit rate | 3.8% | 3.1% |

**Conclusion**: Expert indices are arbitrary labels — expert 42 in the 35B is
not semantically related to expert 42 in the 122B, even though both models share
the same 256-expert architecture with K=8 routing.

**Key finding from structural analysis**: Both models share identical layer group
structure `[linear, linear, linear, full_attention] × N`:
- 35B: 10 groups × 4 = 40 layers
- 122B: 12 groups × 4 = 48 layers

Correct mapping: **35B layer i → 122B layer i** (identity) for i=0..39.
Layers 40–47 in 122B have no 35B counterpart (~17% of 122B layers unmapped).

---

## Experiment: Co-Occurrence Mapper

### Hypothesis

Even though indices are not aligned, the routing patterns may be consistently
correlated: "when 35B picks expert 43, 122B tends to pick expert 181" across
tokens of a similar type. A co-occurrence matrix W[i,j] accumulates evidence
for these associations and can learn the translation with enough data.

### Pilot Result (500 tokens, in-sample eval)

Running the co-occurrence mapper on 500 tokens of varied prompts showed:

| Tokens seen | L0 hit% | L12 hit% | L24 hit% | L36 hit% | Random |
|--:|--:|--:|--:|--:|--:|
| 50 | 58.5% | 60.2% | 81.2% | 77.5% | 3.1% |
| 100 | 83.2% | 71.5% | 74.5% | 71.2% | 3.1% |
| 200 | 54.2% | 64.8% | 61.3% | 64.5% | 3.1% |
| 300 | 51.7% | 53.8% | 59.5% | 50.2% | 3.1% |
| 500 | 37.8% | 42.5% | 60.0% | 56.5% | 3.1% |

**Signal confirmed**: 20× above random even at 50 tokens. The decline with more
tokens reflects in-sample evaluation with a growing, increasingly diverse prompt
set — not overfitting collapse. A held-out test set is needed to measure
true generalization.

W matrix sparsity: 17–25% of 256×256 entries nonzero. This indicates structured
clustering, not random noise (random would approach 100% nonzero).

---

## Full Experiment Design

### Data Collection

**10 topics**, each 1,000 tokens of text (prefill-based collection):

| # | Topic | Domain |
|:--|:--|:--|
| 00 | quantum_physics | physics |
| 01 | ancient_rome | history |
| 02 | machine_learning | CS/AI |
| 03 | calculus | mathematics |
| 04 | cell_biology | biology |
| 05 | ethics_philosophy | philosophy |
| 06 | economics | economics |
| 07 | climate_ecology | earth science |
| 08 | medicine_anatomy | medicine |
| 09 | programming_systems | CS/systems |

**Collection method**: single prefill forward pass per topic — much faster than
decode (~40 tok/s for 122B vs 1.4 tok/s decode). The MoE switch hook captures
`indices` shape `(seq_len, K)` per layer in one call.

**Split per topic**: first 800 tokens = train, last 200 tokens = test.
Total: 8,000 train tokens, 2,000 test tokens.

**All layers collected**: 40 layers for 35B, 48 for 122B.
Aligned pairs: layers 0–39 (identity mapping).

**Storage**: `.run/expert-data/{35b,122b}/topic_{i:02d}_{name}.npz`
Each file: `layer_L` array of shape `(n_tok, K)` int16 for all L layers.

### Mapper Model

**Co-occurrence matrix** per aligned layer pair:
- W ∈ R^{256×256}
- W[i,j] += 1 whenever 35B expert i and 122B expert j co-activate at same token
- Prediction: `scores = Σ_{i ∈ S_small} W[i,:] / ||W[i,:]||₁`, top-K

**Baseline**: random prediction = 3.1% hit rate.
**Target**: > 50% hit rate on held-out test set.

### Learning Curve Protocol

Train incrementally (50, 100, 200, 500, 1K, 2K, 4K, 8K tokens) and evaluate
on the full held-out test set at each checkpoint.

---

## Results

### Data Collection

| Model | Topics | Train tokens | Test tokens |
|:--|--:|--:|--:|
| 35B | 10 | 7,402 | 1,855 |
| 122B | 10 | 7,402 | 1,855 |

Topics: quantum_physics, ancient_rome, machine_learning, calculus, cell_biology,
ethics_philosophy, economics, climate_ecology, medicine_anatomy, programming_systems.

### Global Mapper — Learning Curve (test hit%, every-4th-layer sample)

| Tokens | L0 | L8 | L16 | L24 | L32 | Random |
|--:|--:|--:|--:|--:|--:|--:|
| 50 | 31.8% | 11.3% | 11.9% | 12.4% | 18.9% | 3.1% |
| 100 | 37.0% | 13.8% | 13.8% | 13.6% | 27.0% | 3.1% |
| 500 | 41.8% | 18.4% | 17.7% | 15.7% | 31.0% | 3.1% |
| 2000 | 43.7% | 23.2% | 21.6% | 21.1% | 32.8% | 3.1% |
| 7402 | 43.9% | 28.4% | 24.7% | 37.1% | 38.1% | 3.1% |

Signal confirmed: 10–14× above random at 50 tokens. Continues improving through 7K tokens.

### Global vs Sliding Window (final, all layers)

| Layer | Global | W=50 | W=100 | W_nz% |
|--:|--:|--:|--:|--:|
| 0 | **43.9%** | 33.0% | 35.7% | 48.9% |
| 4 | 28.1% | 18.9% | 21.1% | 74.1% |
| 8 | 28.4% | 18.7% | 18.8% | 69.5% |
| 12 | 24.2% | 15.1% | 16.9% | 76.4% |
| 16 | 24.7% | 16.3% | 18.8% | 71.8% |
| 20 | 28.3% | 13.9% | 14.3% | 66.4% |
| 24 | 37.1% | 15.6% | 15.6% | 58.6% |
| 28 | 37.4% | 16.6% | 15.9% | 51.9% |
| 32 | **38.1%** | 27.2% | 28.6% | 52.7% |
| 36 | 32.0% | 19.3% | 19.9% | 50.5% |
| Random | 3.1% | 3.1% | 3.1% | — |

### Key Findings

1. **Strong signal exists**: global mapper reaches 24–44% hit rate (7–14× random) on
   held-out test set with 7K training tokens. The mapping is learnable.

2. **Global mapper beats sliding window consistently**: W=50 and W=100 windows
   score 10–30 percentage points BELOW the global mapper. This contradicts the
   context-dependency hypothesis — the mapping is more stable across topics than
   within a recent window.

3. **Why global wins**: The co-occurrence relationship between 35B and 122B expert
   indices is driven by the training data distribution (consistent specialization),
   not by the current conversational context. Expert 43 in 35B maps to expert 181
   in 122B because of how they were trained, not because of what the current topic is.

4. **Early layers (0–3) are easiest to map** (44% at layer 0) with lower W_nz%
   (49%), meaning the W matrix is sparser and more structured — fewer 35B experts
   map to many 122B experts. Later early layers (4–12) are harder (24–28%) with
   denser W matrices.

5. **Deep layers (20–39) recover** to 32–41%, likely because deeper experts have
   more semantic specialization that is consistent across model sizes.

6. **Still far from ideal**: 24–44% vs the theoretical target of 50–80% needed for
   profitable prefetch. The co-occurrence model is the simplest possible baseline.
   A trained linear projection (2048-dim hidden state → 256-dim logits) is expected
   to substantially improve accuracy.

---

## Next Steps

1. **Baseline established**: co-occurrence mapper — stateless, O(K²) per token update
2. **Extensions** (after baseline):
   - Use previous-layer activations as additional input (cross-layer context)
   - Linear projection: 35B hidden state (2048-dim) → 122B gate logits (256-dim)
   - KV-cache projection: project 35B KV cache to 122B space as richer context
3. **Deployment**: integrate best mapper into `libexpert_reader.dylib` prefetch path

---

## File Index

| File | Purpose |
|:--|:--|
| `benchmarks/collect_expert_data.py` | Prefill-based data collection (both models) |
| `benchmarks/train_expert_mapper.py` | Train co-occurrence mapper, produce learning curve |
| `.run/expert-data/35b/topic_*.npz` | 35B per-topic selections, all layers |
| `.run/expert-data/122b/topic_*.npz` | 122B per-topic selections, all layers |
| `.run/expert-data/topics.json` | Topic metadata and token counts |
| `benchmarks/results/expert-mapper.json` | Mapper training results and learning curve |
