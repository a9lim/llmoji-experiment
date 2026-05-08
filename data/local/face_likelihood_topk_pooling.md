# Top-k per-prompt pooling — face_likelihood

**GT floor:** total_emit_count ≥ 3
**Source:** full per-cell parquets

Each cell: accuracy / κ. Bold = best k for the encoder. **'all' = mean over all prompts** (current default in script 50 / 52 / 53).

| encoder | k=1 | k=2 | k=3 | k=5 | k=all | best |
|---|---|---|---|---|---|---|
| gemma | 37% / 0.29 | 37% / 0.29 | 39% / 0.31 | 39% / 0.32 | **42% / 0.35** | k=all |

## Best-k per encoder

- **gemma**: no meaningful difference (baseline-all 42%, best 42%)
