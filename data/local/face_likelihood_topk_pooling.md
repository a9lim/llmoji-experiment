# Top-k per-prompt pooling — face_likelihood

**GT floor:** total_emit_count ≥ 3
**Source:** full per-cell parquets

Each cell: accuracy / κ. Bold = best k for the encoder. **'all' = mean over all prompts** (current default in script 50 / 52 / 53).

| encoder | k=1 | k=2 | k=3 | k=5 | k=all | best |
|---|---|---|---|---|---|---|
| gemma | 38% / 0.30 | 38% / 0.30 | 39% / 0.31 | 40% / 0.32 | **43% / 0.35** | k=all |
| gpt_oss_20b | 34% / 0.25 | 33% / 0.23 | 35% / 0.25 | **36% / 0.27** | 35% / 0.26 | k=5 |
| granite | 22% / 0.11 | 22% / 0.11 | 23% / 0.13 | 24% / 0.13 | **27% / 0.17** | k=all |
| ministral | 27% / 0.17 | **28% / 0.18** | 26% / 0.16 | 24% / 0.14 | 21% / 0.11 | k=2 |
| qwen | **20% / 0.11** | 19% / 0.10 | 19% / 0.10 | 18% / 0.09 | 18% / 0.09 | k=1 |

## Best-k per encoder

- **gemma**: no meaningful difference (baseline-all 43%, best 43%)
- **gpt_oss_20b**: **+0.8pp lift** at k=5 (baseline-all 35%, best 36%)
- **granite**: no meaningful difference (baseline-all 27%, best 27%)
- **ministral**: **+7.0pp lift** at k=2 (baseline-all 21%, best 28%)
- **qwen**: **+2.1pp lift** at k=1 (baseline-all 18%, best 20%)
