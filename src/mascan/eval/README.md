# Evaluation

MAScan has two evaluation workflows. Both read cases from `eval_papers/` and
write results to `eval_results/`. The `-pre` targets preview the commands
without making paid API calls; the plain targets add `--execute` and run for
real.

Set `OPENAI_API_KEY` and `EVAL_JUDGE_MODEL` in `.env` before running paid
evaluations.

## Gold-standard evaluation

Compares MAScan against a baseline on real PESTEL case reports, using an LLM
judge and human raters.

```bash
make gold-eval-pre     # preview only, no API calls
make gold-eval         # run responses, judge, and build the human packet
# ... human raters fill in the CSV files ...
make gold-eval-post    # score after raters return their CSVs
```

## Market-scenario evaluation

A small three-case scenario comparison (MAScan vs zero-shot), with no human
step.

```bash
make market-scenario-eval-pre   # preview only
make market-scenario-eval       # run for real
```

## Files

| Path | Content |
|---|---|
| `eval_papers/` | Case PDFs, manifests, and case definitions |
| `eval_results/` | Generated responses, judge output, and human packets |
| This folder (`src/mascan/eval/`) | Evaluation code (costing, exports, analysis) |
