# Market-scenario evaluation — 3 special cases

Focused MAScan vs zero-shot evaluation on three instructor scenarios from `eval_papers/market-analysis-scenarios.pdf`:

| Case ID | Topic |
|---|---|
| `market_evonik` | Evonik — European specialty chemicals competitiveness |
| `market_volkswagen` | Volkswagen — strategic transformation |
| `market_biontech` | BioNTech — post-COVID oncology strategy |

**Same judge rubric** as the 25-case gold standard, with prompts that also require a strategic roadmap (investments, risks/opportunities, rough cost estimates).

**No human calibration** — LLM judge + paired stats only.  
**Separate output directory:** `eval_results/market_scenarios/` (does not overwrite the main 25-case results).

---

## 0. Prerequisites

Same setup as the main eval (see `docs/EVAL_25_CASES.md` §0):

```powershell
uv sync
# .env with OPENAI_API_KEY, FIRECRAWL_API_KEY, EVAL_JUDGE_MODEL=gpt-4o
$env:PYTHONPATH = "src"
```

Required files (already in repo or untracked locally):

- `eval_papers/market_scenario_cases.json`
- `eval_papers/market_scenario_manifest.json`
- `eval_papers/market-analysis-scenarios.pdf`

---

## 1. Preview commands (free)

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/run_market_scenario_eval.py
```

Or via Makefile:

```bash
make market-scenario-eval-pre
```

This prints the planned steps and writes nothing to paid APIs.

---

## 2. Run the full 3-case evaluation (paid)

One command runs pricing init (if needed) and the complete pre-human pipeline:

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/run_market_scenario_eval.py --execute --init-pricing
```

Or via Makefile:

```bash
make market-scenario-eval
```

**Resume a partial run:**

```powershell
uv run python scripts/run_market_scenario_eval.py --execute --skip-existing
```

**Bypass preflight** (only if you know checks are satisfied):

```powershell
uv run python scripts/run_market_scenario_eval.py --execute --skip-preflight
```

---

## 3. What gets produced

All outputs live under `eval_results/market_scenarios/`:

| Output | Purpose |
|---|---|
| `model_pricing.json` | Token pricing snapshot |
| `preflight.md` / `preflight.json` | Preflight report |
| `responses_mascan.json` | MAScan outputs (3 cases) |
| `responses_zero_shot.json` | Zero-shot outputs |
| `responses_all.json` | Merged responses |
| `judged_all.json` | LLM judge scores |
| `judged_all_priced.json` | Scores + cost |
| `system_summary.json` | Per-system aggregates |
| `mascan_vs_zero_shot.json` | Paired comparison |
| `case_trace.csv` / `case_trace.json` | Per-case trace |
| `experiment_report.md` | Final report |

---

## 4. Manual step-by-step (alternative)

If you prefer running the underlying scripts directly:

### 4a. Initialize pricing

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/init_gold_pricing.py `
  --manifest eval_papers/market_scenario_manifest.json `
  --source-url https://openai.com/api/pricing/ `
  --captured-at 2026-07-14 `
  --notes "Market-scenario mini eval (3 cases)."
```

Output: `eval_results/market_scenarios/model_pricing.json`

### 4b. Run pre-human pipeline

```powershell
uv run python scripts/run_gold_pre_human.py `
  --manifest eval_papers/market_scenario_manifest.json `
  --preflight-out eval_results/market_scenarios/preflight.json `
  --preflight-markdown-out eval_results/market_scenarios/preflight.md `
  --trace-csv-out eval_results/market_scenarios/case_trace.csv `
  --execute
```

Note: the market manifest has **no** `human_calibration` block, so human packet export is skipped automatically.

---

## 5. Read results

```powershell
Get-Content eval_results/market_scenarios/experiment_report.md
Get-Content eval_results/market_scenarios/mascan_vs_zero_shot.json | ConvertFrom-Json
```

---

## Differences from the 25-case eval

| | 25-case gold eval | 3-case market scenarios |
|---|---|---|
| Manifest | `gold_experiment_manifest.two_system.json` | `market_scenario_manifest.json` |
| Cases | 25 frozen papers | 3 instructor scenarios |
| Output dir | `eval_results/` | `eval_results/market_scenarios/` |
| Human raters | Yes (5×5) | No |
| Post-human script | `run_gold_post_human.py` | Not used |
