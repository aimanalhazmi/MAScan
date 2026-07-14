# Gold-standard evaluation — 25 cases (MAScan vs zero-shot)

Step-by-step commands for the full experiment on the frozen 25-case dataset.

**Systems compared:** MAScan (`gpt-4o-mini`) vs zero-shot same model  
**LLM judge:** `gpt-4o` (via `EVAL_JUDGE_MODEL`)  
**Human calibration:** 5 raters × 5 cases (25 ratings total, disjoint assignments)

---

## 0. Prerequisites

From the repo root:

```powershell
uv sync
Copy-Item .env.example .env
# Edit .env: OPENAI_API_KEY, FIRECRAWL_API_KEY
# Use FIRECRAWL_API_KEY only — leave FIRECRAWL_API_URL commented out on Windows
```

Recommended `.env` values:

```env
OPENAI_MODEL_DEFAULT=gpt-4o-mini
EVAL_JUDGE_MODEL=gpt-4o
```

Run tests once (optional but recommended):

```powershell
$env:PYTHONPATH = "src"
uv run pytest tests/eval -q
```

---

## 1. Initialize model pricing

Required before the paid pipeline (creates `eval_results/model_pricing.json`):

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/init_gold_pricing.py `
  --manifest eval_papers/gold_experiment_manifest.two_system.json `
  --source-url https://openai.com/api/pricing/ `
  --captured-at 2026-07-13 `
  --notes "25-case gold eval: MAScan vs zero-shot gpt-4o-mini, judge gpt-4o"
```

To recreate the file:

```powershell
uv run python scripts/init_gold_pricing.py `
  --manifest eval_papers/gold_experiment_manifest.two_system.json `
  --overwrite
```

---

## 2. Preview the pre-human pipeline (free)

Shows every command that will run, without API calls:

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/run_gold_pre_human.py `
  --manifest eval_papers/gold_experiment_manifest.two_system.json `
  --reviewer-out-dir eval_results/human_reviewers `
  --trace-csv-out eval_results/case_trace.csv `
  --preflight-out eval_results/pre_human_preflight.json `
  --preflight-markdown-out eval_results/pre_human_preflight.md
```

Read `eval_results/pre_human_preflight.md` and fix any **errors** before continuing.

---

## 3. Run the pre-human phase (paid)

Collects responses, runs the LLM judge, prices results, builds stats, and exports human-rater files:

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/run_gold_pre_human.py `
  --manifest eval_papers/gold_experiment_manifest.two_system.json `
  --reviewer-out-dir eval_results/human_reviewers `
  --trace-csv-out eval_results/case_trace.csv `
  --preflight-out eval_results/pre_human_preflight.json `
  --preflight-markdown-out eval_results/pre_human_preflight.md `
  --execute
```

**Resume a partial run** (skip steps whose outputs already exist):

```powershell
uv run python scripts/run_gold_pre_human.py `
  --manifest eval_papers/gold_experiment_manifest.two_system.json `
  --reviewer-out-dir eval_results/human_reviewers `
  --trace-csv-out eval_results/case_trace.csv `
  --execute --skip-existing
```

### What this phase produces

| Output | Purpose |
|---|---|
| `eval_results/responses_mascan.json` | MAScan outputs (25 cases) |
| `eval_results/responses_zero_shot.json` | Zero-shot outputs |
| `eval_results/judged_all.json` | LLM judge scores |
| `eval_results/judged_all_priced.json` | Judge scores + token cost |
| `eval_results/system_summary.json` | Per-system aggregates |
| `eval_results/mascan_vs_zero_shot.json` | Paired comparison + p-value |
| `eval_results/gold_experiment_report.md` | Pre-human report |
| `eval_results/human_packet.json` | Human calibration packet (internal) |
| `eval_results/human_answer_key.json` | **Do not share with raters** |
| `eval_results/human_reviewers/` | Per-rater packet + rating template |

---

## 4. Re-export human reviewer files (optional)

If you changed export formatting after the main run:

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/export_human_reviewer_files.py `
  --packet eval_results/human_packet.json `
  --ratings-template eval_results/human_ratings_template.json `
  --out-dir eval_results/human_reviewers
```

Each rater gets:

- `rater_N_packet.md` — prompts, gold reference, anonymized responses A/B
- `rater_N_ratings.xlsx` — fill columns **H** (depth: 1/2/3) and **I** (correct: true/false)
- `rater_N_ratings.csv` — CSV alternative

Instructions: `eval_results/human_reviewers/HOW_TO_RATE.md`

**Share with raters:** packet + xlsx/csv only. **Never share** `human_answer_key.json`.

---

## 5. Collect human ratings

1. Distribute one packet + one rating file per rater.
2. Wait until all 5 files are returned (`rater_1` … `rater_5`).
3. Place completed files in `eval_results/human_reviewers/`.

---

## 6. Preview the post-human phase (free)

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/run_gold_post_human.py `
  --manifest eval_papers/gold_experiment_manifest.two_system.json `
  --ratings-csv `
    eval_results/human_reviewers/rater_1_ratings.xlsx `
    eval_results/human_reviewers/rater_2_ratings.xlsx `
    eval_results/human_reviewers/rater_3_ratings.xlsx `
    eval_results/human_reviewers/rater_4_ratings.xlsx `
    eval_results/human_reviewers/rater_5_ratings.xlsx `
  --preflight-out eval_results/post_human_preflight.json `
  --preflight-markdown-out eval_results/post_human_preflight.md
```

CSV files work too — use `.csv` instead of `.xlsx` in the paths above.

---

## 7. Run the post-human phase

Imports ratings, recomputes IRR, refreshes summaries and the final report:

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/run_gold_post_human.py `
  --manifest eval_papers/gold_experiment_manifest.two_system.json `
  --ratings-csv `
    eval_results/human_reviewers/rater_1_ratings.xlsx `
    eval_results/human_reviewers/rater_2_ratings.xlsx `
    eval_results/human_reviewers/rater_3_ratings.xlsx `
    eval_results/human_reviewers/rater_4_ratings.xlsx `
    eval_results/human_reviewers/rater_5_ratings.xlsx `
  --preflight-out eval_results/post_human_preflight.json `
  --preflight-markdown-out eval_results/post_human_preflight.md `
  --execute
```

### Post-human outputs

| Output | Purpose |
|---|---|
| `eval_results/human_ratings.json` | Merged rater scores |
| `eval_results/human_irr.json` | Inter-rater reliability |
| `eval_results/gold_experiment_report.md` | Updated final report |
| `eval_results/gold_methodology_appendix.md` | Methods appendix |
| `eval_results/readiness_report.json` | Readiness gate |

---

## Makefile shortcuts (Git Bash / WSL)

The Makefile targets use `gold_experiment_manifest.example.json` (3-system example). For the real 25-case two-system run, prefer the commands above.

```bash
make gold-eval-pre    # dry-run (example manifest)
make gold-eval        # execute pre-human (example manifest)
make gold-eval-post   # execute post-human (example manifest)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Preflight: missing pricing file | Run step 1 |
| Firecrawl DNS error on Windows | Comment out `FIRECRAWL_API_URL` in `.env`; use cloud key only |
| PermissionError writing CSV | Close the file in Excel/editor, rerun with `--skip-existing` |
| Partial run after interruption | Add `--skip-existing` to the same command |
