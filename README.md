# anchor-public

Public methodology code for an evaluation of a clinical AI verification layer.

This repository contains the analytic and evaluation utilities sufficient for an independent reviewer to verify how the evaluation pipeline was constructed. The companion paper, supplement, results, figures, and tables are not part of this repository.

## What is in this repository

| Component | Released here |
|---|---|
| Statistical analysis utilities (Wilson and Newcombe Method 10 intervals; paired bootstrap CIs; κ CIs) | ✅ `scripts/compute_cis.py` |
| Comp-matrix chunk merger | ✅ `scripts/merge_comp_matrix_chunks.py` |
| Strip-before-grade response normalizer | ✅ `scripts/strip_anchor_annotations.py` |
| Bare-LLM and deployment-equivalent guardrail prompt specifications (the two non-proprietary conditions) | ✅ `prompts/` |

## What is *not* in this repository

| Component | Reason |
|---|---|
| Verification-layer rule library, specialist-agent prompts, concept-decomposition template, production source code | Proprietary; not released |
| Patient-level data | Protected health information; not released |
| Per-case raw model responses and judge adjudications | Confidential; available to journal-editor-designated reviewers under NDA on request |
| Aggregate result tables, figures, manuscript, supplement, checklists | Withheld pending submission outcome |

## Setup

```bash
git clone https://github.com/sanjaybasu/anchor-public.git
cd anchor-public
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## What the scripts do

- **`scripts/compute_cis.py`** — Wilson and Newcombe Method 10 confidence intervals for binary proportions and paired differences; paired-bootstrap percentile intervals (1,000 resamples, locked seed); McNemar exact P-values from discordant counts.
- **`scripts/merge_comp_matrix_chunks.py`** — Merges parallel-chunk evaluation outputs (per-substrate × per-chunk JSON files) into a single canonical results file with deterministic ordering.
- **`scripts/strip_anchor_annotations.py`** — Strips operational verification-layer annotations from a response prior to grading, preserving only the upstream-LLM clinical recommendation. Used to control for length / format effects between conditions.

## Prompt files

- **`prompts/bare_prompt.txt`** — bare frontier-model condition (no system prompt; only the structured-response directive).
- **`prompts/safety_prompt.txt`** — deployment-equivalent guardrail system prompt; operationally equivalent to the safety-prompt structures cited in the deployment-equivalent guardrail stack adopted at major U.S. health-system AI assurance programmes.

## License

MIT (see `LICENSE`).
