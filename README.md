# anchor-public

Public reproducibility kit for **A formal verification layer for clinical large language models: a pre-registered retrospective evaluation and pragmatic randomized controlled trial in high-risk multidisciplinary care management** (manuscript under review, 2026).

This repository contains the analytic code and aggregate canonical result tables sufficient for an independent investigator to verify every reported number, figure, and table in the manuscript and supplementary appendix. It is the public-facing companion to the manuscript's **Code availability** and **Data availability** statements.

## Scope and what is *not* in this repository

| Layer | Released here? | Where it lives |
|---|---|---|
| Statistical analysis pipeline (Wilson / Newcombe Method 10 / paired bootstrap / κ CI / cost-effectiveness CI) | ✅ `scripts/compute_cis.py` | this repo |
| Aggregate canonical result tables (architectural argument, v3 PRIMARY, wt-rich, diagnostic test, self-review-vs-LNN) | ✅ `results/*.json` | this repo |
| Bare-LLM and deployment-equivalent safety-prompted prompt specifications (C1, C2 conditions) | ✅ `prompts/` | this repo |
| Strip-before-grade utility (per SAP §8.4.4 v3) | ✅ `scripts/strip_anchor_annotations.py` | this repo |
| Comp-matrix chunk merger (per SAP §8.4.4 v3) | ✅ `scripts/merge_comp_matrix_chunks.py` | this repo |
| Pre-registered Statistical Analysis Plan, OSF pre-registration archive, ClinicalTrials.gov registration template, CONSORT-AI / TRIPOD-LLM checklists | ✅ `docs/` (or via OSF DOI) | this repo / OSF |
| **ANCHOR rule library** (3,206 rules, weights, activation thresholds, formal Łukasiewicz expressions) | ❌ Withheld for IP protection | not released |
| **Specialist-agent prompt set** (5 agents) | ❌ Withheld for IP protection | not released |
| **Concept-decomposition template** | ❌ Withheld for IP protection | not released |
| **Production verification-layer source code** | ❌ Withheld for IP protection | not released |
| **Patient-level data** (Waymark integrated-network subset of Aim 1; Aim 2 prospective trial dataset) | ❌ Withheld for HIPAA / Waymark privacy practices | not released; aggregate trial results to ClinicalTrials.gov per Final Rule (42 CFR Part 11) |
| **Per-case raw model responses and judge adjudications** (n = 1,363 retrospective + n = 525 HealthBench Professional) | ⚠️ Confidential peer-review access only | available to Editor-designated reviewers under non-disclosure-agreement-protected access; contact corresponding author with the Editor's request |

The architectural-design specification of the verification layer (specialty distribution, guideline-body provenance, aggregate library statistics — sufficient for an independent investigator to reproduce the *framework* per Riegel et al. 2020) is provided in **Supplementary Appendix S2** of the manuscript.

## Reproducibility recipe

```bash
# 1. Clone
git clone https://github.com/sanjaybasu/anchor-public.git
cd anchor-public

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Recompute confidence intervals from canonical aggregate tables
python scripts/compute_cis.py --canonical

# 4. Verify reported numbers in the manuscript Tables 1–3 and Figures 1–3b
#    against the regenerated CI tables in results/.
```

## What `compute_cis.py` reproduces

Running `compute_cis.py --canonical` against the JSONs in `results/` regenerates:

- **Table 1** (calibration κ table; n = 110 cohort)
- **Table 2** (architectural argument: pooled and per-model LNN-vs-self-review with Newcombe 95% CIs and McNemar exact P-values; n = 200 hazard cohort)
- **Table 3** (v3 PRIMARY comp matrix C5 − C2 paired bootstrap CIs across the three frontier LLMs)
- **Figure 2** (architectural argument forest plot)
- **Figure 3a** (comp matrix forest plot)
- **Figure 3b** (ablation stacked-bar)
- **Supplementary Table S6** (wt-rich n = 1,371 cohort C5 − C2 by domain)

The script is deterministic with locked seeds (per SAP §8.19): `seed = 20260430` for primary bootstrap; `seed = 20260425` for κ CI bootstrap; `N_BOOTSTRAP = 1000`.

## File map

```
anchor-public/
├── README.md                  # this file
├── LICENSE                    # MIT
├── requirements.txt           # numpy, scipy
├── results/
│   ├── architectural_argument_cis_canonical.json
│   ├── self_review_vs_lnn_2026-04-26_canonical.json
│   ├── v3_primary_cis_table.json
│   ├── wtrich_canonical_summary.json
│   └── diagnostic_test_characterization_canonical.json
├── scripts/
│   ├── compute_cis.py
│   ├── merge_comp_matrix_chunks.py
│   └── strip_anchor_annotations.py
├── prompts/
│   ├── bare_prompt.txt        # C1 condition (no system prompt)
│   └── safety_prompt.txt      # C2 condition (deployment-equivalent guardrails)
└── docs/
    ├── statistical_analysis_plan.md
    ├── osf_preregistration.md
    ├── clinicaltrials_gov_registration.md
    └── tripod_llm_checklist.md
    # (CONSORT-AI checklist is provided as Supplementary Appendix S10 of the manuscript.)
```

## Citation

If you use this repository, please cite:

> Basu S, Patel S, Sheth P, Morgan J, Batniji R. *A formal verification layer for clinical large language models: a pre-registered retrospective evaluation and pragmatic randomized controlled trial in high-risk multidisciplinary care management*. (under review, 2026.)

## License

MIT (see `LICENSE`).

## Contact

Sanjay Basu, MD, PhD — sanjay.basu@ucsf.edu
