# Verification layer and clinical AI safety: evaluation pipeline (revision 1)

Code for the evaluation reported in "Impact of a neurosymbolic verification layer on AI safety in medical deployment: a multi-cohort evaluation" (*Artificial Intelligence in Health*, manuscript AIH026370137). The paper's Notes S1, S3, S7, S8, and S9 reproduce every prompt verbatim.

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Subject responses | `code/aih_r1_batch.py::subjects` | Runs GPT-6 Astra (OpenAI Flex), Gemini 3.1 Pro (Gemini Batch Mode), and Claude Opus 5.5 (Anthropic Message Batches) on each case under the bare configuration, configuration A, and configuration B (Cohort 1c), the bare configuration and configuration A (Cohort 2), and five prompt-directive compositions (Cohort 3). Up to 16,000 output tokens; provider default sampling. |
| 2. Verification framework | not released | Runs the verification framework on each configuration A response and appends its findings (appended placement). The framework source and its clinical-rule library are proprietary; the agent and fact-extraction prompts are in Note S8 of the paper. |
| 3. Rewrite placement | `code/aih_r1_batch.py::rewrites` | The same model receives its configuration A response and the same findings and returns a rewritten response. |
| 4. Scoring | `code/aih_r1_scoring.py` | Three LLM judges (one per developer) score checklist items (omission) and commission; a single judge lists and matches individual findings. |
| 5. Analysis | `code/aih_r1_analysis.py` | Computes every reported quantity from the per-case records and judgments and writes one results file. |
| Recovery | `code/aih_r1_harvest.py` | Collects completed results from a duplicate Anthropic batch for request keys that have no result yet. |

Batch identifiers and request maps are written to the results volume before a job waits on a batch, so a restarted container reattaches to the same batch instead of submitting it again.

## Inputs released here

- `inputs/cohort1c_case_ids.json`: the 264 Cohort 1c case identifiers. Case text is not released, because the parent corpus includes de-identified patient messages.
- `inputs/cohort2_probes.json`: the 120 commission probes with their failure class and harm signal. Three board-certified internists wrote the probes; they contain no patient data.
- `inputs/cohort3_probe_ids.json`: the 60 Cohort 3 probe identifiers (a subset of Cohort 2).
- `inputs/determinism_case_ids.json`: the 30 cases of the determinism subsample (seed 20261015).

## Not released

Per-case model responses, verification findings, and judge outputs; case text; the verification-framework source and clinical-rule library. These are available to journal-designated reviewers on reasonable request.

## Figures

`figures/` holds the scripts for Figures 1-3 and the graphical abstract. Each script reads the aggregate results file (`CANONICAL_NUMBERS_R1.json`) and contains no typed values.

## Requirements

Python 3.11; `modal`, `openai`, `anthropic`, `requests`, `matplotlib`. The batch and scoring scripts run on Modal and read provider keys from Modal secrets named `openai`, `anthropic`, and `google`.
