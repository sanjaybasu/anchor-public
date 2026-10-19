# Verification feedback and self-revision: evaluation pipeline (revision 1)

Code for the evaluation reported in "Verification feedback and self-revision for omissions in clinical recommendations by large language models: a paired evaluation" (*Artificial Intelligence in Health*, manuscript AIH026370137). Each first response of three LLMs is delivered in six versions, which cross three kinds of feedback (none; the full findings of a rule-and-agent verification framework; case-grounded calibrated alerts) with two deliveries (appended for the clinician, or returned to the model for revision). The paper's Notes S3 and S7 to S11 reproduce every prompt verbatim.

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. First responses and guardrail configurations | `code/aih_r1_batch.py` (`subjects`, `subjects_v2`, `replicate`) | Runs GPT-6 Astra, Gemini 3.1 Pro, and Claude Opus 5.5 on each case under the bare configuration, configuration A, configuration B, and configuration A with the provider's search tool (Cohort 1c); the bare configuration, configuration A, and configuration A with search (Cohort 2); and five compositions of four prompt directives (Cohort 3). A second, independent configuration A run is the resampling control. Up to 16,000 output tokens; provider default sampling. |
| 2. Full findings | not released | Runs the rule-and-agent verification framework on each configuration A response and appends its findings. The framework source and its clinical-rule library are proprietary; Note S8 reproduces its agent and fact-extraction prompts and fixed texts. |
| 3. Calibrated alerts | `code/aih_r1_v2.py`, `code/aih_r1_v2_select.py` | Selects the case's facts and the response's actions and final diagnoses (GPT-5.5), evaluates the clinical rules as Boolean logic, collects blind drafts of required actions from the two other models, groups and matches the candidate actions against the response (GPT-5.5), sets each model's alert threshold by conformal risk control on the development cases, and writes at most two alerts per response. `inputs/calibrated_alerts_split.json` holds the development and test case identifiers; `inputs/calibrated_alerts_config.json` holds the chosen weights, number of alerts, and thresholds. |
| 4. Revision and self-revision | `code/aih_r1_batch.py` (`rewrites_fx`, `rewrites_v2`, `followups_v2`, `regen`) | The same model receives its configuration A response and the feedback, or no feedback (self-revision), with a fixed instruction to revise, and returns a revised response. |
| 5. Scoring | `code/aih_r1_scoring.py` | Three LLM judges, one per developer, score each delivered text item by item against the case's checklist (omission) or on the commission rubric; a single judge maps findings statements and alerts to checklist items. |
| 6. Physician review | `code/aih_r1_probes_v2.py`, `code/aih_r1_physician_agreement.py` | Applies the pre-specified retention rule to the physicians' probe ratings and builds the reviewed probe set; computes physician agreement with the judges on the sampled texts. Ratings are never filled in by code. |
| 7. Analysis | `code/aih_r1_analysis.py` | Computes every reported quantity from the per-case records and judgments and writes one results file. |
| Diagnostics | `code/aih_r1_certificate_diagnostics.py` | Counts the rule library's evaluable rules and the evaluator parameters that the trained checkpoint covers (Note S12). |
| Recovery | `code/aih_r1_harvest.py` | Collects completed results from a duplicate Anthropic batch for request keys that have no result yet. |
| Table 2 quotes | `code/aih_r1_table2_quotes.py` | Quotes, verbatim, the passage of each Table 2 text that addresses the item; the table keeps a quote only if it appears verbatim. |

Batch identifiers and request maps are written to the results volume before a job waits on a batch, so a restarted container reattaches to the same batch instead of submitting it again. A provider refusal is a final outcome and is not requested again.

## Inputs released here

- `inputs/cohort1c_case_ids.json`: the 264 Cohort 1c case identifiers. Case text is not released, because the parent corpus includes de-identified patient messages.
- `inputs/calibrated_alerts_split.json`: the development (88) and test (176) case identifiers.
- `inputs/cohort2_probes.json`: the 120 commission probes as assembled, with their failure class and harm signal.
- `inputs/cohort2_probes_reviewed.json`: the 118 probes that three board-certified internists retained after review, with the edits that they made; revisions of the assembled probes were drafted with AI assistance and then reviewed. The probes are synthetic scenarios and contain no patient data.
- `inputs/cohort3_probe_ids.json` and `inputs/cohort3_probe_ids_reviewed.json`: the Cohort 3 probe identifiers before and after review (60 and 58).
- `inputs/determinism_case_ids.json`: the 30 cases of the determinism subsample (seed 20261015).

## Not released

Per-case model responses, verification findings, alerts, and judge outputs; case text; the physicians' workbooks; the verification-framework source and clinical-rule library. These are available to journal-designated reviewers on reasonable request.

## Results

`CANONICAL_NUMBERS_R1.json` holds every reported quantity (aggregate counts, proportions, means, and confidence intervals; no case text and no per-case outputs). It is written by `code/aih_r1_analysis.py`, and the manuscript, supplement, tables, and figures read their values from it.

## Figures

`figures/` holds the scripts for Figures 1-3 and the graphical abstract. Each script reads the aggregate results file (`CANONICAL_NUMBERS_R1.json`) and contains no typed values.

## Requirements

The runs used Python 3.11 on Modal with `modal`, `openai`, `anthropic`, and `requests`, reading provider keys from Modal secrets named `openai`, `anthropic`, and `google`; the analysis and figures used Python 3.12 with `matplotlib`, `python-docx`, and `openpyxl`.
