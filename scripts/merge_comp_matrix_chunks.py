"""
Merge parallel-chunk comp matrix outputs into a single canonical results file.

Designed for the 6-way parallel launch (2026-05-01):
    3 models (gpt55, gemini31, claude47) × 2 case chunks (0-681, 682-1363)

Each parallel job writes to:
    /results/comprehensive_matrix_partial_{model}_{chunk}.json   (every 100 cases)
    /results/comprehensive_matrix_results_{model}_{chunk}.json   (on completion)

This script handles ANY combination of complete + partial files:
- If a final file exists, prefer it
- Else use the latest partial
- Concatenate by (model, condition) and deduplicate by case_id

Usage (after parallel jobs complete or partial):
    # Pull every partial/results file from Modal volume
    modal volume get mira-results comprehensive_matrix_partial_gpt55_p1.json /tmp/cm_gpt55_p1.json
    modal volume get mira-results comprehensive_matrix_partial_gpt55_p2.json /tmp/cm_gpt55_p2.json
    modal volume get mira-results comprehensive_matrix_partial_gemini31_p1.json /tmp/cm_gemini31_p1.json
    modal volume get mira-results comprehensive_matrix_partial_gemini31_p2.json /tmp/cm_gemini31_p2.json
    modal volume get mira-results comprehensive_matrix_partial_claude47_p1.json /tmp/cm_claude47_p1.json
    modal volume get mira-results comprehensive_matrix_partial_claude47_p2.json /tmp/cm_claude47_p2.json

    # Merge
    python merge_comp_matrix_chunks.py \\
        --inputs /tmp/cm_gpt55_p1.json /tmp/cm_gpt55_p2.json \\
                 /tmp/cm_gemini31_p1.json /tmp/cm_gemini31_p2.json \\
                 /tmp/cm_claude47_p1.json /tmp/cm_claude47_p2.json \\
        --output /tmp/cm_merged.json

    # Resume helper: identify missing cases per model
    python merge_comp_matrix_chunks.py \\
        --inputs /tmp/cm_*.json \\
        --resume-report /tmp/missing_cases.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_chunk(path: Path) -> dict:
    """Load a chunk file, gracefully handling partial vs complete."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"WARN: could not parse {path}: {e}")
        return {}


def merge_chunks(chunk_paths: list[Path]) -> dict:
    """Merge N chunk files into a single canonical results dict.

    Per-condition arrays (triage_hits, triage_per_case, domain_scores, etc.)
    are concatenated. Per-case records are deduplicated by case_id.
    """
    merged: dict = {
        "benchmark": "comprehensive_matrix",
        "merged_from": [str(p) for p in chunk_paths],
        "n_chunks_merged": 0,
        "all_results": defaultdict(lambda: defaultdict(lambda: {
            "label": "",
            "triage_hits": [],
            "triage_per_case": [],
            "domain_scores": defaultdict(list),
            "calibration": {"pred_probs": [], "actual_labels": []},
            "confusion": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "omission_rates": [],
            "lnn_pass": 0, "lnn_fail": 0, "lnn_unknown": 0,
            "icd10_validity": [],
            "api_calls_total": 0,
            "cost_usd_total": 0.0,
            "latency_s": [],
            "by_difficulty": defaultdict(list),
            "by_specialty": defaultdict(list),
            "violations_by_cat": defaultdict(int),
            "rule_firing_counter": defaultdict(int),
            "case_ids_seen": [],
        })),
        "all_case_records": [],
        "case_ids_by_model": defaultdict(set),
    }

    for path in chunk_paths:
        chunk = load_chunk(path)
        if not chunk:
            continue
        merged["n_chunks_merged"] += 1

        # Aggregated all_results (per model × condition)
        for model_key, model_arms in (chunk.get("all_results") or {}).items():
            for cond_key, cond_data in model_arms.items():
                tgt = merged["all_results"][model_key][cond_key]
                tgt["label"] = cond_data.get("label", tgt["label"])
                for arr_field in ("triage_hits", "triage_per_case", "omission_rates",
                                  "icd10_validity", "latency_s"):
                    tgt[arr_field].extend(cond_data.get(arr_field, []) or [])
                for d, vals in (cond_data.get("domain_scores") or {}).items():
                    tgt["domain_scores"][d].extend(vals or [])
                conf = cond_data.get("confusion") or {}
                for k in ("tp", "fp", "tn", "fn"):
                    tgt["confusion"][k] += conf.get(k, 0)
                cal = cond_data.get("calibration") or {}
                tgt["calibration"]["pred_probs"].extend(cal.get("pred_probs", []) or [])
                tgt["calibration"]["actual_labels"].extend(cal.get("actual_labels", []) or [])
                for k in ("lnn_pass", "lnn_fail", "lnn_unknown",
                          "api_calls_total", "cost_usd_total"):
                    tgt[k] = tgt.get(k, 0) + cond_data.get(k, 0)
                # Per-difficulty / per-specialty hit lists
                for d, vals in (cond_data.get("by_difficulty") or {}).items():
                    tgt["by_difficulty"][d].extend(vals or [])
                for s, vals in (cond_data.get("by_specialty") or {}).items():
                    tgt["by_specialty"][s].extend(vals or [])
                # Violations + rule firing counters (sum)
                for k, v in (cond_data.get("violations_by_cat") or {}).items():
                    tgt["violations_by_cat"][k] = tgt["violations_by_cat"].get(k, 0) + v
                for r, n in (cond_data.get("rule_firing_counter") or {}).items():
                    tgt["rule_firing_counter"][r] = tgt["rule_firing_counter"].get(r, 0) + n

        # Per-case records (if present in chunk)
        for rec in chunk.get("per_case_records", []) or []:
            case_id = rec.get("case_id")
            model_key = rec.get("model_key", "unknown")
            if case_id and case_id in merged["case_ids_by_model"][model_key]:
                continue  # dedupe: same (model, case_id) already merged
            if case_id:
                merged["case_ids_by_model"][model_key].add(case_id)
            merged["all_case_records"].append(rec)

    # Convert defaultdict-of-set to dict-of-list for JSON serialization
    merged["case_ids_by_model"] = {
        m: sorted(s) for m, s in merged["case_ids_by_model"].items()
    }
    # Convert nested defaultdicts to plain dicts
    merged["all_results"] = json.loads(json.dumps(merged["all_results"], default=list))
    return merged


def report_missing_cases(chunk_paths: list[Path], total_cases: int = 1363) -> dict:
    """Report which (model, case_id) pairs are missing across all chunks.

    Used to construct a resume launch with only the missing cases.
    """
    seen_by_model: dict[str, set[str]] = defaultdict(set)
    for path in chunk_paths:
        chunk = load_chunk(path)
        if not chunk:
            continue
        for rec in chunk.get("per_case_records", []) or []:
            case_id = rec.get("case_id")
            model_key = rec.get("model_key", "unknown")
            if case_id and model_key != "unknown":
                seen_by_model[model_key].add(case_id)

    report = {
        "expected_total_per_model": total_cases,
        "n_seen_by_model": {m: len(s) for m, s in seen_by_model.items()},
        "n_missing_by_model": {
            m: total_cases - len(s) for m, s in seen_by_model.items()
        },
        "missing_case_ids_by_model": {
            m: "(set computation deferred — needs canonical case-ID list)"
            for m in seen_by_model
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel comp matrix chunks")
    parser.add_argument("--inputs", nargs="+", required=True, help="chunk JSON paths")
    parser.add_argument("--output", help="merged JSON output path")
    parser.add_argument("--resume-report", help="missing-case JSON output path")
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    print(f"Loading {len(paths)} chunk file(s):")
    for p in paths:
        size = p.stat().st_size if p.exists() else 0
        print(f"  {p}  ({size:,} bytes)")

    if args.output:
        merged = merge_chunks(paths)
        Path(args.output).write_text(json.dumps(merged, indent=2, default=list))
        print(f"\nMerged {merged['n_chunks_merged']} chunks -> {args.output}")
        print(f"  Models present: {list(merged['all_results'].keys())}")
        for m, cases in merged["case_ids_by_model"].items():
            print(f"  {m}: {len(cases)} unique case_ids")
        print(f"  Total per_case_records: {len(merged['all_case_records'])}")

    if args.resume_report:
        report = report_missing_cases(paths)
        Path(args.resume_report).write_text(json.dumps(report, indent=2))
        print(f"\nResume report -> {args.resume_report}")
        for m, n in report["n_missing_by_model"].items():
            print(f"  {m}: {n} cases missing of {report['expected_total_per_model']}")


if __name__ == "__main__":
    main()
