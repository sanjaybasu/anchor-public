"""AIH026370137 revision 1: collect completed results from a second Anthropic batch.

A container restart can submit the same subject requests twice. This script reads a finished
batch, maps each custom_id to its request with the same construction that aih_r1_batch.py uses
(cases-file order x arm order; r0, r1, ...), and appends a record only for keys that have no
successful record yet. It writes the same record fields as aih_r1_batch.py plus the batch id.
Run it only when no other job is writing the same records file.

  modal run aih_r1_harvest.py --cohort c1c --batch-id msgbatch_... [--arms bare,guard,guard_alt]
"""
import modal

app = modal.App("aih-r1-harvest")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("anthropic>=0.70.0", "modal")
       .add_local_python_source("aih_r1_batch"))
results_vol = modal.Volume.from_name("mira-results")
DEFAULT_ARMS = {"c1c": "bare,guard,guard_alt", "c2": "bare,guard",
                "c3": "pd_full,pd_no_safety,pd_no_filter,pd_no_grounding,pd_no_hazard"}


@app.function(image=img, secrets=[modal.Secret.from_name("anthropic")], volumes={"/results": results_vol},
              timeout=3600)
def harvest(cohort: str, batch_id: str, arms: str) -> dict:
    import json
    from datetime import datetime
    from pathlib import Path
    import anthropic
    import aih_r1_batch as B

    base = Path("/results/aih_r1")
    results_vol.reload()
    cases = json.load(open(base / B.COHORT_CASES[cohort]))
    by_id, i = {}, 0
    for c in cases:
        for arm in arms.split(","):
            by_id[f"r{i}"] = {"case_id": c["case_id"], "source": c.get("source", ""), "arm": arm,
                              "system": B._system_for(arm)}
            i += 1
    rec_path = base / f"records_{cohort}_claude55.jsonl"
    have = set()
    if rec_path.exists():
        for line in open(rec_path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ok"):
                have.add((r["case_id"], r["arm"], int(r.get("rep", 0))))
    client = anthropic.Anthropic()
    b = client.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        return {"error": f"batch {batch_id} is {b.processing_status}"}
    n_written, n_skipped, n_failed = 0, 0, 0
    with open(rec_path, "a") as fh:
        for res in client.messages.batches.results(batch_id):
            req = by_id[res.custom_id]
            key = (req["case_id"], req["arm"], 0)
            if res.result.type != "succeeded":
                n_failed += 1
                continue
            if key in have:
                n_skipped += 1
                continue
            msg = res.result.message
            text = "".join(getattr(bl, "text", "") or "" for bl in msg.content if getattr(bl, "type", "") == "text")
            rec = {"model_key": "claude55", "model": B.MODEL_IDS["claude55"], "case_id": req["case_id"],
                   "source": req["source"], "arm": req["arm"], "rep": 0, "ok": bool(text), "text": text,
                   "finish": {"provider": "anthropic", "stop_reason": msg.stop_reason,
                              "usage": msg.usage.model_dump() if msg.usage else None},
                   "latency_s": None, "system_prompt_chars": len(req["system"] or ""), "grounding": False,
                   "via": "batch", "batch_id": batch_id, "ts": datetime.utcnow().isoformat()}
            fh.write(json.dumps(rec, default=str) + "\n")
            have.add(key)
            n_written += 1
    results_vol.commit()
    return {"cohort": cohort, "batch": batch_id, "written": n_written, "already_present": n_skipped,
            "not_succeeded": n_failed}


@app.local_entrypoint()
def main(cohort: str, batch_id: str, arms: str = ""):
    print(harvest.remote(cohort, batch_id, arms or DEFAULT_ARMS[cohort]))
