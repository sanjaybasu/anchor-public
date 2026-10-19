"""Build the physician-reviewed commission probes (Cohorts 2 and 3) from the returned rater workbooks.

Step 1 (python aih_r1_probes_v2.py decide [--fetch]): read the returned rater workbooks in adjudication/returned/
(Rater_<X>.xlsx; with --fetch, first download the Google Sheets in adjudication/drive_links.json as .xlsx with the
local OAuth token), read tab "3 Probes", and apply the pre-specified rule (Note S10.3; plan A15):
a probe is retained when a majority of raters chose "Accept" or "Accept with edits" and a majority answered "Yes"
to whether the harm signal names an action they would not take. Writes adjudication/probe_decisions.json and
adjudication/probe_edits_for_review.md, which lists every rater edit on a retained probe.

Step 2 (S.B.): resolve the edits and record the final text of each retained probe that received an applied edit in
adjudication/probe_edits_applied.json: {"<code>": {"case_text": "...", "harm_signal": "...", "note": "..."}}.
A retained probe with no entry keeps its drafted text; an entry with "applied": false records that S.B. reviewed the
rater comments and applied no edit. Ratings and edits are never filled in by code.

Step 3 (python aih_r1_probes_v2.py build): write cohort2_probes_v2.json, cohort3_probes_v2.json (the retained subset
of the 60 Cohort 3 probes; a replacement draft maps to its original by original text), probe_review_summary.json
(counts for Note S4), and adjudication/data_S1_probes.csv (Data S1).
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "results_aih_r1"
ADJ = DATA / "adjudication"
DRAFTS = ADJ / "probe_drafts"
TOKEN = Path.home() / ".config" / "slides-comments" / "token.json"
CLASSES = ("ddi", "lost_in_middle_lab", "wrong_treatment", "fake_citation")
CODES = {"ddi": "DDI", "lost_in_middle_lab": "LAB", "wrong_treatment": "TRT", "fake_citation": "CIT"}


def fetch_rater_workbooks() -> dict:
    """Download each rater sheet as .xlsx into adjudication/returned/ and return {rater: path}."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    links = json.load(open(ADJ / "drive_links.json"))
    creds = Credentials.from_authorized_user_file(str(TOKEN))
    drive = build("drive", "v3", credentials=creds)
    out_dir = ADJ / "returned"
    out_dir.mkdir(exist_ok=True)
    paths = {}
    for rater, url in links.items():
        if not rater.startswith("Rater"):
            continue
        fid = url.split("/d/")[1].split("/")[0]
        req = drive.files().export_media(
            fileId=fid, mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        path = out_dir / f"{rater.replace(' ', '_')}.xlsx"
        with open(path, "wb") as fh:
            dl = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
        paths[rater] = path
    return paths


def read_probe_tab(path: Path) -> dict:
    """{code: {"decision", "harm", "edits"}} from tab '3 Probes' (columns A, I, J, K)."""
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True)["3 Probes"]
    rows = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r or not r[0]:
            continue
        rows[str(r[0]).strip()] = {"decision": (str(r[8]).strip() if r[8] else ""),
                                   "harm": (str(r[9]).strip() if r[9] else ""),
                                   "edits": (str(r[10]).strip() if r[10] else "")}
    return rows


def drafts_by_code() -> dict:
    out = {}
    for cls in CLASSES:
        for i, p in enumerate(json.load(open(DRAFTS / f"revised_{cls}.json")), 1):
            out[f"{CODES[cls]}-{i:02d}"] = {**p, "class": cls}
    return out


def decide():
    paths = (fetch_rater_workbooks() if "--fetch" in sys.argv else
             {p.stem.replace("_", " "): p for p in sorted((ADJ / "returned").glob("Rater_*.xlsx"))})
    ratings = {rater: read_probe_tab(p) for rater, p in paths.items()}
    drafts = drafts_by_code()
    n_raters = len(ratings)
    majority = n_raters // 2 + 1
    decisions, review = {}, []
    for code, d in drafts.items():
        rs = {rater: ratings[rater].get(code, {}) for rater in ratings}
        accept = sum(1 for v in rs.values() if v.get("decision", "").lower().startswith("accept"))
        harm_yes = sum(1 for v in rs.values() if v.get("harm", "").lower() == "yes")
        complete = all(v.get("decision") and v.get("harm") for v in rs.values())
        retained = complete and accept >= majority and harm_yes >= majority
        edits = {rater: v.get("edits") for rater, v in rs.items() if v.get("edits")}
        decisions[code] = {"case_id": d["case_id"], "class": d["class"], "ratings": rs, "n_accept": accept,
                           "n_harm_yes": harm_yes, "complete": complete, "retained": retained, "edits": edits}
        if retained and edits:
            review += [f"## {code} ({d['class']}; {d['case_id']})", "", f"*Drafted probe.* {d['revised_text']}", "",
                       f"*Drafted harm signal.* {d['revised_harm_signal']}", ""]
            review += [f"*{rater} ({rs[rater].get('decision')}; harm signal: {rs[rater].get('harm')}).* {e}"
                       for rater, e in edits.items()]
            review.append("")
    incomplete = [c for c, v in decisions.items() if not v["complete"]]
    json.dump({"n_raters": n_raters, "majority": majority, "probes": decisions},
              open(ADJ / "probe_decisions.json", "w"), indent=1)
    (ADJ / "probe_edits_for_review.md").write_text(
        "# Rater edits on retained probes, for resolution by S.B.\n\nRecord the final text of each probe that "
        "receives an applied edit in probe_edits_applied.json.\n\n" + "\n".join(review) + "\n")
    print(f"raters={n_raters} retained={sum(v['retained'] for v in decisions.values())} "
          f"dropped={sum(not v['retained'] for v in decisions.values())} incomplete={len(incomplete)} "
          f"retained_with_edits_to_review={sum(1 for v in decisions.values() if v['retained'] and v['edits'])}")
    if incomplete:
        print("incomplete ratings:", ", ".join(incomplete[:20]))


def build():
    dec = json.load(open(ADJ / "probe_decisions.json"))["probes"]
    applied_path = ADJ / "probe_edits_applied.json"
    applied = json.load(open(applied_path)) if applied_path.exists() else {}
    pending = [c for c, v in dec.items() if v["retained"] and v["edits"] and c not in applied]
    if pending:
        raise SystemExit(f"edits not yet resolved for {len(pending)} retained probes: {', '.join(pending[:15])}")
    drafts = drafts_by_code()
    orig2 = {p["case_id"]: p for p in json.load(open(DATA / "cohort2_probes.json"))}
    orig_by_text = {p["case_text"].strip(): p for p in orig2.values()}
    c3_ids = {p["case_id"] for p in json.load(open(DATA / "cohort3_probes.json"))}
    v2, v3, rows, counts = [], [], [], Counter()
    for code, v in dec.items():
        d = drafts[code]
        orig = orig2.get(d["case_id"]) or orig_by_text.get((d.get("original_text") or "").strip())
        if orig is None:
            raise SystemExit(f"{code}: cannot map draft {d['case_id']} to an original probe")
        if not v["retained"]:
            counts["dropped"] += 1
            rows.append([code, d["class"], d["case_id"], orig["case_id"], "dropped", "", "", "", "", ""])
            continue
        a = applied.get(code)
        note = (a or {}).get("note", "")
        if a and not a.get("applied", True):
            a = None                                    # reviewed by S.B.; no edit applied
        counts["accepted_with_edits" if a else "accepted"] += 1
        text = (a or {}).get("case_text") or d["revised_text"]
        harm = (a or {}).get("harm_signal") or d["revised_harm_signal"]
        # a nonexistent-citation probe records its evidence of non-existence as 'verification'
        src = d.get("source") or d.get("verification") or ""
        src = "; ".join(map(str, src)) if isinstance(src, list) else str(src)
        pid = d["case_id"] + ("_r1" if a else "")      # an edited probe is a new probe and is run again
        rec = {**{k: orig[k] for k in orig if k not in ("case_id", "case_text", "harm_signal", "name")},
               "case_id": pid, "name": pid.split(":")[-1], "case_text": text, "harm_signal": harm,
               "category": d["class"], "source": f"commission_probe:{d['class']}", "original_case_id": orig["case_id"],
               "correct_action": d.get("correct_action", ""), "evidence_source": src}
        v2.append(rec)
        if orig["case_id"] in c3_ids:
            v3.append(rec)
        rows.append([code, d["class"], pid, orig["case_id"], "retained, edited by raters" if a else "retained",
                     text, harm, d.get("correct_action", ""), src, note])
    json.dump(v2, open(DATA / "cohort2_probes_v2.json", "w"), indent=1)
    json.dump(v3, open(DATA / "cohort3_probes_v2.json", "w"), indent=1)
    summary = {"n_reviewed": len(dec), "accepted": counts["accepted"],
               "accepted_with_edits": counts["accepted_with_edits"], "dropped": counts["dropped"],
               "n_cohort2": len(v2), "n_cohort3": len(v3),
               "cohort2_by_class": Counter(p["category"] for p in v2),
               "cohort3_by_class": Counter(p["category"] for p in v3)}
    json.dump(summary, open(DATA / "probe_review_summary.json", "w"), indent=1)
    with open(ADJ / "data_S1_probes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "class", "probe_id", "original_probe_id", "status", "probe_text", "harm_signal",
                    "correct_action", "source", "edit_note"])
        w.writerows(rows)
    print(json.dumps(summary, indent=1))


def draft():
    """All 120 drafted probes in their drafted form (cohort2_probes_v2.json, cohort3_probes_v2.json), so that the
    Cohort 2 and 3 runs can start before the physician review returns. 'build' later replaces both files with the
    retained probes; the analysis reads only the probe identifiers in the final files."""
    drafts = drafts_by_code()
    orig2 = {p["case_id"]: p for p in json.load(open(DATA / "cohort2_probes.json"))}
    orig_by_text = {p["case_text"].strip(): p for p in orig2.values()}
    c3_ids = {p["case_id"] for p in json.load(open(DATA / "cohort3_probes.json"))}
    v2, v3 = [], []
    for code, d in drafts.items():
        orig = orig2.get(d["case_id"]) or orig_by_text.get((d.get("original_text") or "").strip())
        if orig is None:
            raise SystemExit(f"{code}: cannot map draft {d['case_id']} to an original probe")
        src = d.get("source") or d.get("verification") or ""
        rec = {**{k: orig[k] for k in orig if k not in ("case_id", "case_text", "harm_signal", "name")},
               "case_id": d["case_id"], "name": d["case_id"].split(":")[-1], "case_text": d["revised_text"],
               "harm_signal": d["revised_harm_signal"], "category": d["class"],
               "source": f"commission_probe:{d['class']}", "original_case_id": orig["case_id"],
               "correct_action": d.get("correct_action", ""),
               "evidence_source": "; ".join(map(str, src)) if isinstance(src, list) else str(src), "status": "draft"}
        v2.append(rec)
        if orig["case_id"] in c3_ids:
            v3.append(rec)
    json.dump(v2, open(DATA / "cohort2_probes_v2.json", "w"), indent=1)
    json.dump(v3, open(DATA / "cohort3_probes_v2.json", "w"), indent=1)
    print(f"draft probe files: Cohort 2 {len(v2)}, Cohort 3 {len(v3)}; by class {dict(Counter(p['category'] for p in v3))}")


if __name__ == "__main__":
    {"decide": decide, "build": build, "draft": draft}[sys.argv[1] if len(sys.argv) > 1 else "decide"]()
