"""Select calibrated verification alerts from the candidate groups (redesigned verification, AIH026370137 R1).

A candidate group (aih_r1_v2.py match) that the response does not contain is scored by the sum, over its sources,
of the source's confidence: each drafter contributes its stated confidence, and the rule library contributes the
highest severity/10 of its fired rules, times a rule weight. A group that the response contains in part is scored
at a partial weight. For each model, the alert threshold is set by conformal risk control on the development cases:
the smallest threshold at which (n/(n+1)) R(threshold) + 1/(n+1) <= alpha, where R is the share of complete
configuration A responses (no High-grade item omitted, three-judge majority) whose highest group score exceeds the
threshold. A response receives at most K alerts: its highest-scoring groups above the threshold.

  python aih_r1_v2_select.py tune     grid over rule weight, partial weight, and K on the development cases
                                      (labels from aih_r1_v2.py label); writes results_aih_r1/v2_tuning.json
  python aih_r1_v2_select.py freeze --rule-weight W --partial-weight P --k K
                                      writes the frozen settings and thresholds (v2_config.json), the alerts for
                                      every case (v2_alerts_c1c_<model>.json), and the delivery records for the test
                                      cases (records_c1c_<model>__v2.jsonl: configuration A rows and appended rows)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from aih_r1_analysis import MODELS, item_majority, load_judgments

DATA = Path(__file__).resolve().parent / "results_aih_r1"
ALPHA = 0.10


def load_jsonl(p):
    out = []
    if p.exists():
        for line in open(p):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def guard_status(mk, cases):
    """{case_id: set of omitted High-grade item numbers (1-based)} for the configuration A response."""
    judg = load_judgments("", "c1c", mk)
    out = {}
    for cid, c in cases.items():
        items = c["checklist"]
        high = [i for i, it in enumerate(items) if it.get("grade") == "High"]
        if not high:
            continue
        v = item_majority(judg.get(f"{mk}|{cid}|guard|0", {}), len(items))
        if v is not None:
            out[cid] = {i + 1 for i in high if not v[i]}
    return out


def group_scores(rec, rule_w, partial_w):
    cands = rec["candidates"]
    scored = []
    for gi, g in enumerate(rec["groups"]):
        if g["in_response"] not in ("no", "partly"):
            continue
        by_src = defaultdict(float)
        best_reason = ("", -1.0)
        for m in g["members"]:
            c = cands[m - 1]
            src = "rule" if c["source"] == "rule" else c["source"]
            conf = float(c.get("confidence") or 0.0)
            by_src[src] = max(by_src[src], conf)
            if conf > best_reason[1]:
                best_reason = (c.get("reason", ""), conf)
        s = sum(v * (rule_w if k == "rule" else 1.0) for k, v in by_src.items())
        if g["in_response"] == "partly":
            s *= partial_w
        if s > 0:
            scored.append({"group": gi, "score": s, "action": g["action"], "reason": best_reason[0],
                           "sources": sorted(by_src)})
    scored.sort(key=lambda x: -x["score"])
    return scored


def conformal_threshold(max_scores_complete, alpha=ALPHA):
    """Smallest threshold t with (n/(n+1)) * mean(max_score > t) + 1/(n+1) <= alpha."""
    n = len(max_scores_complete)
    if n == 0:
        return float("inf")
    grid = sorted(set([0.0] + list(max_scores_complete)))
    for t in grid:
        r = sum(1 for s in max_scores_complete if s > t) / n
        if (n / (n + 1)) * r + 1 / (n + 1) <= alpha:
            return t
    return float("inf")


def load_all():
    cases = json.load(open(DATA / "cohort1c_cases_checklists.json"))
    split = json.load(open(DATA / "v2_split.json"))
    match = {mk: {r["case_id"]: r for r in load_jsonl(DATA / f"v2_match_c1c_{mk}.jsonl") if r.get("ok")} for mk, _ in MODELS}
    labels = {mk: {r["case_id"]: r["labels"] for r in load_jsonl(DATA / f"v2_label_c1c_{mk}.jsonl") if r.get("ok")}
              for mk, _ in MODELS}
    status = {mk: guard_status(mk, cases) for mk, _ in MODELS}
    return cases, split, match, labels, status


def evaluate(mk, ids, match, labels, status, rule_w, partial_w, k, thr):
    """Alerts, catches, and precision on the given cases (labels required for catches and precision)."""
    n_alert_resp = n_complete = n_complete_alert = n_alerts = n_hit = n_omitted = n_caught = 0
    for cid in ids:
        if cid not in match[mk] or cid not in status[mk]:
            continue
        sc = [g for g in group_scores(match[mk][cid], rule_w, partial_w) if g["score"] > thr][:k]
        om = status[mk][cid]
        n_omitted += len(om)
        complete = not om
        n_complete += complete
        if sc:
            n_alert_resp += 1
            n_complete_alert += complete
        lab = labels[mk].get(cid)
        if lab is not None:
            hit_items = {lab[g["group"]] for g in sc if g["group"] < len(lab)}
            n_alerts += len(sc)
            n_hit += sum(1 for g in sc if g["group"] < len(lab) and lab[g["group"]] in om)
            n_caught += len(hit_items & om)
    return {"alerted_responses": n_alert_resp, "complete": n_complete, "complete_alerted": n_complete_alert,
            "alerts": n_alerts, "alerts_on_omitted_high": n_hit, "omitted_high": n_omitted, "caught_high": n_caught}


def thresholds(match, status, dev, rule_w, partial_w, alpha=ALPHA):
    out = {}
    for mk, _ in MODELS:
        ms = []
        for cid in dev:
            if cid in match[mk] and cid in status[mk] and not status[mk][cid]:
                sc = group_scores(match[mk][cid], rule_w, partial_w)
                ms.append(sc[0]["score"] if sc else 0.0)
        out[mk] = conformal_threshold(ms, alpha)
    return out


def tune():
    cases, split, match, labels, status = load_all()
    dev = split["development"]
    rows = []
    for rule_w in (0.0, 0.5, 1.0):
        for partial_w in (0.0, 0.5, 1.0):
            thr = thresholds(match, status, dev, rule_w, partial_w)
            for k in (1, 2, 3):
                tot = defaultdict(int)
                per = {}
                for mk, _ in MODELS:
                    e = evaluate(mk, dev, match, labels, status, rule_w, partial_w, k, thr[mk])
                    per[mk] = e
                    for kk, vv in e.items():
                        tot[kk] += vv
                prec = tot["alerts_on_omitted_high"] / tot["alerts"] if tot["alerts"] else None
                rows.append({"rule_weight": rule_w, "partial_weight": partial_w, "k": k,
                             "thresholds": thr, "total": dict(tot), "precision": prec, "per_model": per})
    rows.sort(key=lambda r: (-r["total"]["caught_high"], -(r["precision"] or 0)))
    (DATA / "v2_tuning.json").write_text(json.dumps(rows, indent=1))
    for r in rows[:10]:
        t = r["total"]
        print(f"rule_w={r['rule_weight']} partial_w={r['partial_weight']} k={r['k']} | caught {t['caught_high']}/"
              f"{t['omitted_high']} | alerts {t['alerts']} precision {r['precision'] and round(r['precision'], 3)} | "
              f"complete alerted {t['complete_alerted']}/{t['complete']} | thr "
              f"{ {m: round(v, 2) for m, v in r['thresholds'].items()} }")


ALERT_HEADER = "--- Verification alerts ---\nThe verification found these actions missing from the response:"


def alerts_text(sc):
    lines = [ALERT_HEADER] + [f"{i + 1}. {g['action']}" + (f" ({g['reason']})" if g.get("reason") else "")
                              for i, g in enumerate(sc)]
    return "\n".join(lines)


def freeze(rule_w, partial_w, k):
    cases, split, match, labels, status = load_all()
    dev, test = split["development"], split["test"]
    thr = thresholds(match, status, dev, rule_w, partial_w)
    rf = json.load(open(DATA / "v2_rules_facts.json"))
    cfg = {"alpha": ALPHA, "rule_weight": rule_w, "partial_weight": partial_w, "k": k, "thresholds": thr,
           "calibration": "conformal risk control on development complete responses, per model",
           "n_rules_evaluable": sum(1 for r in rf["rules"] if r["evaluable"]), "n_rules": len(rf["rules"])}
    (DATA / "v2_config.json").write_text(json.dumps(cfg, indent=1))
    for mk, _ in MODELS:
        al = {}
        for cid, rec in match[mk].items():
            sc = [g for g in group_scores(rec, rule_w, partial_w) if g["score"] > thr[mk]][:k]
            al[cid] = sc
        (DATA / f"v2_alerts_c1c_{mk}.json").write_text(json.dumps(al, indent=1))
        guard = {}
        for r in load_jsonl(DATA / f"records_c1c_{mk}__fx.jsonl"):
            if r.get("ok") and r["arm"] == "guard" and int(r.get("rep", 0)) == 0:
                guard[r["case_id"]] = r
        rows = []
        for cid in test:
            g = guard.get(cid)
            if g is None:
                continue
            rows.append(json.dumps(g))
            if al.get(cid):
                txt = alerts_text(al[cid])
                rows.append(json.dumps({**{kk: g[kk] for kk in ("model_key", "model", "case_id", "source")},
                                        "arm": "append", "rep": 0, "ok": True,
                                        "text": (g.get("text") or "") + "\n\n" + txt,
                                        "findings": "[[RESPONSE]]\n\n" + txt, "n_alerts": len(al[cid])}))
        (DATA / f"records_c1c_{mk}__v2.jsonl").write_text("\n".join(rows) + "\n")
        n_al = sum(1 for cid in test if al.get(cid))
        print(f"{mk}: threshold {thr[mk]:.3f}; test responses with alerts {n_al}/{len(test)}")
    print("frozen:", cfg)


def apply(cohort):
    """Alerts for another cohort with the frozen settings and thresholds (no recalibration): writes
    v2_alerts_<cohort>_<model>.json and records_<cohort>_<model>__v2.jsonl (configuration A rows and appended rows)."""
    cfg = json.load(open(DATA / "v2_config.json"))
    for mk, _ in MODELS:
        match = {r["case_id"]: r for r in load_jsonl(DATA / f"v2_match_{cohort}_{mk}.jsonl") if r.get("ok")}
        al = {cid: [g for g in group_scores(rec, cfg["rule_weight"], cfg["partial_weight"])
                    if g["score"] > cfg["thresholds"][mk]][:cfg["k"]] for cid, rec in match.items()}
        (DATA / f"v2_alerts_{cohort}_{mk}.json").write_text(json.dumps(al, indent=1))
        # keep the rewrites already made from these alerts (the alerts of a matched case do not change)
        out_path = DATA / f"records_{cohort}_{mk}__v2.jsonl"
        kept = [json.dumps(r) for r in load_jsonl(out_path) if r.get("arm") not in ("guard", "append")
                and al.get(r["case_id"])] if out_path.exists() else []
        rows = []
        for r in load_jsonl(DATA / f"records_{cohort}_{mk}.jsonl"):
            if not (r.get("ok") and r["arm"] == "guard" and int(r.get("rep", 0)) == 0):
                continue
            rows.append(json.dumps(r))
            if al.get(r["case_id"]):
                txt = alerts_text(al[r["case_id"]])
                rows.append(json.dumps({**{kk: r[kk] for kk in ("model_key", "model", "case_id", "source")},
                                        "arm": "append", "rep": 0, "ok": True,
                                        "text": (r.get("text") or "") + "\n\n" + txt,
                                        "findings": "[[RESPONSE]]\n\n" + txt, "n_alerts": len(al[r["case_id"]])}))
        out_path.write_text("\n".join(rows + kept) + "\n")
        print(f"{cohort}/{mk}: {sum(1 for v in al.values() if v)} of {len(al)} probe responses with alerts; "
              f"{len(kept)} earlier rewrite rows kept")


def curve():
    """Alert metrics on the test cases at three false-alert budgets, with the frozen weights and K (labels from
    aih_r1_v2.py label on the test cases). Writes results_aih_r1/v2_curve.json."""
    cfg = json.load(open(DATA / "v2_config.json"))
    cases, split, match, labels, status = load_all()
    dev, test = split["development"], split["test"]
    out = {}
    for alpha in (0.10, 0.20, 0.30):
        thr = thresholds(match, status, dev, cfg["rule_weight"], cfg["partial_weight"], alpha)
        tot = defaultdict(int)
        for mk, _ in MODELS:
            e = evaluate(mk, test, match, labels, status, cfg["rule_weight"], cfg["partial_weight"], cfg["k"], thr[mk])
            for kk, vv in e.items():
                tot[kk] += vv
        def _wilson(k, n, z=1.959964):
            if not n:
                return [None, None]
            ph, d = k / n, 1 + z * z / n
            c, h = (ph + z * z / (2 * n)) / d, z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
            return [max(0.0, c - h), min(1.0, c + h)]
        out[f"alpha{round(alpha * 100):02d}"] = {"alpha": alpha, "thresholds": thr, **dict(tot),
                               "false_alert_rate_ci": _wilson(tot["complete_alerted"], tot["complete"]),
                               "precision_ci": _wilson(tot["alerts_on_omitted_high"], tot["alerts"]),
                               "catch_ci": _wilson(tot["caught_high"], tot["omitted_high"]),
                               "false_alert_rate": tot["complete_alerted"] / tot["complete"] if tot["complete"] else None,
                               "precision": tot["alerts_on_omitted_high"] / tot["alerts"] if tot["alerts"] else None,
                               "catch": tot["caught_high"] / tot["omitted_high"] if tot["omitted_high"] else None}
    (DATA / "v2_curve.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "thresholds"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "freeze", "curve", "apply"))
    ap.add_argument("--cohort", default="c2v2")
    ap.add_argument("--rule-weight", type=float, default=1.0)
    ap.add_argument("--partial-weight", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    {"tune": tune, "curve": curve, "apply": lambda: apply(a.cohort)}.get(
        a.cmd, lambda: freeze(a.rule_weight, a.partial_weight, a.k))()
