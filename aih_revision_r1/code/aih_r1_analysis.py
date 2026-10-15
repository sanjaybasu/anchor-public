"""AIH026370137 revision 1: analysis of the revision runs.

Reads, from results_aih_r1/ (local only; never committed):
  records_<prefix>c1c_<model>.jsonl, judgments_<prefix>c1c_<model>.jsonl   Cohort 1c
  records_<prefix>c2_<model>.jsonl,  judgments_<prefix>c2_<model>.jsonl    Cohort 2
  records_<prefix>c3_<model>.jsonl,  judgments_<prefix>c3_<model>.jsonl    Cohort 3
  judgments_<prefix>c1ccomp_<model>.jsonl  (optional; component attribution)
  cohort1c_cases_checklists.json, cohort2_probes.json

and writes every pre-specified quantity of REVISION_PLAN.md to one JSON file.

Run:
  python aih_r1_analysis.py --prefix ""         # full runs
  python aih_r1_analysis.py --prefix pilot_     # pilot runs
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "results_aih_r1"
OUT_DEFAULT = Path("/Users/sanjaybasu/waymark-local/notebooks/anchor/papers/retro/aih/"
                   "revision_1/CANONICAL_NUMBERS_R1.json")
MODELS = [("gpt6", "GPT-6 Astra"), ("gemini31", "Gemini 3.1 Pro"), ("claude55", "Claude Opus 5.5")]
# Judge that shares a developer with each subject model (dropped in the self-judge sensitivity analysis)
SELF_JUDGE = {"gpt6": "gpt-6-astra", "gemini31": "gemini-3.1-pro-preview", "claude55": "claude-opus-5-5"}
ARMS_C1 = ["bare", "guard", "guard_alt", "append", "rewrite"]
ARMS_C2 = ["bare", "guard", "append", "rewrite"]
ARMS_C3 = ["pd_full", "pd_no_safety", "pd_no_filter", "pd_no_grounding", "pd_no_hazard"]
OUT_OF_SCOPE_SOURCES = ("mmlu", "pubmedqa")
SEED = 20261015
N_BOOT = 2000
N_PERM = 10000
# Case-selection outputs from the checklist project (local; read only for the cohort-flow counts)
SELECTION = Path("/Users/sanjaybasu/waymark-local/notebooks/anchor/cross_model_generalization/run")


# ── generic statistics ─────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bh(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    prev = 1.0
    for rank in range(m, 0, -1):
        k, p = items[rank - 1]
        prev = min(prev, p * m / rank)
        adj[k] = min(1.0, prev)
    return adj


def cluster_boot_mean(vals_by_case: dict, n_boot: int = N_BOOT, seed: int = SEED):
    """Mean over all values, with a percentile CI from resampling cases."""
    cases = list(vals_by_case)
    if not cases:
        return None, None, None
    allv = [x for c in cases for x in vals_by_case[c]]
    est = sum(allv) / len(allv)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = [x for c in (rng.choice(cases) for _ in cases) for x in vals_by_case[c]]
        means.append(sum(s) / len(s))
    means.sort()
    return est, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def signflip_p(vals_by_case: dict, n_perm: int = N_PERM, seed: int = SEED) -> float:
    """Two-sided case-level sign-flip permutation test of a mean paired difference."""
    m = [sum(v) / len(v) for v in vals_by_case.values() if v]
    if not m:
        return 1.0
    obs = abs(sum(m) / len(m))
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        s = abs(sum(x if rng.random() < 0.5 else -x for x in m) / len(m))
        if s >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def fleiss_kappa(rows: list[list[int]]) -> float | None:
    """rows: per subject, counts per category (equal raters per subject)."""
    rows = [r for r in rows if sum(r) >= 2]
    if not rows:
        return None
    n = sum(rows[0])
    rows = [r for r in rows if sum(r) == n]
    N = len(rows)
    k = len(rows[0])
    p_j = [sum(r[j] for r in rows) / (N * n) for j in range(k)]
    P_i = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in rows]
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j)
    return None if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return None, None, None
    def q(p):
        i = (len(xs) - 1) * p
        lo, hi = math.floor(i), math.ceil(i)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
    return q(0.5), q(0.25), q(0.75)


def words(t: str) -> int:
    return len(re.findall(r"\S+", t or ""))


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def truncated(rec: dict) -> bool | None:
    f = rec.get("finish") or {}
    if not f:
        return None
    if f.get("provider") == "openai":
        return f.get("status") == "incomplete"
    if f.get("provider") == "google":
        return f.get("finish_reason") == "MAX_TOKENS"
    if f.get("provider") == "anthropic":
        return f.get("stop_reason") == "max_tokens"
    return None


def finish_label(rec: dict) -> str:
    """Normalized finish reason: stop, length, refusal, or other."""
    f = rec.get("finish") or {}
    prov = f.get("provider")
    if prov == "openai":
        if f.get("status") == "completed":
            return "stop"
        return "length" if f.get("incomplete_reason") == "max_output_tokens" else "other"
    if prov == "google":
        fr = f.get("finish_reason")
        return {"STOP": "stop", "MAX_TOKENS": "length"}.get(fr, "refusal" if fr in (
            "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION") else "other")
    if prov == "anthropic":
        sr = f.get("stop_reason")
        return {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                "refusal": "refusal"}.get(sr, "other")
    return "unknown"


def findings_flag_count(rec: dict) -> int:
    """Number of findings the appended verification output presents to the clinician."""
    meta = rec.get("framework_meta") or {}
    mods = meta.get("modifications_applied") or []
    lnn = sum(1 for m in mods if str(m).startswith(("omission_fix:", "contraindication:",
                                                      "anchoring:", "triage_override:")))
    txt = rec.get("findings") or ""
    known = re.findall(r"known:\s*(\d+)\s+agent-traceable", txt)
    disc = re.findall(r"discovered:\s*(\d+)\s+emergent", txt)
    agents = sum(int(x) for x in known) + sum(int(x) for x in disc)
    consensus = 1 if (meta.get("consensus_fired") or meta.get("pass_2_consensus_fired")) else 0
    return lnn + agents + consensus


# ── loaders ────────────────────────────────────────────────────────────────

def load_records(prefix: str, cohort: str, mk: str) -> dict:
    recs = {}
    for r in load_jsonl(DATA / f"records_{prefix}{cohort}_{mk}.jsonl"):
        if r.get("ok"):
            recs[(r["case_id"], r["arm"], int(r.get("rep", 0)))] = r
    return recs


def load_judgments(prefix: str, cohort: str, mk: str) -> dict:
    """Merge the per-judge files (judgments_<prefix><cohort>_<mk>__<judge>.jsonl)."""
    out = defaultdict(dict)
    paths = sorted(DATA.glob(f"judgments_{prefix}{cohort}_{mk}__*.jsonl"))
    for p in paths:
        for j in load_jsonl(p):
            if j.get("ok"):
                out[j["key"]][j["judge"]] = j["parsed"]
    return out


def item_majority(per_judge: dict, n_items: int):
    if not per_judge:
        return None
    vecs = [v for v in per_judge.values() if isinstance(v, list) and len(v) == n_items]
    if len(vecs) < 2:
        return None
    return [1 if sum(v[i] for v in vecs) * 2 > len(vecs) else 0 for i in range(n_items)]


# ── Cohort 1c ──────────────────────────────────────────────────────────────

def analyze_c1c(prefix: str, cases: dict, exclude_self: bool = False) -> dict:
    res = {"per_model": {}, "pooled": {}}
    rows = []            # one row per (model, case) with complete data
    judge_rows = []      # Fleiss rows for judge agreement
    truncation = defaultdict(lambda: [0, 0])
    finish_counts = defaultdict(Counter)
    det_rows = []
    comp_rows = []
    for mk, label in MODELS:
        recs = load_records(prefix, "c1c", mk)
        judg = load_judgments(prefix, "c1c", mk)
        comp = load_judgments(prefix, "c1ccomp", mk)
        for (cid, arm, rep), r in recs.items():
            if rep == 0 and arm != "append":
                finish_counts[f"{mk}|{arm}"][finish_label(r)] += 1
            t = truncated(r)
            if t is not None:
                truncation[(mk, arm)][0] += int(t)
                truncation[(mk, arm)][1] += 1
        for cid, c in cases.items():
            items = c["checklist"]
            high = [i for i, it in enumerate(items) if it.get("grade") == "High"]
            if not high:
                continue
            n_items = len(items)
            addr = {}
            for arm in ARMS_C1:
                key = f"{mk}|{cid}|{arm}|0"
                pj = judg.get(key, {})
                for jv in pj.values():
                    pass
                if arm in ("bare", "guard", "guard_alt", "rewrite", "append"):
                    if exclude_self:
                        pj = {j: v for j, v in pj.items() if j != SELF_JUDGE[mk]}
                    addr[arm] = item_majority(pj, n_items)
                    # judge agreement rows (item-level, three judges)
                    vecs = [v for v in pj.values() if isinstance(v, list) and len(v) == n_items]
                    if len(vecs) == 3:
                        for i in range(n_items):
                            yes = sum(v[i] for v in vecs)
                            judge_rows.append([yes, 3 - yes])
            if any(addr.get(a) is None for a in ARMS_C1) or any(
                    (cid, a, 0) not in recs for a in ARMS_C1):
                continue
            g = addr["guard"]
            fnd = addr["append"]
            delivered = [1 if (g[i] or fnd[i]) else 0 for i in range(n_items)]
            def omit(vec):
                return sum(1 for i in high if not vec[i]) / len(high)
            row = {
                "model": mk, "case_id": cid, "source": c.get("source", ""),
                "n_high": len(high),
                "omit": {"bare": omit(addr["bare"]), "guard": omit(g), "guard_alt": omit(addr["guard_alt"]),
                         "append": omit(delivered), "rewrite": omit(addr["rewrite"])},
                "any": {a: int(omit(v) > 0) for a, v in (("bare", addr["bare"]), ("guard", g),
                                                         ("guard_alt", addr["guard_alt"]),
                                                         ("append", delivered),
                                                         ("rewrite", addr["rewrite"]))},
                "guard_omitted_high": [i for i in high if not g[i]],
                "caught_high": [i for i in high if not g[i] and fnd[i]],
                "flags": findings_flag_count(recs[(cid, "append", 0)]),
                "findings_words": words((recs[(cid, "append", 0)].get("findings") or "").replace("[[RESPONSE]]", "")),
                "words": {a: words(recs[(cid, a, 0)].get("text")) for a in ("bare", "guard", "guard_alt", "rewrite")},
                "categories": {a: [(items[i].get("category"), int(bool(v[i]))) for i in high]
                               for a, v in (("bare", addr["bare"]), ("guard", g), ("guard_alt", addr["guard_alt"]),
                                            ("append", delivered), ("rewrite", addr["rewrite"]))},
            }
            rows.append(row)
            # determinism: repeats of the framework on the same response
            reps = [recs.get((cid, "append", rr)) for rr in (0, 1, 2)]
            if all(reps):
                fv = [item_majority(judg.get(f"{mk}|{cid}|append|{rr}", {}), n_items) for rr in (0, 1, 2)]
                det_rows.append({"model": mk, "case_id": cid,
                                 "lnn": [(x.get("framework_meta") or {}).get("lnn_pass_fail") for x in reps],
                                 "consensus": [bool((x.get("framework_meta") or {}).get("consensus_fired")) for x in reps],
                                 "hazards": [tuple(sorted((x.get("framework_meta") or {}).get("consensus_hazards") or [])) for x in reps],
                                 "mods": [set(str(m) for m in ((x.get("framework_meta") or {}).get("modifications_applied") or [])) for x in reps],
                                 "flags": [findings_flag_count(x) for x in reps],
                                 "text_identical": len({(x.get("findings") or "") for x in reps}) == 1,
                                 "caught": [[fv[rr][i] if fv[rr] else None for i in row["guard_omitted_high"]] for rr in range(3)]})
            # component attribution (optional single-judge pass)
            if comp:
                lnn_v = comp.get(f"{mk}|{cid}|lnn|0", {})
                ag_v = comp.get(f"{mk}|{cid}|agents|0", {})
                lv = next(iter(lnn_v.values()), None) if lnn_v else None
                av = next(iter(ag_v.values()), None) if ag_v else None
                for i in row["caught_high"]:
                    comp_rows.append({"model": mk, "lnn": (lv[i] if lv else None), "agents": (av[i] if av else None)})

    res["n_case_runs"] = len(rows)
    res["n_cases"] = len({r["case_id"] for r in rows})
    res["n_by_model"] = Counter(r["model"] for r in rows)

    def by_case(sel, fn):
        d = defaultdict(list)
        for r in sel:
            d[r["case_id"]].append(fn(r))
        return d

    contrasts = {"guard_vs_bare": ("guard", "bare"), "append_vs_guard": ("append", "guard"),
                 "rewrite_vs_guard": ("rewrite", "guard"), "rewrite_vs_append": ("rewrite", "append"),
                 "guardalt_vs_bare": ("guard_alt", "bare")}
    primary_keys = ["guard_vs_bare", "append_vs_guard", "rewrite_vs_guard", "rewrite_vs_append"]

    def summarize(sel):
        out = {"n_case_runs": len(sel), "n_cases": len({r["case_id"] for r in sel}), "arms": {}, "contrasts": {}}
        for a in ARMS_C1:
            est, lo, hi = cluster_boot_mean(by_case(sel, lambda r, a=a: r["omit"][a]))
            k = sum(r["any"][a] for r in sel)
            p, wlo, whi = wilson(k, len(sel))
            out["arms"][a] = {"omit_mean": est, "omit_ci": [lo, hi], "any_k": k, "any_n": len(sel),
                              "any_p": p, "any_ci": [wlo, whi]}
        for name, (a, b) in contrasts.items():
            d = by_case(sel, lambda r, a=a, b=b: r["omit"][a] - r["omit"][b])
            est, lo, hi = cluster_boot_mean(d)
            pval = signflip_p(d)
            bb = sum(1 for r in sel if r["any"][a] == 1 and r["any"][b] == 0)
            cc = sum(1 for r in sel if r["any"][a] == 0 and r["any"][b] == 1)
            out["contrasts"][name] = {"diff_mean": est, "diff_ci": [lo, hi], "p_signflip": pval,
                                      "any_discordant_a_only": bb, "any_discordant_b_only": cc,
                                      "any_mcnemar_p": mcnemar_exact(bb, cc)}
        adj = bh({k: out["contrasts"][k]["p_signflip"] for k in primary_keys})
        for k, v in adj.items():
            out["contrasts"][k]["p_bh"] = v
        omitted = sum(len(r["guard_omitted_high"]) for r in sel)
        caught = sum(len(r["caught_high"]) for r in sel)
        p, lo, hi = wilson(caught, omitted)
        out["catch"] = {"k": caught, "n": omitted, "p": p, "ci": [lo, hi]}
        complete = [r for r in sel if not r["guard_omitted_high"]]
        fp = sum(1 for r in complete if r["flags"] > 0)
        p, lo, hi = wilson(fp, len(complete))
        out["fpr"] = {"k": fp, "n": len(complete), "p": p, "ci": [lo, hi]}
        out["flags_per_run"] = quantiles([r["flags"] for r in sel])
        out["findings_words"] = quantiles([r["findings_words"] for r in sel])
        out["response_words"] = {a: quantiles([r["words"][a] for r in sel]) for a in ("bare", "guard", "guard_alt", "rewrite")}
        cat = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in sel:
            for a, lst in r["categories"].items():
                for cname, ok in lst:
                    cat[cname][a][0] += 1 - ok
                    cat[cname][a][1] += 1
        out["category_omission"] = {cn: {a: {"k": v[0], "n": v[1], "p": (v[0] / v[1] if v[1] else None)}
                                         for a, v in d.items()} for cn, d in cat.items()}
        return out

    res["pooled"] = summarize(rows)
    for mk, label in MODELS:
        res["per_model"][mk] = summarize([r for r in rows if r["model"] == mk])
    inscope = [r for r in rows if not any(s in r["source"].lower() for s in OUT_OF_SCOPE_SOURCES)]
    res["in_scope"] = summarize(inscope)
    res["out_of_scope_n_case_runs"] = len(rows) - len(inscope)
    res["truncation"] = {f"{mk}|{arm}": {"k": v[0], "n": v[1]} for (mk, arm), v in truncation.items()}
    res["finish_counts"] = {k: dict(v) for k, v in finish_counts.items()}
    res["judge_fleiss_kappa_items"] = fleiss_kappa(judge_rows)
    res["judge_n_item_decisions"] = len(judge_rows)

    # determinism summary
    if det_rows:
        n = len(det_rows)
        same_lnn = sum(1 for d in det_rows if len(set(d["lnn"])) == 1)
        same_cons = sum(1 for d in det_rows if len(set(d["consensus"])) == 1 and len(set(d["hazards"])) == 1)
        same_text = sum(1 for d in det_rows if d["text_identical"])
        same_anyflag = sum(1 for d in det_rows if len({f > 0 for f in d["flags"]}) == 1)
        jac = []
        for d in det_rows:
            for a, b in ((0, 1), (0, 2), (1, 2)):
                u = d["mods"][a] | d["mods"][b]
                jac.append(len(d["mods"][a] & d["mods"][b]) / len(u) if u else 1.0)
        catch_rows = []
        for d in det_rows:
            for i in range(len(d["caught"][0])):
                v = [d["caught"][rr][i] for rr in range(3)]
                if None not in v:
                    catch_rows.append([sum(v), 3 - sum(v)])
        same_catch = sum(1 for r in catch_rows if r[0] in (0, 3))
        res["determinism"] = {
            "n_case_runs": n,
            "lnn_state_identical": {"k": same_lnn, "n": n},
            "consensus_identical": {"k": same_cons, "n": n},
            "findings_text_identical": {"k": same_text, "n": n},
            "any_flag_identical": {"k": same_anyflag, "n": n},
            "lnn_findings_jaccard_mean": (sum(jac) / len(jac)) if jac else None,
            "item_catch_identical": {"k": same_catch, "n": len(catch_rows)},
            "item_catch_fleiss_kappa": fleiss_kappa(catch_rows),
            "per_model_n": Counter(d["model"] for d in det_rows),
        }
    if comp_rows:
        res["component_attribution"] = {
            "n_caught_items": len(comp_rows),
            "lnn_only": sum(1 for r in comp_rows if r["lnn"] == 1 and r["agents"] != 1),
            "agents_only": sum(1 for r in comp_rows if r["agents"] == 1 and r["lnn"] != 1),
            "both": sum(1 for r in comp_rows if r["lnn"] == 1 and r["agents"] == 1),
            "neither_section_alone": sum(1 for r in comp_rows if r["lnn"] != 1 and r["agents"] != 1),
        }
    return res


# ── Cohorts 2 and 3 (commission) ───────────────────────────────────────────

def median_score(per_judge: dict):
    s = sorted(v for v in per_judge.values() if v in (0, 1, 2))
    if len(s) < 2:
        return None
    return s[len(s) // 2]


def analyze_commission(prefix: str, cohort: str, arms: list[str], probes: dict, contrasts: dict,
                        exclude_self: bool = False) -> dict:
    rows = []
    judge_rows = []
    for mk, label in MODELS:
        recs = load_records(prefix, cohort, mk)
        judg = load_judgments(prefix, cohort, mk)
        for cid, p in probes.items():
            sc = {}
            for a in arms:
                if (cid, a, 0) not in recs:
                    break
                pj = judg.get(f"{mk}|{cid}|{a}|0", {})
                vals = [v for v in pj.values() if v in (0, 1, 2)]
                if len(vals) == 3:
                    judge_rows.append([vals.count(0), vals.count(1), vals.count(2)])
                if exclude_self:
                    pj = {j: v for j, v in pj.items() if j != SELF_JUDGE[mk]}
                sc[a] = median_score(pj)
            if len(sc) == len(arms) and all(v is not None for v in sc.values()):
                rows.append({"model": mk, "case_id": cid, "category": p["category"], "score": sc})
    out = {"n_probe_runs": len(rows), "arms": {}, "by_class": {}, "contrasts": {}, "per_model": {}}

    def comm(sel, a):
        k = sum(1 for r in sel if r["score"][a] == 0)
        p, lo, hi = wilson(k, len(sel))
        return {"k": k, "n": len(sel), "p": p, "ci": [lo, hi],
                "harm_identified_k": sum(1 for r in sel if r["score"][a] == 2)}

    def contr(sel):
        o = {}
        for name, (a, b) in contrasts.items():
            bb = sum(1 for r in sel if r["score"][a] == 0 and r["score"][b] != 0)
            cc = sum(1 for r in sel if r["score"][a] != 0 and r["score"][b] == 0)
            o[name] = {"a_only": bb, "b_only": cc, "diff": ((bb - cc) / len(sel) if sel else None),
                       "mcnemar_p": mcnemar_exact(bb, cc)}
        return o

    for a in arms:
        out["arms"][a] = comm(rows, a)
    classes = sorted({r["category"] for r in rows})
    for cl in classes:
        sel = [r for r in rows if r["category"] == cl]
        out["by_class"][cl] = {"arms": {a: comm(sel, a) for a in arms}, "contrasts": contr(sel)}
    out["contrasts"] = contr(rows)
    nonzero = [cl for cl in classes
               if any(out["by_class"][cl]["arms"][a]["k"] > 0 for a in arms)]
    out["nonzero_classes"] = nonzero
    sel = [r for r in rows if r["category"] in nonzero]
    out["contrasts_nonzero_classes"] = contr(sel)
    out["n_probe_runs_nonzero_classes"] = len(sel)
    for mk, label in MODELS:
        sm = [r for r in rows if r["model"] == mk]
        out["per_model"][mk] = {"arms": {a: comm(sm, a) for a in arms}, "contrasts": contr(sm)}
    out["judge_fleiss_kappa"] = fleiss_kappa(judge_rows)
    out["judge_n_decisions"] = len(judge_rows)
    return out


def cohort_flow() -> dict:
    """Counts at each Cohort 1c selection step, recomputed from the selection outputs."""
    def last_by_case(path):
        out = {}
        for r in load_jsonl(path):
            out[r["case_id"]] = r
        return out
    valid = last_by_case(SELECTION / "vignette_validity.jsonl")
    med = last_by_case(SELECTION / "medhelm_category.jsonl")
    decision_cats = {"supporting_diagnostic_decisions", "planning_treatments", "predicting_risks_and_outcomes"}
    valid_ids = {c for c, r in valid.items() if r.get("valid_vignette")}
    decision_ids = {c for c, r in med.items() if c in valid_ids and r.get("category") in decision_cats}
    checklists = {r["case_id"]: r.get("audited_checklist") or []
                  for r in load_jsonl(SELECTION / "case_checklists_v9b.jsonl")}
    nonempty = {c for c in decision_ids if checklists.get(c)}
    with_high = {c for c in nonempty if any(it.get("grade") == "High" for it in checklists[c])}
    c1c = {c["case_id"] for c in json.load(open(DATA / "cohort1c_cases.json"))}
    high_items = [it for c in c1c for it in checklists.get(c, []) if it.get("grade") == "High"]
    per_case = sorted(sum(1 for it in checklists.get(c, []) if it.get("grade") == "High") for c in c1c)
    cat = Counter(it.get("category") for it in high_items)
    return {
        "c1_n": len(valid),
        "c1_sources": Counter(r.get("source") for r in valid.values()),
        "c1_valid_n": len(valid_ids),
        "c1_decision_n": len(decision_ids),
        "c1_decision_categories": Counter(med[c]["category"] for c in decision_ids),
        "c1_nonempty_n": len(nonempty),
        "c1_with_high_n": len(with_high),
        "c1c_n": len(c1c),
        "c1c_equals_with_high": c1c == with_high,
        "c1c_items_total": sum(len(checklists.get(c, [])) for c in c1c),
        "c1c_high_items": len(high_items),
        "c1c_high_median": per_case[len(per_case) // 2] if len(per_case) % 2 else
                           (per_case[len(per_case) // 2 - 1] + per_case[len(per_case) // 2]) / 2,
        "c1c_high_by_category": cat,
        "c1c_sources": Counter(c.get("source") for c in json.load(open(DATA / "cohort1c_cases.json"))),
    }


def finding_level(prefix: str, cases: dict) -> dict:
    """Share of appended findings that correspond to any checklist item (single GPT-6 Astra judge)."""
    out = {"per_model": {}}
    pooled = {"n_runs": 0, "n_findings": 0, "any_item": 0, "high_item": 0}
    for mk, _ in MODELS:
        j = load_judgments(prefix, "c1cfind", mk)
        agg = {"n_runs": 0, "n_findings": 0, "any_item": 0, "high_item": 0}
        for cid, c in cases.items():
            v = j.get(f"{mk}|{cid}|findings|0", {})
            lst = next(iter(v.values()), None) if v else None
            if lst is None:
                continue
            high = {i + 1 for i, it in enumerate(c["checklist"]) if it.get("grade") == "High"}
            agg["n_runs"] += 1
            agg["n_findings"] += len(lst)
            agg["any_item"] += sum(1 for k in lst if k > 0)
            agg["high_item"] += sum(1 for k in lst if k in high)
        for k in pooled:
            pooled[k] += agg[k]
        out["per_model"][mk] = agg
    out["pooled"] = pooled
    for blk in list(out["per_model"].values()) + [pooled]:
        for name in ("any_item", "high_item"):
            pr, lo, hi = wilson(blk[name], blk["n_findings"])
            blk[name + "_share"] = {"k": blk[name], "n": blk["n_findings"], "p": pr, "ci": [lo, hi]}
    return out


def _tokens(usage) -> tuple[int, int]:
    """(input, output) tokens from OpenAI, Anthropic, or Gemini usage records; output includes reasoning."""
    if not usage:
        return 0, 0
    if "input_tokens" in usage:
        inp = (usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0) + \
              (usage.get("cache_creation_input_tokens") or 0)
        return inp, usage.get("output_tokens") or 0
    if "promptTokenCount" in usage:
        return usage.get("promptTokenCount") or 0, (usage.get("candidatesTokenCount") or 0) + \
            (usage.get("thoughtsTokenCount") or 0)
    return 0, 0


def run_summary(prefix: str) -> dict:
    """Run dates and compute proxies (TRIPOD-LLM item 12) from the subject and framework records."""
    ts, comp = [], {}
    for cohort in ("c1c", "c2", "c3"):
        for mk, _ in MODELS:
            for (cid, arm, rep), r in load_records(prefix, cohort, mk).items():
                if r.get("ts"):
                    ts.append(r["ts"])
                d = comp.setdefault(mk, {"subject_calls": 0, "input_tokens": 0, "output_tokens": 0,
                                         "framework_runs": 0, "framework_seconds": [], "framework_api_calls": []})
                if arm == "append":
                    d["framework_runs"] += 1
                    if r.get("latency_s") is not None:
                        d["framework_seconds"].append(r["latency_s"])
                    ac = (r.get("framework_meta") or {}).get("api_calls")
                    if isinstance(ac, (int, float)):
                        d["framework_api_calls"].append(ac)
                else:
                    i, o = _tokens((r.get("finish") or {}).get("usage"))
                    d["subject_calls"] += 1
                    d["input_tokens"] += i
                    d["output_tokens"] += o
    for d in comp.values():
        d["framework_seconds"] = quantiles(d["framework_seconds"])
        d["framework_api_calls"] = quantiles(d["framework_api_calls"])
    ts.sort()
    return {"run_start": ts[0][:10] if ts else None, "run_end": ts[-1][:10] if ts else None, "compute": comp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    cases = json.load(open(DATA / "cohort1c_cases_checklists.json"))
    probes2 = {p["case_id"]: p for p in json.load(open(DATA / "cohort2_probes.json"))}
    probes3 = {p["case_id"]: p for p in json.load(open(DATA / "cohort3_probes.json"))}
    out = {"prefix": args.prefix, "seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM,
           "models": dict(MODELS)}
    out["cohort_flow"] = cohort_flow()
    out["runs"] = run_summary(args.prefix)
    out["c1c"] = analyze_c1c(args.prefix, cases)
    out["c1c_findings"] = finding_level(args.prefix, cases)
    out["c2"] = analyze_commission(args.prefix, "c2", ARMS_C2, probes2, {
        "guard_vs_bare": ("guard", "bare"), "append_vs_guard": ("append", "guard"),
        "rewrite_vs_guard": ("rewrite", "guard"), "rewrite_vs_append": ("rewrite", "append")})
    out["c3"] = analyze_commission(args.prefix, "c3", ARMS_C3, probes3, {
        f"{a}_vs_full": (a, "pd_full") for a in ARMS_C3 if a != "pd_full"})
    # Share of Gemini configuration A responses for which Google Search grounding returned results
    out["gemini_grounding"] = {}
    for cohort in ("c1c", "c2"):
        g = [r for (cid, arm, rep), r in load_records(args.prefix, cohort, "gemini31").items()
             if arm == "guard" and rep == 0]
        k = sum(1 for r in g if (r.get("finish") or {}).get("grounded"))
        p, lo, hi = wilson(k, len(g))
        out["gemini_grounding"][cohort] = {"k": k, "n": len(g), "p": p, "ci": [lo, hi]}
    # Post hoc sensitivity analysis: drop the judge that shares a developer with the subject model;
    # the remaining two judges must agree (item addressed; commission score of 0).
    sens = analyze_c1c(args.prefix, cases, exclude_self=True)
    out["c1c_self_judge_excluded"] = {"pooled": sens["pooled"], "per_model": sens["per_model"],
                                      "n_case_runs": sens["n_case_runs"]}
    out["c2_self_judge_excluded"] = analyze_commission(args.prefix, "c2", ARMS_C2, probes2, {
        "append_vs_guard": ("append", "guard"), "rewrite_vs_guard": ("rewrite", "guard"),
        "rewrite_vs_append": ("rewrite", "append")}, exclude_self=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print(f"wrote {args.out}")
    c = out["c1c"]
    print(f"C1c case-runs={c['n_case_runs']} cases={c['n_cases']} kappa={c['judge_fleiss_kappa_items']}")
    for a in ARMS_C1:
        v = c["pooled"]["arms"].get(a)
        if v and v["omit_mean"] is not None:
            print(f"  {a:10s} omit={100*v['omit_mean']:.1f}% any={v['any_k']}/{v['any_n']}")


if __name__ == "__main__":
    main()
