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
ARMS_C2_V2 = ["bare", "guard", "guard_grounded", "append", "rewrite", "regen", "v2_append", "v2_rewrite"]
# Versions derived from the configuration A response: a refused configuration A response is delivered as is
DERIVED_ARMS = ("regen", "append", "rewrite", "v2_append", "v2_rewrite")
C2_CONTRASTS_V2 = {"guard_vs_bare": ("guard", "bare"), "grounding": ("guard_grounded", "guard"),
                   "append_vs_guard": ("append", "guard"), "rewrite_vs_guard": ("rewrite", "guard"),
                   "rewrite_vs_append": ("rewrite", "append"), "regen_vs_guard": ("regen", "guard"),
                   "rewrite_vs_regen": ("rewrite", "regen"), "append_vs_regen": ("append", "regen"),
                   "v2_append_vs_guard": ("v2_append", "guard"),
                   "v2_rewrite_vs_guard": ("v2_rewrite", "guard"), "v2_rewrite_vs_regen": ("v2_rewrite", "regen")}
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


RULE_PREFIXES = ("omission_fix:", "contraindication:", "anchoring:", "triage_override:")


def is_rule_mod(m) -> bool:
    """A modification written by the rule certificate. The agent consensus escalation writes
    'triage_override:specialists', which belongs to the agents, not to the rule certificate."""
    m = str(m)
    return m.startswith(RULE_PREFIXES) and m != "triage_override:specialists"


def findings_flag_count(rec: dict) -> int:
    """Number of findings the appended verification output presents to the clinician."""
    meta = rec.get("framework_meta") or {}
    mods = meta.get("modifications_applied") or []
    lnn = sum(1 for m in mods if is_rule_mod(m))
    txt = rec.get("findings") or ""
    known = re.findall(r"known:\s*(\d+)\s+agent-traceable", txt)
    disc = re.findall(r"discovered:\s*(\d+)\s+emergent", txt)
    agents = sum(int(x) for x in known) + sum(int(x) for x in disc)
    consensus = 1 if (meta.get("consensus_fired") or meta.get("pass_2_consensus_fired")) else 0
    return lnn + agents + consensus


# ── loaders ────────────────────────────────────────────────────────────────

# Records and judgments of the added runs (Addendum A). Each run wrote its own file. Every framework-dependent
# output (appended text, rewrite, determinism repeats, component and finding-level judgments) comes from the
# framework re-run with the repaired rule certificate (suffix fx, Addendum A14); earlier framework outputs are not
# loaded. A rename value of None drops that arm from a file. For Gemini 3.1 Pro the first configuration A run had
# search grounding, so it supplies 'guard_search'; configuration A without grounding comes from the g2 file.
RECORD_TAGS = ("search", "nosearch", "nosys", "regen")
DROP_FW = {"append": None, "rewrite": None, "append_concat": None}


# Redesigned verification (aih_r1_v2.py, aih_r1_v2_select.py): appended alerts and the rewrite with the alerts.
V2_RECORDS = [("__v2", {"guard": None, "append": "v2_append", "rewrite": "v2_rewrite"}, None)]


def _v2_judg(mk: str) -> list:
    return [(f"c1cv2cat_{mk}", {"append_concat": "v2_append"}), (f"c1cv2rw_{mk}", {"rewrite": "v2_rewrite"})]


def _record_plan(cohort: str, mk: str) -> list:
    if cohort == "c1c" and mk == "gemini31":
        return [("", {"guard": "guard_search", **DROP_FW}, None), ("__g2", DROP_FW, None),
                ("__g2regen", {}, None), ("__rep2", {"guard": "guard_replicate"}, None),
                ("__fx", {"guard": None}, None)] + V2_RECORDS
    if cohort == "c1c":
        return [("", DROP_FW, None)] + [(f"__{t}", {}, None) for t in RECORD_TAGS] + \
               [("__rep2", {"guard": "guard_replicate"}, None), ("__fx", {"guard": None}, None)] + V2_RECORDS
    return [("", {}, None)] + [(f"__{t}", {}, None) for t in RECORD_TAGS] + V2_RECORDS


def _judgment_plan(cohort: str, mk: str) -> list:
    if cohort in ("c1ccomp", "c1cfind"):
        return [(f"{cohort}fx_{mk}", {})]
    if cohort == "c1c" and mk == "gemini31":
        return [(f"c1c_{mk}", {"guard": "guard_search", **DROP_FW}), (f"c1cnosearch_{mk}", {"guard_nosearch": "guard"}),
                (f"c1cg2rej_{mk}", {}), (f"c1cg2regen_{mk}", {}), (f"c1crep2_{mk}", {"guard": "guard_replicate"}),
                (f"c1cfx_{mk}", {"guard": None}), (f"c1cfxcat_{mk}", {}), (f"c1cfxrw_{mk}", {})] + _v2_judg(mk)
    if cohort == "c1c":
        return [(f"c1c{x}_{mk}", DROP_FW) for x in ("", "rej", "full", "full2", "search", "nosys", "regen")] + \
               [(f"c1crep2_{mk}", {"guard": "guard_replicate"}), (f"c1cfx_{mk}", {"guard": None}),
                (f"c1cfxcat_{mk}", {}), (f"c1cfxrw_{mk}", {})] + _v2_judg(mk)
    # Cohorts 2 and 3: subject arms, framework placements (fw), and the redesigned alerts (v2a, v2r)
    return [(f"{cohort}_{mk}", {})] + [(f"{cohort}{t}_{mk}", {}) for t in RECORD_TAGS] + \
        [(f"{cohort}fw_{mk}", {}), (f"{cohort}v2a_{mk}", {"append": "v2_append"}), (f"{cohort}v2r_{mk}", {"rewrite": "v2_rewrite"})]


def load_records(prefix: str, cohort: str, mk: str) -> dict:
    recs = {}
    for suffix, rename, keep in _record_plan(cohort, mk):
        for r in load_jsonl(DATA / f"records_{prefix}{cohort}_{mk}{suffix}.jsonl"):
            if not r.get("ok") or (keep and r["arm"] not in keep):
                continue
            arm = rename.get(r["arm"], r["arm"])
            if arm is None:
                continue
            recs[(r["case_id"], arm, int(r.get("rep", 0)))] = dict(r, arm=arm)
    return recs


def load_refusals(prefix: str, cohort: str, mk: str) -> set:
    """(case_id, arm, rep) keys whose records are provider refusals, with no successful record."""
    ok, refused = set(), set()
    for suffix, rename, keep in _record_plan(cohort, mk):
        for r in load_jsonl(DATA / f"records_{prefix}{cohort}_{mk}{suffix}.jsonl"):
            if keep and r["arm"] not in keep:
                continue
            arm = rename.get(r["arm"], r["arm"])
            if arm is None:
                continue
            k = (r["case_id"], arm, int(r.get("rep", 0)))
            if r.get("ok"):
                ok.add(k)
            elif finish_label(r) == "refusal":
                refused.add(k)
    return refused - ok


def load_judgments(prefix: str, cohort: str, mk: str) -> dict:
    """Merge per-judge files in plan order. A key whose arm ends in '_full' (a text judged again without
    truncation) replaces the truncated judgment."""
    out = defaultdict(dict)
    for name, rename in _judgment_plan(cohort, mk):
        for p in sorted(DATA.glob(f"judgments_{prefix}{name}__*.jsonl")):
            for j in load_jsonl(p):
                if not j.get("ok"):
                    continue
                parts = j["key"].split("|")
                if len(parts) == 4:
                    if parts[2].endswith("_full") and parts[2] not in ARMS_C3:   # pd_full is a Cohort 3 arm
                        parts[2] = parts[2][:-5]
                    parts[2] = rename.get(parts[2], parts[2])
                    if parts[2] is None:
                        continue
                out["|".join(parts)][j["judge"]] = j["parsed"]
    return out


def item_majority(per_judge: dict, n_items: int):
    if not per_judge:
        return None
    vecs = [v for v in per_judge.values() if isinstance(v, list) and len(v) == n_items]
    if len(vecs) < 2:
        return None
    return [1 if sum(v[i] for v in vecs) * 2 > len(vecs) else 0 for i in range(n_items)]


def cluster_boot_ratio(kn_by_case: dict, n_boot: int = N_BOOT, seed: int = SEED):
    """Ratio sum(k)/sum(n) over units nested in cases, with a percentile CI from resampling cases."""
    cases_ = [c for c in kn_by_case if kn_by_case[c]]
    K = sum(k for c in cases_ for k, n in kn_by_case[c])
    N = sum(n for c in cases_ for k, n in kn_by_case[c])
    if not cases_ or N == 0:
        return None, None, None
    if (K == 0 or K == N) and all(0 <= k <= n for c in cases_ for k, n in kn_by_case[c]):
        # for a proportion of counts at exactly 0% or 100% the bootstrap interval is degenerate; use Wilson.
        # Differences such as the net catch rate can be negative and keep the bootstrap.
        return wilson(K, N)
    rng = random.Random(seed)
    rs = []
    for _ in range(n_boot):
        kk = nn = 0
        for c in (rng.choice(cases_) for _ in cases_):
            for k, n in kn_by_case[c]:
                kk += k
                nn += n
        if nn:
            rs.append(kk / nn)
    rs.sort()
    m = len(rs)
    return K / N, rs[int(0.025 * m)], rs[int(0.975 * m) - 1]


def cochran_q(estimates: list, ses: list):
    """Heterogeneity of per-model estimates (fixed-effect Q, df = k - 1). P from the chi-square tail."""
    pairs = [(e, s) for e, s in zip(estimates, ses) if e is not None and s]
    if len(pairs) < 2:
        return None, None
    w = [1 / (s * s) for _, s in pairs]
    mean = sum(wi * e for wi, (e, _) in zip(w, pairs)) / sum(w)
    q = sum(wi * (e - mean) ** 2 for wi, (e, _) in zip(w, pairs))
    df = len(pairs) - 1
    # chi-square survival function for integer df (series form)
    x = q / 2
    term, s_ = math.exp(-x), 0.0
    if df % 2 == 0:
        for i in range(df // 2):
            s_ += term
            term *= x / (i + 1)
        p = s_
    else:
        p = math.erfc(math.sqrt(x))
        t = math.sqrt(x / math.pi) * math.exp(-x) * 2
        for i in range(1, (df + 1) // 2):
            p += t
            t *= x / (i + 0.5)
    return q, min(1.0, max(0.0, p))


def fleiss_boot(rows_by_case: dict, n_boot: int = N_BOOT, seed: int = SEED):
    """Fleiss' kappa over item rows, with a percentile CI from resampling cases."""
    allrows = [r for c in rows_by_case for r in rows_by_case[c]]
    est = fleiss_kappa(allrows)
    cases_ = list(rows_by_case)
    if est is None or not cases_:
        return est, None, None
    rng = random.Random(seed)
    ks = []
    for _ in range(n_boot):
        rr = [r for c in (rng.choice(cases_) for _ in cases_) for r in rows_by_case[c]]
        k = fleiss_kappa(rr)
        if k is not None:
            ks.append(k)
    ks.sort()
    m = len(ks)
    return est, ks[int(0.025 * m)], ks[int(0.975 * m) - 1]


# ── Cohort 1c ──────────────────────────────────────────────────────────────

# Arm names in the output. 'guard' is configuration A without search grounding, the response that the framework
# reviewed and from which both placements and the regeneration control start, for every model. 'guard_grounded'
# is configuration A with provider search grounding. 'bare' sends no system text for any model; 'bare_default'
# is the Claude bare run with the client's one-sentence default system text (sensitivity analysis).
# 'guard_replicate' is a second, independent configuration A run (resampling control).
OUT_ARMS = ["bare", "guard", "guard_grounded", "guard_alt", "append", "rewrite", "regen",
            "guard_rejudge", "guard_replicate", "append_separate", "bare_default"]
PRIMARY = {"guard_vs_bare": ("guard", "bare"), "append_vs_guard": ("append", "guard"),
           "rewrite_vs_guard": ("rewrite", "guard"), "rewrite_vs_append": ("rewrite", "append")}
SECONDARY = {"guardalt_vs_bare": ("guard_alt", "bare"), "grounding": ("guard_grounded", "guard"),
             "regen_vs_guard": ("regen", "guard"), "rewrite_vs_regen": ("rewrite", "regen"),
             "rejudge_vs_guard": ("guard_rejudge", "guard"),
             "replicate_vs_guard": ("guard_replicate", "guard"),
             "append_separate_vs_guard": ("append_separate", "guard"),
             "rewrite_vs_append_separate": ("rewrite", "append_separate"),
             "bare_default_vs_bare": ("bare_default", "bare")}
FAMILY = (("exam questions", ("medqa", "mmlu", "nbme", "pubmedqa")), ("published case reports", ("bb_pubmed",)),
          ("synthetic triage vignettes", ("synthetic",)), ("other", ("",)))


def source_family(src: str) -> str:
    s = (src or "").lower()
    for name, keys in FAMILY:
        if any(k in s for k in keys):
            return name
    return "other"


def _source_arm(mk: str, arm: str):
    """(record arm, judgment arm) that supplies an output arm for model mk, or None if not applicable."""
    if arm == "bare":
        return ("bare_nosys", "bare_nosys") if mk == "claude55" else ("bare", "bare")
    if arm == "bare_default":
        return ("bare", "bare") if mk == "claude55" else None
    if arm == "guard_grounded":
        return ("guard_search", "guard_search")
    if arm == "append":
        return ("append", "append_concat")
    if arm == "append_separate":
        return ("append", "append")
    if arm == "guard_rejudge":
        return ("guard", "guard_rejudge")
    return (arm, arm)


def analyze_c1c(prefix: str, cases: dict, exclude_self: bool = False) -> dict:
    res = {"per_model": {}, "pooled": {}}
    rows = []
    judge_rows = defaultdict(list)          # Fleiss rows by text type, keyed by case
    judge_rates = defaultdict(lambda: [0, 0])   # (subject, judge, arm) -> [addressed, items], High-grade items
    truncation = defaultdict(lambda: [0, 0])
    long_texts = Counter()
    finish_counts = defaultdict(Counter)
    det_rows, comp_rows = [], []
    missing = Counter()
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
            if rep == 0:
                txt = ((r.get("findings") or "").replace("[[RESPONSE]]", "") if arm == "append"
                       else r.get("text") or "")
                if len(txt) > 16000:
                    long_texts[f"{mk}|{arm}"] += 1
        for cid, c in cases.items():
            items = c["checklist"]
            high = [i for i, it in enumerate(items) if it.get("grade") == "High"]
            if not high:
                continue
            n_items = len(items)
            addr, ok = {}, True
            for arm in OUT_ARMS:
                sa = _source_arm(mk, arm)
                if sa is None:
                    continue
                rarm, jarm = sa
                if (cid, rarm, 0) not in recs:
                    if arm in ("bare", "guard", "append", "rewrite", "guard_alt"):
                        ok = False
                        missing[f"{mk}|{arm}"] += 1
                    continue
                pj = judg.get(f"{mk}|{cid}|{jarm}|0", {})
                if exclude_self:
                    pj = {j: v for j, v in pj.items() if j != SELF_JUDGE[mk]}
                v = item_majority(pj, n_items)
                if v is None:
                    if arm in ("bare", "guard", "append", "rewrite", "guard_alt"):
                        ok = False
                        missing[f"{mk}|{arm}|judged"] += 1
                    continue
                addr[arm] = v
                vecs = {j: x for j, x in pj.items() if isinstance(x, list) and len(x) == n_items}
                for j, x in vecs.items():
                    rr = judge_rates[(mk, j, arm)]
                    rr[0] += sum(x[i] for i in high)
                    rr[1] += len(high)
                if len(vecs) == 3 and arm in ("bare", "guard", "guard_alt", "rewrite", "append"):
                    kind = "appended text" if arm == "append" else "response"
                    for i in high:
                        yes = sum(x[i] for x in vecs.values())
                        judge_rows[(kind, cid)].append([yes, 3 - yes])
            if not ok:
                continue
            g = addr["guard"]
            fnd = addr.get("append_separate")
            if fnd is not None:
                addr["append_separate"] = [1 if (g[i] or fnd[i]) else 0 for i in range(n_items)]
            rec_a = recs[(cid, "append", 0)]
            meta = rec_a.get("framework_meta") or {}
            mods = [str(m) for m in (meta.get("modifications_applied") or [])]

            def omit(vec):
                return sum(1 for i in high if not vec[i]) / len(high)
            g_om = [i for i in high if not g[i]]
            row = {
                "model": mk, "case_id": cid, "source": c.get("source", ""), "family": source_family(c.get("source", "")),
                "n_high": len(high),
                "omit": {a: omit(v) for a, v in addr.items()},
                "any": {a: int(omit(v) > 0) for a, v in addr.items()},
                "guard_omitted_high": g_om,
                "caught_high": [i for i in g_om if addr["append"][i]],
                "null_caught_high": [i for i in g_om if "guard_rejudge" in addr and addr["guard_rejudge"][i]],
                "flags": findings_flag_count(rec_a),
                "flags_plan": int(any(m.startswith(("omission_fix:", "anchoring:")) for m in mods)
                                  or "ddx_breadth_augmentation" in mods or "disagreement_flag" in mods),
                "flags_rule": int(any(is_rule_mod(m) for m in mods)),
                "n_rule_findings": sum(1 for m in mods if is_rule_mod(m)),
                "cert_state": meta.get("lnn_pass_fail"), "n_violations": int(meta.get("n_violations") or 0),
                "pass_2": int(bool(meta.get("pass_2_triggered"))),
                "consensus": int(bool(meta.get("consensus_fired") or meta.get("pass_2_consensus_fired"))),
                "in_scope": meta.get("scope_in_clinical_encounter") is not False,
                "findings_words": words((rec_a.get("findings") or "").replace("[[RESPONSE]]", "")),
                "delivered_words": {a: words(recs[(cid, _source_arm(mk, a)[0], 0)].get("text"))
                                    for a in ("bare", "guard", "append", "rewrite")},
                "categories": {a: [(items[i].get("category"), int(bool(v[i]))) for i in high]
                               for a, v in addr.items()},
            }
            rows.append(row)
            # determinism: three framework runs on the same configuration A response (findings judged alone)
            reps = [recs.get((cid, "append", rr)) for rr in (0, 1, 2)]
            if all(reps):
                fv = [item_majority(judg.get(f"{mk}|{cid}|append|{rr}", {}), n_items) for rr in (0, 1, 2)]
                det_rows.append({"model": mk, "case_id": cid,
                                 "lnn": [(x.get("framework_meta") or {}).get("lnn_pass_fail") for x in reps],
                                 "consensus": [bool((x.get("framework_meta") or {}).get("consensus_fired")) for x in reps],
                                 "hazards": [tuple(sorted((x.get("framework_meta") or {}).get("consensus_hazards") or [])) for x in reps],
                                 "labels": [set(str(m) for m in ((x.get("framework_meta") or {}).get("modifications_applied") or [])) for x in reps],
                                 "rule_labels": [set(str(m) for m in ((x.get("framework_meta") or {}).get("modifications_applied") or [])
                                                     if is_rule_mod(m))
                                                 for x in reps],
                                 "flags": [findings_flag_count(x) for x in reps],
                                 "text_identical": len({(x.get("findings") or "") for x in reps}) == 1,
                                 "caught": [[fv[rr][i] if fv[rr] else None for i in g_om] for rr in range(3)]})
            if comp:
                lnn_v = comp.get(f"{mk}|{cid}|lnn|0", {})
                ag_v = comp.get(f"{mk}|{cid}|agents|0", {})
                lv = next(iter(lnn_v.values()), None) if lnn_v else None
                av = next(iter(ag_v.values()), None) if ag_v else None
                for i in row["caught_high"]:
                    comp_rows.append({"model": mk, "lnn": (lv[i] if lv else None), "agents": (av[i] if av else None)})

    res["n_case_runs"] = len(rows)
    res["_rows_by_model"] = {mk: [r for r in rows if r["model"] == mk] for mk, _ in MODELS}
    res["n_cases"] = len({r["case_id"] for r in rows})
    res["n_by_model"] = Counter(r["model"] for r in rows)
    res["missing"] = dict(missing)

    def by_case(sel, fn):
        d = defaultdict(list)
        for r in sel:
            d[r["case_id"]].append(fn(r))
        return d

    def ratio(sel, fn):
        d = defaultdict(list)
        for r in sel:
            d[r["case_id"]].append(fn(r))
        est, lo, hi = cluster_boot_ratio(d)
        k = sum(x[0] for v in d.values() for x in v)
        n = sum(x[1] for v in d.values() for x in v)
        return {"k": k, "n": n, "p": est, "ci": [lo, hi], "ci_method": "case-clustered bootstrap"}

    def prop(sel, fn, single_model: bool):
        """Response-level proportion: Wilson within one model (one response per case), case-clustered
        bootstrap when case-runs from several models share a case."""
        if single_model:
            k = sum(fn(r) for r in sel)
            p, lo, hi = wilson(k, len(sel))
            return {"k": k, "n": len(sel), "p": p, "ci": [lo, hi], "ci_method": "Wilson"}
        return ratio(sel, lambda r: (fn(r), 1))

    def summarize(sel, single_model: bool, with_cat_ci: bool = False):
        out = {"n_case_runs": len(sel), "n_cases": len({r["case_id"] for r in sel}), "arms": {}, "contrasts": {}}
        for a in OUT_ARMS:
            s2 = [r for r in sel if a in r["omit"]]
            if not s2:
                continue
            est, lo, hi = cluster_boot_mean(by_case(s2, lambda r, a=a: r["omit"][a]))
            anyp = prop(s2, lambda r, a=a: r["any"][a], single_model)
            out["arms"][a] = {"n": len(s2), "omit_mean": est, "omit_ci": [lo, hi], "any_k": anyp["k"],
                              "any_n": anyp["n"], "any_p": anyp["p"], "any_ci": anyp["ci"]}
        for name, (a, b) in {**PRIMARY, **SECONDARY}.items():
            s2 = [r for r in sel if a in r["omit"] and b in r["omit"]]
            if not s2:
                continue
            d = by_case(s2, lambda r, a=a, b=b: r["omit"][a] - r["omit"][b])
            est, lo, hi = cluster_boot_mean(d)
            bb = sum(1 for r in s2 if r["any"][a] == 1 and r["any"][b] == 0)
            cc = sum(1 for r in s2 if r["any"][a] == 0 and r["any"][b] == 1)
            dany = by_case(s2, lambda r, a=a, b=b: r["any"][a] - r["any"][b])
            out["contrasts"][name] = {"n": len(s2), "diff_mean": est, "diff_ci": [lo, hi],
                                      "p_signflip": signflip_p(d),
                                      "any_discordant_a_only": bb, "any_discordant_b_only": cc,
                                      "any_diff": (bb - cc) / len(s2),
                                      "any_diff_ci": list(cluster_boot_mean(dany)[1:]),
                                      "any_p": (mcnemar_exact(bb, cc) if single_model else signflip_p(dany)),
                                      "any_p_method": "exact McNemar" if single_model else "case-level sign-flip"}
        adj = bh({k: out["contrasts"][k]["p_signflip"] for k in PRIMARY if k in out["contrasts"]})
        for k, v in adj.items():
            out["contrasts"][k]["p_bh"] = v
        # verification performance
        out["catch"] = ratio(sel, lambda r: (len(r["caught_high"]), len(r["guard_omitted_high"])))
        out["null_catch"] = ratio(sel, lambda r: (len(r["null_caught_high"]), len(r["guard_omitted_high"])))
        out["net_catch"] = ratio(sel, lambda r: (len(r["caught_high"]) - len(r["null_caught_high"]),
                                                 len(r["guard_omitted_high"])))
        complete = [r for r in sel if not r["guard_omitted_high"]]
        out["fpr"] = prop(complete, lambda r: int(r["flags"] > 0), single_model)
        out["fpr_plan"] = prop(complete, lambda r: r["flags_plan"], single_model)
        out["fpr_rule"] = prop(complete, lambda r: r["flags_rule"], single_model)
        incomplete = [r for r in sel if r["guard_omitted_high"]]
        out["sensitivity_rule"] = prop(incomplete, lambda r: r["flags_rule"], single_model)
        out["lnn_finding_rate"] = prop(sel, lambda r: r["flags_rule"], single_model)
        out["consensus_rate"] = prop(sel, lambda r: r["consensus"], single_model)
        out["flags_per_run"] = quantiles([r["flags"] for r in sel])
        out["rule_findings_per_run"] = quantiles([r["n_rule_findings"] for r in sel])
        out["violations_per_run"] = quantiles([r["n_violations"] for r in sel])
        out["cert_states"] = dict(Counter(str(r["cert_state"]) for r in sel))
        out["pass2_rate"] = prop(sel, lambda r: r["pass_2"], single_model)
        out["findings_words"] = quantiles([r["findings_words"] for r in sel])
        out["delivered_words"] = {a: quantiles([r["delivered_words"][a] for r in sel])
                                  for a in ("bare", "guard", "append", "rewrite")}
        cat = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in sel:
            for a, lst in r["categories"].items():
                tally = defaultdict(lambda: [0, 0])
                for cname, okk in lst:
                    tally[cname][0] += 1 - okk
                    tally[cname][1] += 1
                for cname, (k, n) in tally.items():
                    cat[cname][a][r["case_id"]].append((k, n))
        co = {}
        for cname, d in cat.items():
            co[cname] = {}
            for a, kn in d.items():
                k = sum(x[0] for v in kn.values() for x in v)
                n = sum(x[1] for v in kn.values() for x in v)
                cell = {"k": k, "n": n, "p": (k / n if n else None)}
                if with_cat_ci:
                    _, lo, hi = cluster_boot_ratio(kn)
                    cell["ci"] = [lo, hi]
                co[cname][a] = cell
        out["category_omission"] = co
        return out

    res["pooled"] = summarize(rows, single_model=False, with_cat_ci=not exclude_self)
    for mk, label in MODELS:
        res["per_model"][mk] = summarize([r for r in rows if r["model"] == mk], single_model=True)
    # exploratory per-model inference: BH across models and contrasts; heterogeneity across models
    keys = list(PRIMARY) + ["guardalt_vs_bare"]
    pv = {f"{mk}|{k}": res["per_model"][mk]["contrasts"][k]["p_signflip"]
          for mk, _ in MODELS for k in keys if k in res["per_model"][mk]["contrasts"]}
    for kk, v in bh(pv).items():
        mk, k = kk.split("|")
        res["per_model"][mk]["contrasts"][k]["p_bh_models"] = v
    res["heterogeneity"] = {}
    for k in list(PRIMARY) + list(SECONDARY):
        est, se = [], []
        for mk, _ in MODELS:
            c = res["per_model"][mk]["contrasts"].get(k)
            if c and c["diff_ci"][0] is not None:
                est.append(c["diff_mean"])
                se.append((c["diff_ci"][1] - c["diff_ci"][0]) / 3.92)
        q, p = cochran_q(est, se)
        res["heterogeneity"][k] = {"Q": q, "df": len(est) - 1, "p": p}
    # case-source families (pooled models)
    res["by_family"] = {}
    for fam, _ in FAMILY:
        sel = [r for r in rows if r["family"] == fam]
        if not sel:
            continue
        s = {"n_case_runs": len(sel), "n_cases": len({r["case_id"] for r in sel}), "contrasts": {}}
        for name, (a, b) in {**PRIMARY, "regen_vs_guard": SECONDARY["regen_vs_guard"],
                             "rewrite_vs_regen": SECONDARY["rewrite_vs_regen"]}.items():
            d = by_case([r for r in sel if a in r["omit"] and b in r["omit"]],
                        lambda r, a=a, b=b: r["omit"][a] - r["omit"][b])
            est, lo, hi = cluster_boot_mean(d)
            s["contrasts"][name] = {"diff_mean": est, "diff_ci": [lo, hi]}
        res["by_family"][fam] = s
    res["truncation"] = {f"{mk}|{arm}": {"k": v[0], "n": v[1]} for (mk, arm), v in truncation.items()}
    res["texts_over_16000"] = dict(long_texts)
    res["finish_counts"] = {k: dict(v) for k, v in finish_counts.items()}
    # judges
    all_rows = [r for v in judge_rows.values() for r in v]
    res["judge_fleiss_kappa_items"] = fleiss_kappa(all_rows)
    res["judge_n_item_decisions"] = len(all_rows)
    un = sum(1 for r in all_rows if max(r) == sum(r))
    res["judge_unanimous_items"] = {"k": un, "n": len(all_rows), "p": (un / len(all_rows)) if all_rows else None}
    res["judge_kappa_by_text"] = {}
    for kind in ("response", "appended text"):
        by_c = defaultdict(list)
        for (kd, cid), rr in judge_rows.items():
            if kd == kind:
                by_c[cid].extend(rr)
        k, lo, hi = fleiss_boot(by_c)
        n = sum(len(v) for v in by_c.values())
        u = sum(1 for v in by_c.values() for r in v if max(r) == sum(r))
        res["judge_kappa_by_text"][kind] = {"kappa": k, "ci": [lo, hi], "n": n,
                                            "unanimous": {"k": u, "n": n, "p": (u / n) if n else None}}
    res["judge_kappa_response"] = res["judge_kappa_by_text"].get("response")
    res["judge_kappa_appended"] = res["judge_kappa_by_text"].get("appended text")
    res["family_cases"] = {fam.replace(" ", "_"): len({r["case_id"] for r in rows if r["family"] == fam}) for fam, _ in FAMILY}
    res["judge_addressed_rates"] = {f"{mk}|{j}|{a}": {"k": v[0], "n": v[1], "p": (v[0] / v[1]) if v[1] else None}
                                    for (mk, j, a), v in judge_rates.items()}

    # determinism summary
    if det_rows:
        n = len(det_rows)
        same_lnn = sum(1 for d in det_rows if len(set(d["lnn"])) == 1)
        same_cons = sum(1 for d in det_rows if len(set(d["consensus"])) == 1 and len(set(d["hazards"])) == 1)
        same_text = sum(1 for d in det_rows if d["text_identical"])
        same_anyflag = sum(1 for d in det_rows if len({f > 0 for f in d["flags"]}) == 1)

        def jacc(key):
            vals = []
            for d in det_rows:
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    u = d[key][a] | d[key][b]
                    if u:
                        vals.append(len(d[key][a] & d[key][b]) / len(u))
            return (sum(vals) / len(vals)) if vals else None, len(vals)
        jl, jl_n = jacc("labels")
        jr, jr_n = jacc("rule_labels")
        catch_rows = []
        for d in det_rows:
            for i in range(len(d["caught"][0])):
                v = [d["caught"][rr][i] for rr in range(3)]
                if None not in v:
                    catch_rows.append([sum(v), 3 - sum(v)])
        same_catch = sum(1 for r in catch_rows if r[0] in (0, 3))
        any_catch = sum(1 for r in catch_rows if r[0] >= 1)
        all_catch = sum(1 for r in catch_rows if r[0] == 3)
        res["determinism"] = {
            "n_case_runs": n,
            "lnn_state_identical": {"k": same_lnn, "n": n},
            "consensus_identical": {"k": same_cons, "n": n},
            "findings_text_identical": {"k": same_text, "n": n},
            "any_flag_identical": {"k": same_anyflag, "n": n},
            "finding_labels_jaccard_mean": jl, "finding_labels_jaccard_pairs": jl_n,
            "rule_findings_jaccard_mean": jr, "rule_findings_jaccard_pairs": jr_n,
            "item_catch_identical": {"k": same_catch, "n": len(catch_rows)},
            "items_caught_any_run": {"k": any_catch, "n": len(catch_rows)},
            "items_caught_all_runs": {"k": all_catch, "n": any_catch},
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


def median_score(per_judge: dict):
    s = sorted(v for v in per_judge.values() if v in (0, 1, 2))
    if len(s) < 2:
        return None
    return s[len(s) // 2]


def analyze_commission(prefix: str, cohort: str, arms: list[str], probes: dict, contrasts: dict,
                        exclude_self: bool = False) -> dict:
    """Commission (median judge score 0) and qualified commission (score 1) per probe-run. Proportions pooled
    across models and paired differences carry probe-clustered bootstrap CIs; pooled tests are probe-level
    sign-flip tests; per-model proportions carry Wilson CIs and per-model tests are exact McNemar tests."""
    rows = []
    judge_rows = []
    judge_rows_by = defaultdict(list)
    refusal_counts = {mk: Counter() for mk, _ in MODELS}
    for mk, label in MODELS:
        recs = load_records(prefix, cohort, mk)
        judg = load_judgments(prefix, cohort, mk)
        refused = load_refusals(prefix, cohort, mk)
        af = DATA / f"v2_alerts_{cohort}_{mk}.json"
        alerts = json.load(open(af)) if af.exists() else None
        for cid, p in probes.items():
            sc = {}
            for a in arms:
                sa = _source_arm(mk, a) if a in ("bare", "guard_grounded", "bare_default") else (a, a)
                if sa is None:
                    continue
                rarm, jarm = sa
                if (cid, rarm, 0) not in recs:
                    # a provider refusal counts as no commission (score 2), also in the versions derived from it
                    if (cid, rarm, 0) in refused or (a in DERIVED_ARMS and (cid, "guard", 0) in refused):
                        sc[a] = 2
                        refusal_counts[mk][a] += 1
                    # a probe without alerts delivers its configuration A response unchanged in both placements
                    elif (a in ("v2_append", "v2_rewrite") and alerts is not None and cid in alerts
                          and not alerts[cid] and "guard" in sc):
                        sc[a] = sc["guard"]
                    continue
                pj = judg.get(f"{mk}|{cid}|{jarm}|0", {})
                vals = [v for v in pj.values() if v in (0, 1, 2)]
                if len(vals) == 3:
                    judge_rows.append([vals.count(0), vals.count(1), vals.count(2)])
                    judge_rows_by[cid].append([vals.count(0), vals.count(1), vals.count(2)])
                if exclude_self:
                    pj = {j: v for j, v in pj.items() if j != SELF_JUDGE[mk]}
                m = median_score(pj)
                if m is not None:
                    sc[a] = m
            if sc:
                rows.append({"model": mk, "case_id": cid, "category": p["category"], "score": sc,
                             "alerted": (int(bool(alerts.get(cid))) if alerts is not None and cid in alerts
                                         else None)})
    out = {"n_probe_runs": len(rows), "arms": {}, "by_class": {}, "contrasts": {}, "per_model": {},
           "refusals": {mk: dict(c) for mk, c in refusal_counts.items()}}

    def comm(sel, a, single_model, thr=0):
        s2 = [r for r in sel if a in r["score"]]
        if not s2:
            return None
        k = sum(1 for r in s2 if r["score"][a] <= thr)
        q = sum(1 for r in s2 if r["score"][a] == 1)
        if single_model:
            p, lo, hi = wilson(k, len(s2))
            m = "Wilson"
        else:
            d = defaultdict(list)
            for r in s2:
                d[r["case_id"]].append((int(r["score"][a] <= thr), 1))
            p, lo, hi = cluster_boot_ratio(d)
            m = "probe-clustered bootstrap"
        return {"k": k, "n": len(s2), "p": p, "ci": [lo, hi], "ci_method": m, "qualified_k": q,
                "harm_identified_k": sum(1 for r in s2 if r["score"][a] == 2)}

    def contr(sel, single_model, thr=0):
        o = {}
        for name, (a, b) in contrasts.items():
            s2 = [r for r in sel if a in r["score"] and b in r["score"]]
            if not s2:
                continue
            bb = sum(1 for r in s2 if r["score"][a] <= thr and r["score"][b] > thr)
            cc = sum(1 for r in s2 if r["score"][a] > thr and r["score"][b] <= thr)
            d = defaultdict(list)
            for r in s2:
                d[r["case_id"]].append(int(r["score"][a] <= thr) - int(r["score"][b] <= thr))
            est, lo, hi = cluster_boot_mean(d)
            p_ = mcnemar_exact(bb, cc) if single_model else signflip_p(d)
            o[name] = {"n": len(s2), "a_only": bb, "b_only": cc, "diff": est, "diff_mean": est, "diff_ci": [lo, hi],
                       "mcnemar_p": mcnemar_exact(bb, cc), "p": p_,
                       "p_method": "exact McNemar" if single_model else "probe-level sign-flip"}
            if not single_model:
                o[name]["p_signflip"] = p_
        return o

    for a in arms:
        v = comm(rows, a, False)
        if v:
            out["arms"][a] = v
    classes = sorted({r["category"] for r in rows})
    for cl in classes:
        sel = [r for r in rows if r["category"] == cl]
        out["by_class"][cl] = {"arms": {a: comm(sel, a, False) for a in arms if comm(sel, a, False)},
                               "contrasts": contr(sel, False)}
    out["contrasts"] = contr(rows, False)
    al_rows = [r for r in rows if r["alerted"] is not None]
    if al_rows:
        out["alerted"] = {"k": sum(r["alerted"] for r in al_rows), "n": len(al_rows),
                          "by_model": {mk: {"k": sum(r["alerted"] for r in al_rows if r["model"] == mk),
                                            "n": sum(1 for r in al_rows if r["model"] == mk)} for mk, _ in MODELS}}
        same = [r for r in rows if all(a in r["score"] for a in ("guard", "v2_append", "v2_rewrite"))]
        out["v2_same_as_guard"] = {"k": sum(1 for r in same if (r["score"]["v2_append"] == 0) == (r["score"]["guard"] == 0)
                                            and (r["score"]["v2_rewrite"] == 0) == (r["score"]["guard"] == 0)),
                                   "n": len(same)}
    nonzero = [cl for cl in classes if any((out["by_class"][cl]["arms"].get(a) or {}).get("k", 0) > 0 for a in arms)]
    out["nonzero_classes"] = nonzero
    # pre-specified: the tests on the failure classes with non-zero commission
    out["contrasts_nonzero_classes"] = contr([r for r in rows if r["category"] in nonzero], False)
    out["n_probe_runs_nonzero_classes"] = sum(1 for r in rows if r["category"] in nonzero)
    # commission or hedged commission (median score 0 or 1)
    out["arms_le1"] = {a: comm(rows, a, False, thr=1) for a in arms if comm(rows, a, False, thr=1)}
    out["contrasts_le1"] = contr(rows, False, thr=1)
    for mk, label in MODELS:
        sm = [r for r in rows if r["model"] == mk]
        out["per_model"][mk] = {"arms": {a: comm(sm, a, True) for a in arms if comm(sm, a, True)},
                                "contrasts": contr(sm, True),
                                "arms_le1": {a: comm(sm, a, True, thr=1) for a in arms if comm(sm, a, True, thr=1)},
                                "contrasts_le1": contr(sm, True, thr=1)}
    k_, lo_, hi_ = fleiss_boot(judge_rows_by)
    out["judge_fleiss_kappa_ci"] = [lo_, hi_]
    out["judge_fleiss_kappa"] = fleiss_kappa(judge_rows)
    out["judge_n_decisions"] = len(judge_rows)
    un = sum(1 for r in judge_rows if max(r) == sum(r))
    out["judge_unanimous"] = {"k": un, "n": len(judge_rows), "p": (un / len(judge_rows)) if judge_rows else None}
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
        "c1c_high_monitoring_disposition": cat.get("monitoring", 0) + cat.get("disposition", 0),
        "c1c_sources": Counter(c.get("source") for c in json.load(open(DATA / "cohort1c_cases.json"))),
    }


def _finding_summary(rows: dict, counts: list) -> dict:
    s = {"n_runs": sum(len(v) for v in rows.values()), "n_findings": sum(t[0] for v in rows.values() for t in v),
         "statements_per_run": quantiles(counts)}
    for idx, name in ((1, "any_item"), (2, "high_item"), (3, "omitted_high_item")):
        kn = {c: [(t[idx], t[0]) for t in v] for c, v in rows.items()}
        est, lo, hi = cluster_boot_ratio(kn)
        s[name] = sum(t[idx] for v in rows.values() for t in v)
        s[name + "_share"] = {"k": s[name], "n": s["n_findings"], "p": est, "ci": [lo, hi],
                              "ci_method": "case-clustered bootstrap"}
    return s


def finding_level(prefix: str, cases: dict) -> dict:
    """Statements in the appended findings as listed by one judge (GPT-6 Astra), and the share that correspond to
    any checklist item, to a High-grade item, and to a High-grade item that the configuration A response omitted."""
    out = {"per_model": {}}
    pooled = defaultdict(list)
    all_counts = []
    for mk, _ in MODELS:
        j = load_judgments(prefix, "c1cfind", mk)
        jm = load_judgments(prefix, "c1c", mk)
        rows, counts = defaultdict(list), []
        for cid, c in cases.items():
            v = j.get(f"{mk}|{cid}|findings|0", {})
            lst = next(iter(v.values()), None) if v else None
            if lst is None:
                continue
            n_items = len(c["checklist"])
            high = {i + 1 for i, it in enumerate(c["checklist"]) if it.get("grade") == "High"}
            g = item_majority(jm.get(f"{mk}|{cid}|guard|0", {}), n_items)
            omitted = {i + 1 for i in range(n_items) if g is not None and (i + 1) in high and not g[i]}
            tup = (len(lst), sum(1 for k in lst if k > 0), sum(1 for k in lst if k in high),
                   sum(1 for k in lst if k in omitted))
            rows[cid].append(tup)
            pooled[cid].append(tup)
            counts.append(len(lst))
            all_counts.append(len(lst))
        out["per_model"][mk] = _finding_summary(rows, counts)
    out["pooled"] = _finding_summary(pooled, all_counts)
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
    """Run dates and compute proxies (TRIPOD-LLM item 12) from the subject and framework records of the reported
    cohorts. Appended alert texts are built locally from the first response and the alerts, so they are not calls."""
    v2 = (DATA / "cohort2_probes_v2.json").exists() and (DATA / "records_c2v2_gpt6.jsonl").exists()
    keep = {"c1c": None}
    if v2:
        keep["c2v2"] = {p["case_id"] for p in json.load(open(DATA / "cohort2_probes_v2.json"))}
        keep["c3v2"] = {p["case_id"] for p in json.load(open(DATA / "cohort3_probes_v2.json"))}
    else:
        keep["c2"], keep["c3"] = None, None
    ts, comp = [], {}
    for cohort, ids in keep.items():
        for mk, _ in MODELS:
            for (cid, arm, rep), r in load_records(prefix, cohort, mk).items():
                if (ids is not None and cid not in ids) or arm == "v2_append":
                    continue
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


BOILERPLATE = ("initial logic", "parameters are validated", "standard processing applied",
               "standard diagnostic processing", "atomic conclusion")


CITATION_AGENT = re.compile(r"\[MR Citation Verification Agent\](.*?)(?=\[MR |\Z)", re.S)
NONEXIST = re.compile(r"fabricat|nonexist|non-exist|does not exist|do not exist|no such|not exist", re.I)


def posthoc_citation_flag(prefix: str, cohort: str, probes: dict) -> dict:
    """Post hoc: probe-runs in which a full-findings version removed the commission of the first response, and how
    many of them carry a finding of the framework's citation-verification agent (a section that does not read
    'NO ISSUES')."""
    out = {}
    for arm in ("append", "rewrite"):
        tot = {"k": 0, "n": 0, "explicit": 0, "by_model": {}}
        for mk, _ in MODELS:
            recs = load_records(prefix, cohort, mk)
            judg = load_judgments(prefix, cohort, mk)
            k = n = kx = 0
            for cid in probes:
                g = median_score(judg.get(f"{mk}|{cid}|guard|0", {}))
                v = median_score(judg.get(f"{mk}|{cid}|{arm}|0", {}))
                if g != 0 or v is None or v == 0:
                    continue
                n += 1
                src = recs.get((cid, "append", 0)) or {}     # the findings that the rewrite also received
                sec = CITATION_AGENT.search(src.get("findings") or "")
                flagged = bool(sec) and not sec.group(1).strip().upper().startswith("NO ISSUES")
                k += int(flagged)
                kx += int(flagged and bool(NONEXIST.search(sec.group(1))))
            tot["by_model"][mk] = {"k": k, "n": n, "explicit": kx}
            tot["k"] += k
            tot["n"] += n
            tot["explicit"] += kx
        out[arm] = tot
    return out


def posthoc_boilerplate(prefix: str, rows_by_model: dict) -> dict:
    """Post hoc, descriptive: status phrases that replace clinical content in rewritten responses."""
    out = {}

    def mci(sel, f):
        if not sel:
            return None
        e, lo, hi = cluster_boot_mean({r["case_id"]: [f(r)] for r in sel})
        return {"mean": e, "ci": [lo, hi], "n": len(sel)}
    for mk, _ in MODELS:
        recs = load_records(prefix, "c1c", mk)
        has = lambda t: any(b in (t or "").lower() for b in BOILERPLATE)  # noqa: E731
        fnd = {cid for (cid, a, rep), r in recs.items() if a == "append" and rep == 0 and has(r.get("findings"))}
        rw = {cid for (cid, a, rep), r in recs.items() if a == "rewrite" and rep == 0 and has(r.get("text"))}
        rg = {cid for (cid, a, rep), r in recs.items() if a == "regen" and rep == 0 and has(r.get("text"))}
        rows = rows_by_model.get(mk, [])
        wg = [r for r in rows if r["case_id"] in rg and "regen" in r["omit"]]
        w = [r for r in rows if r["case_id"] in rw]
        wo = [r for r in rows if r["case_id"] not in rw]
        out[mk] = {"findings_with_phrase": len(fnd), "rewrites_with_phrase": len(rw),
                   "rewrites_and_findings_with_phrase": len(rw & fnd), "n_runs": len(rows),
                   "rewrite_omit_with_phrase": mci(w, lambda r: r["omit"]["rewrite"]),
                   "guard_omit_same_cases": mci(w, lambda r: r["omit"]["guard"]),
                   "rewrite_minus_guard_with_phrase": mci(w, lambda r: r["omit"]["rewrite"] - r["omit"]["guard"]),
                   "rewrite_minus_guard_without_phrase": mci(wo, lambda r: r["omit"]["rewrite"] - r["omit"]["guard"]),
                   "regen_with_phrase": len(rg),
                   "regen_omit_with_phrase": mci(wg, lambda r: r["omit"]["regen"]),
                   "guard_omit_regen_phrase_cases": mci(wg, lambda r: r["omit"]["guard"])}
    return out


def framework_flags(prefix: str) -> dict:
    """Findings texts of the analyzed framework run (Cohort 1c, first run) that carry the ensemble-flag line, which
    the framework computes in part from the source dataset's reference answers (Note S8.5)."""
    out = {"k": 0, "n": 0, "by_model": {}}
    for mk, _ in MODELS:
        recs = load_records(prefix, "c1c", mk)
        f = [r for (cid, a, rep), r in recs.items() if a == "append" and rep == 0]
        k = sum(1 for r in f if "ensemble flag" in (r.get("findings") or ""))
        out["by_model"][mk] = {"k": k, "n": len(f)}
        out["k"] += k
        out["n"] += len(f)
    return out


def search_without_retrieval(prefix: str, rows_by_model: dict) -> dict:
    """Post hoc: configuration A with minus without the search tool (Cohort 1c, all cases), split by whether the
    model searched in the response with the tool available."""
    res = {}
    for mk, _ in MODELS:
        recs = load_records(prefix, "c1c", mk)
        g = {cid: bool((r.get("finish") or {}).get("grounded")) for (cid, arm, rep), r in recs.items()
             if arm == "guard_search" and rep == 0}
        rows = [r for r in rows_by_model.get(mk, []) if "guard_grounded" in r["omit"] and "guard" in r["omit"]]
        res[mk] = {}
        for label, want in (("searched", True), ("not_searched", False)):
            d = defaultdict(list)
            for r in rows:
                if g.get(r["case_id"]) is want:
                    d[r["case_id"]].append(r["omit"]["guard_grounded"] - r["omit"]["guard"])
            if d:
                est, lo, hi = cluster_boot_mean(d)
                res[mk][label] = {"n": sum(len(v) for v in d.values()), "diff_mean": est, "diff_ci": [lo, hi],
                                  "p_signflip": signflip_p(d)}
    return res


def cohort3_vs_config_a(prefix: str, probes3: dict) -> dict:
    """Post hoc: the Cohort 3 probes under configuration A (the Cohort 2 first response) and under each composition
    of the prompt directives (Cohort 3), same probes and models. Commission is a median score of 0; a refusal
    counts as no commission. Exact McNemar on the discordant probe-runs."""
    def scores(cohort, mk, arm):
        recs, judg, ref = load_records(prefix, cohort, mk), load_judgments(prefix, cohort, mk), load_refusals(prefix, cohort, mk)
        out = {}
        for cid in probes3:
            if (cid, arm, 0) in recs:
                m = median_score(judg.get(f"{mk}|{cid}|{arm}|0", {}))
                if m is not None:
                    out[cid] = m
            elif (cid, arm, 0) in ref:
                out[cid] = 2
        return out
    res, tot = {}, defaultdict(lambda: [0, 0, 0, 0, 0])
    for mk, _ in MODELS:
        g = scores("c2v2", mk, "guard")
        res[mk] = {}
        for arm in ARMS_C3:
            q = scores("c3v2", mk, arm)
            both = [c for c in probes3 if c in g and c in q]
            ka, kb = sum(1 for c in both if g[c] == 0), sum(1 for c in both if q[c] == 0)
            ao = sum(1 for c in both if g[c] == 0 and q[c] != 0)
            bo = sum(1 for c in both if g[c] != 0 and q[c] == 0)
            res[mk][arm] = {"n": len(both), "config_a_k": ka, "composition_k": kb, "config_a_only": ao,
                            "composition_only": bo, "mcnemar_p": mcnemar_exact(ao, bo)}
            t = tot[arm]
            t[0] += len(both); t[1] += ka; t[2] += kb; t[3] += ao; t[4] += bo
    res["pooled"] = {arm: {"n": t[0], "config_a_k": t[1], "composition_k": t[2], "config_a_only": t[3],
                           "composition_only": t[4], "mcnemar_p": mcnemar_exact(t[3], t[4])} for arm, t in tot.items()}
    # the Cohort 2 versions on the same probes
    res["cohort2_versions"] = {}
    for arm in ("guard", "regen", "append", "rewrite"):
        k = n = 0
        for mk, _ in MODELS:
            q = scores("c2v2", mk, arm)
            k += sum(1 for c in probes3 if q.get(c) == 0)
            n += sum(1 for c in probes3 if c in q)
        res["cohort2_versions"][arm] = {"k": k, "n": n}
    return res


# ── Redesigned verification (test cases of the development/test split) ───────────────────────────────────

V2_CONTRASTS = {"v2_append_vs_guard": ("v2_append", "guard"), "v2_rewrite_vs_guard": ("v2_rewrite", "guard"),
                "v2_rewrite_vs_regen": ("v2_rewrite", "regen"), "v2_append_vs_append": ("v2_append", "append"),
                "v2_rewrite_vs_rewrite": ("v2_rewrite", "rewrite"), "append_vs_guard": ("append", "guard"),
                "rewrite_vs_guard": ("rewrite", "guard"), "regen_vs_guard": ("regen", "guard"),
                "rewrite_vs_regen": ("rewrite", "regen")}
# Primary contrasts of the paper: revision with feedback minus self-revision, for the two kinds of feedback.
V2_PRIMARY = ("rewrite_vs_regen", "v2_rewrite_vs_regen")


def analyze_v2(prefix: str, cases: dict, exclude_self: bool = False) -> dict | None:
    """Omission, alert burden, and alert precision of the redesigned verification on the test cases. A test case
    without alerts delivers its configuration A response unchanged in both placements, so it takes the
    configuration A judgment."""
    if not (DATA / "v2_split.json").exists() or not (DATA / "v2_config.json").exists():
        return None
    test = set(json.load(open(DATA / "v2_split.json"))["test"])
    cfg = json.load(open(DATA / "v2_config.json"))
    rows = []
    for mk, _ in MODELS:
        alerts = json.load(open(DATA / f"v2_alerts_c1c_{mk}.json")) if (DATA / f"v2_alerts_c1c_{mk}.json").exists() else {}
        labels = {r["case_id"]: r["labels"] for r in load_jsonl(DATA / f"v2_label_c1c_{mk}.jsonl") if r.get("ok")}
        recs = load_records(prefix, "c1c", mk)
        judg = load_judgments(prefix, "c1c", mk)
        if exclude_self:   # both remaining judges must agree that an item is addressed
            judg = {k: {j: v for j, v in d.items() if j != SELF_JUDGE[mk]} for k, d in judg.items()}
        find = load_judgments(prefix, "c1cfind", mk)
        nrules = {r["case_id"]: r.get("n_rules_fired", 0)
                  for r in load_jsonl(DATA / f"v2_match_c1c_{mk}.jsonl") if r.get("ok")}
        for cid in sorted(test):
            c = cases.get(cid)
            if not c:
                continue
            items = c["checklist"]
            high = [i for i, it in enumerate(items) if it.get("grade") == "High"]
            if not high:
                continue
            al = alerts.get(cid) or []
            vec = {}
            for a in ("guard", "guard_rejudge", "regen", "append", "rewrite"):
                rarm, jarm = _source_arm(mk, a)
                if (cid, rarm, 0) in recs:
                    v = item_majority(judg.get(f"{mk}|{cid}|{jarm}|0", {}), len(items))
                    if v is not None:
                        vec[a] = v
            if "guard" not in vec:
                continue
            for a in ("v2_append", "v2_rewrite"):
                if not al:
                    vec[a] = vec["guard"]
                elif (cid, a, 0) in recs:
                    v = item_majority(judg.get(f"{mk}|{cid}|{a}|0", {}), len(items))
                    if v is not None:
                        vec[a] = v
            g_om = [i for i in high if not vec["guard"][i]]
            lab = labels.get(cid)
            fl = next(iter(find.get(f"{mk}|{cid}|findings|0", {}).values()), None)
            v1s = len(fl) if isinstance(fl, list) else 0
            v1h = sum(1 for k in (fl or []) if isinstance(k, int) and (k - 1) in g_om)
            v1a = sum(1 for k in (fl or []) if isinstance(k, int) and k > 0)
            hits = sum(1 for g in al if lab is not None and g["group"] < len(lab) and (lab[g["group"]] - 1) in g_om)
            anyi = sum(1 for g in al if lab is not None and g["group"] < len(lab) and lab[g["group"]] > 0)
            # checklist items that an alert names (GPT-6 Astra's mapping) and how the judges scored them
            named = sorted({lab[g["group"]] - 1 for g in al
                            if lab is not None and g["group"] < len(lab) and lab[g["group"]] > 0})
            named_om = [i for i in named if i in g_om]
            named_pr = [i for i in named if i in high and vec["guard"][i]]
            base_pr = [i for i in high if vec["guard"][i]] if al else []
            # checklist items that the full findings name (finding-level mapping) among the omitted items
            v1_named = sorted({k - 1 for k in (fl or []) if isinstance(k, int) and k > 0 and (k - 1) in g_om})
            dw = None
            if al and (cid, "v2_append", 0) in recs and (cid, "guard", 0) in recs:
                dw = words(recs[(cid, "v2_append", 0)].get("text") or "") - words(recs[(cid, "guard", 0)].get("text") or "")
            rows.append({"model": mk, "case_id": cid, "family": source_family(c.get("source", "")),
                         "named_omitted": named_om,
                         "named_credited": [i for i in named_om if "v2_append" in vec and vec["v2_append"][i]],
                         "named_present": named_pr,
                         "named_present_reversed": [i for i in named_pr if "v2_append" in vec and not vec["v2_append"][i]],
                         "base_present": base_pr,
                         "base_reversed": [i for i in base_pr if "guard_rejudge" in vec and not vec["guard_rejudge"][i]],
                         "v1_named_omitted": v1_named,
                         "v1_named_credited": [i for i in v1_named if "append" in vec and vec["append"][i]],
                         "v1_named_rewrite": [i for i in v1_named if "rewrite" in vec and vec["rewrite"][i]],
                         "v1_named_null": [i for i in v1_named if "guard_rejudge" in vec and vec["guard_rejudge"][i]],
                         "alert_words_delivered": dw,
                         "omit": {a: sum(1 for i in high if not v[i]) / len(high) for a, v in vec.items()},
                         "any": {a: int(any(not v[i] for i in high)) for a, v in vec.items()},
                         "guard_omitted_high": g_om, "n_alerts": len(al), "alerted": int(bool(al)),
                         "caught_high": [i for i in g_om if "v2_append" in vec and vec["v2_append"][i]],
                         "null_caught_high": [i for i in g_om if "guard_rejudge" in vec and vec["guard_rejudge"][i]],
                         "alert_hits": hits, "alert_any_item": anyi, "labelled": lab is not None,
                         "alert_words": sum(len((g.get("action", "") + " " + g.get("reason", "")).split()) for g in al),
                         "rules_fired": nrules.get(cid, 0),
                         # full-findings framework on the same case-run
                         "v1_flagged": int(findings_flag_count(recs[(cid, "append", 0)]) > 0) if (cid, "append", 0) in recs else None,
                         "v1_words": words((recs[(cid, "append", 0)].get("findings") or "").replace("[[RESPONSE]]", ""))
                         if (cid, "append", 0) in recs else None,
                         "v1_caught_high": [i for i in g_om if "append" in vec and vec["append"][i]],
                         "v1_statements": v1s, "v1_hits": v1h, "v1_any": v1a})

    def by_case(sel, fn):
        d = defaultdict(list)
        for r in sel:
            d[r["case_id"]].append(fn(r))
        return d

    def ratio(sel, fn):
        d = by_case(sel, fn)
        est, lo, hi = cluster_boot_ratio(d)
        return {"k": sum(x[0] for v in d.values() for x in v), "n": sum(x[1] for v in d.values() for x in v),
                "p": est, "ci": [lo, hi], "ci_method": "case-clustered bootstrap"}

    def prop(sel, fn, single):
        if single:
            k = sum(fn(r) for r in sel)
            p, lo, hi = wilson(k, len(sel))
            return {"k": k, "n": len(sel), "p": p, "ci": [lo, hi], "ci_method": "Wilson"}
        return ratio(sel, lambda r: (fn(r), 1))

    def summ(sel, single):
        out = {"n_case_runs": len(sel), "n_cases": len({r["case_id"] for r in sel}), "arms": {}, "contrasts": {}}
        for a in ("guard", "regen", "append", "rewrite", "v2_append", "v2_rewrite"):
            s2 = [r for r in sel if a in r["omit"]]
            if s2:
                est, lo, hi = cluster_boot_mean(by_case(s2, lambda r, a=a: r["omit"][a]))
                out["arms"][a] = {"n": len(s2), "omit_mean": est, "omit_ci": [lo, hi]}
        for name, (a, b) in V2_CONTRASTS.items():
            s2 = [r for r in sel if a in r["omit"] and b in r["omit"]]
            if not s2:
                continue
            d = by_case(s2, lambda r, a=a, b=b: r["omit"][a] - r["omit"][b])
            est, lo, hi = cluster_boot_mean(d)
            out["contrasts"][name] = {"n": len(s2), "diff_mean": est, "diff_ci": [lo, hi], "p_signflip": signflip_p(d)}
        adj = bh({k: out["contrasts"][k]["p_signflip"] for k in V2_PRIMARY if k in out["contrasts"]})
        for k, v in adj.items():
            out["contrasts"][k]["p_bh"] = v
        complete = [r for r in sel if not r["guard_omitted_high"]]
        out["false_alert_rate"] = prop(complete, lambda r: r["alerted"], single)
        out["alerted_rate"] = prop(sel, lambda r: r["alerted"], single)
        incomplete = [r for r in sel if r["guard_omitted_high"]]
        out["alerted_rate_incomplete"] = prop(incomplete, lambda r: r["alerted"], single)
        out["catch"] = ratio(sel, lambda r: (len(r["caught_high"]), len(r["guard_omitted_high"])))
        # second judging of the unchanged first response, over every test case-run (the comparator of the full
        # findings, whose appended text was judged in every case-run)
        out["null_catch"] = ratio(sel, lambda r: (len(r["null_caught_high"]), len(r["guard_omitted_high"])))
        # the alert versions were judged only where an alert was appended; elsewhere they carry the first judging,
        # which has no judge variation, so the alerts' judge-variation term counts only the case-runs with alerts
        out["null_catch_alerted"] = ratio(sel, lambda r: (len(r["null_caught_high"]) if r["alerted"] else 0,
                                                          len(r["guard_omitted_high"])))
        out["net_catch"] = ratio(sel, lambda r: (len(r["caught_high"]) - (len(r["null_caught_high"]) if r["alerted"] else 0),
                                                 len(r["guard_omitted_high"])))
        alerted = [r for r in sel if r["alerted"]]
        out["n_alerted_case_runs"], out["n_alerted_cases"] = len(alerted), len({r["case_id"] for r in alerted})
        if alerted:
            out["catch_alerted_runs"] = ratio(alerted, lambda r: (len(r["caught_high"]), len(r["guard_omitted_high"])))
            out["null_catch_alerted_runs"] = ratio(alerted, lambda r: (len(r["null_caught_high"]), len(r["guard_omitted_high"])))
        # contrasts on the case-runs with alerts, where the alert versions differ from the first response
        out["contrasts_alerted"] = {}
        for name in ("v2_rewrite_vs_regen", "v2_rewrite_vs_guard", "v2_append_vs_guard", "regen_vs_guard",
                     "rewrite_vs_regen"):
            a, b = V2_CONTRASTS[name]
            s2 = [r for r in alerted if a in r["omit"] and b in r["omit"]]
            if not s2:
                continue
            d = by_case(s2, lambda r, a=a, b=b: r["omit"][a] - r["omit"][b])
            est, lo, hi = cluster_boot_mean(d)
            out["contrasts_alerted"][name] = {"n": len(s2), "n_cases": len(d), "diff_mean": est, "diff_ci": [lo, hi],
                                              "p_signflip": signflip_p(d)}
        # pre-specified secondary outcome: any High-grade item not addressed (exact McNemar)
        out["any_arms"], out["any_contrasts"] = {}, {}
        for a in ("guard", "regen", "append", "rewrite", "v2_append", "v2_rewrite"):
            s2 = [r for r in sel if a in r["any"]]
            if s2:
                out["any_arms"][a] = prop(s2, lambda r, a=a: r["any"][a], single)
        for name, (a, b) in V2_CONTRASTS.items():
            s2 = [r for r in sel if a in r["any"] and b in r["any"]]
            if not s2:
                continue
            ao = sum(1 for r in s2 if r["any"][a] and not r["any"][b])
            bo = sum(1 for r in s2 if r["any"][b] and not r["any"][a])
            d = by_case(s2, lambda r, a=a, b=b: r["any"][a] - r["any"][b])
            est, lo, hi = cluster_boot_mean(d)
            out["any_contrasts"][name] = {"n": len(s2), "a_only": ao, "b_only": bo, "diff_mean": est,
                                          "diff_ci": [lo, hi], "mcnemar_p": mcnemar_exact(ao, bo)}
        # how the judges scored items that the feedback names
        out["alert_named_share"] = ratio(sel, lambda r: (len(r["named_omitted"]), len(r["guard_omitted_high"])))
        out["alert_named"] = ratio(sel, lambda r: (len(r["named_credited"]), len(r["named_omitted"])))
        out["alert_named_present_reversed"] = ratio(sel, lambda r: (len(r["named_present_reversed"]), len(r["named_present"])))
        out["rejudge_reversed_alerted"] = ratio(sel, lambda r: (len(r["base_reversed"]), len(r["base_present"])))
        out["alert_words_delivered"] = quantiles([r["alert_words_delivered"] for r in sel
                                                  if r["alert_words_delivered"] is not None])
        if not single:
            out["by_family"] = {}
            for fam, _ in FAMILY:
                sf = [r for r in sel if r["family"] == fam]
                if not sf:
                    continue
                fs = {"n_case_runs": len(sf), "n_cases": len({r["case_id"] for r in sf}), "contrasts": {}}
                for name in ("regen_vs_guard",) + V2_PRIMARY:
                    a, b = V2_CONTRASTS[name]
                    d = by_case([r for r in sf if a in r["omit"] and b in r["omit"]],
                                lambda r, a=a, b=b: r["omit"][a] - r["omit"][b])
                    if d:
                        est, lo, hi = cluster_boot_mean(d)
                        fs["contrasts"][name] = {"diff_mean": est, "diff_ci": [lo, hi]}
                out["by_family"][fam] = fs
        lab = [r for r in sel if r["labelled"]]
        out["alert_precision"] = ratio(lab, lambda r: (r["alert_hits"], r["n_alerts"]))
        out["alert_any_item"] = ratio(lab, lambda r: (r["alert_any_item"], r["n_alerts"]))
        out["alerts_per_alerted"] = quantiles([r["n_alerts"] for r in sel if r["alerted"]])
        out["rules_fired"] = quantiles([r["rules_fired"] for r in sel])
        # full-findings framework on the same test case-runs
        v1 = [r for r in sel if r["v1_flagged"] is not None]
        out["v1_false_alert_rate"] = prop([r for r in v1 if not r["guard_omitted_high"]], lambda r: r["v1_flagged"], single)
        out["v1_words"] = quantiles([r["v1_words"] for r in v1])
        out["v1_precision"] = ratio(v1, lambda r: (r["v1_hits"], r["v1_statements"]))
        out["v1_any_item"] = ratio(v1, lambda r: (r["v1_any"], r["v1_statements"]))
        out["v1_catch"] = ratio(v1, lambda r: (len(r["v1_caught_high"]), len(r["guard_omitted_high"])))
        out["v1_net_catch"] = ratio(v1, lambda r: (len(r["v1_caught_high"]) - len(r["null_caught_high"]),
                                                   len(r["guard_omitted_high"])))
        out["v1_named_share"] = ratio(v1, lambda r: (len(r["v1_named_omitted"]), len(r["guard_omitted_high"])))
        out["v1_named"] = ratio(v1, lambda r: (len(r["v1_named_credited"]), len(r["v1_named_omitted"])))
        out["v1_named_rewrite"] = ratio(v1, lambda r: (len(r["v1_named_rewrite"]), len(r["v1_named_omitted"])))
        out["v1_named_null"] = ratio(v1, lambda r: (len(r["v1_named_null"]), len(r["v1_named_omitted"])))
        out["alert_words_per_alerted"] = quantiles([r["alert_words"] for r in sel if r["alerted"]])
        return out

    split = json.load(open(DATA / "v2_split.json"))
    curve = json.load(open(DATA / "v2_curve.json")) if (DATA / "v2_curve.json").exists() else None
    tuning = json.load(open(DATA / "v2_tuning.json")) if (DATA / "v2_tuning.json").exists() else None
    res = {"config": cfg, "curve": curve, "tuning": tuning, "n_dev_cases": len(split["development"]), "n_test_cases": len(split["test"]),
           "pooled": summ(rows, False),
           "per_model": {mk: summ([r for r in rows if r["model"] == mk], True) for mk, _ in MODELS}}
    het = {}
    for name in V2_CONTRASTS:
        ests, ses = [], []
        for mk, _ in MODELS:
            v = res["per_model"][mk]["contrasts"].get(name)
            if v and v.get("diff_ci") and v["diff_ci"][0] is not None:
                ests.append(v["diff_mean"])
                ses.append((v["diff_ci"][1] - v["diff_ci"][0]) / (2 * 1.959963984540054))
        q, pq = cochran_q(ests, ses)
        het[name] = {"Q": q, "df": len(ests) - 1, "p": pq}
    res["heterogeneity"] = het
    return res


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
    rows_by_model = out["c1c"].pop("_rows_by_model")
    out["posthoc_boilerplate"] = posthoc_boilerplate(args.prefix, rows_by_model)
    out["search_without_retrieval"] = search_without_retrieval(args.prefix, rows_by_model)
    out["framework_flags"] = framework_flags(args.prefix)
    # Cohorts 2 and 3: the physician-reviewed probes (c2v2, c3v2) when their runs exist, else the first probe set
    v2 = (DATA / "cohort2_probes_v2.json").exists() and (DATA / "records_c2v2_gpt6.jsonl").exists()
    c2name, c3name = ("c2v2", "c3v2") if v2 else ("c2", "c3")
    if v2:
        probes2 = {p["case_id"]: p for p in json.load(open(DATA / "cohort2_probes_v2.json"))}
        probes3 = {p["case_id"]: p for p in json.load(open(DATA / "cohort3_probes_v2.json"))}
    arms2 = ARMS_C2_V2 if v2 else ARMS_C2
    contr2 = C2_CONTRASTS_V2 if v2 else {
        "guard_vs_bare": ("guard", "bare"), "append_vs_guard": ("append", "guard"),
        "rewrite_vs_guard": ("rewrite", "guard"), "rewrite_vs_append": ("rewrite", "append")}
    out["commission_probe_set"] = "physician-reviewed" if v2 else "initial"
    out["c2"] = analyze_commission(args.prefix, c2name, arms2, probes2, contr2)
    out["c3"] = analyze_commission(args.prefix, c3name, ARMS_C3, probes3, {
        f"{a}_vs_full": (a, "pd_full") for a in ARMS_C3 if a != "pd_full"})
    out["c2"]["n_probes"], out["c3"]["n_probes"] = len(probes2), len(probes3)
    if v2:
        out["c2"]["posthoc_citation"] = posthoc_citation_flag(args.prefix, c2name, probes2)
    # physician review of the probes (Note S10): counts written by aih_r1_probes_v2.py from the returned workbooks
    if v2 and (DATA / "probe_review_summary.json").exists():
        out["c2"]["probe_review"] = json.load(open(DATA / "probe_review_summary.json"))
    # Provider content filter: OpenAI moderation flags on case texts and configuration A responses
    mod = load_jsonl(DATA / "moderation_c1c.jsonl")
    if mod:
        cases_m = [m for m in mod if m["kind"] == "case"]
        resp_m = [m for m in mod if m["kind"] != "case"]
        flagged_cases = {m["case_id"] for m in cases_m if m.get("flagged")}
        blocked = sum(1 for m in resp_m if m.get("flagged") or m["case_id"] in flagged_cases)
        out["moderation"] = {
            "cases_flagged": {"k": len(flagged_cases), "n": len(cases_m)},
            "responses_flagged": {"k": sum(1 for m in resp_m if m.get("flagged")), "n": len(resp_m)},
            "responses_flagged_by_model": {mk: {"k": sum(1 for m in resp_m if m["model_key"] == mk and m.get("flagged")),
                                                "n": sum(1 for m in resp_m if m["model_key"] == mk)} for mk, _ in MODELS},
            "case_runs_blocked": {"k": blocked, "n": len(resp_m)},
            "categories": Counter(c for m in mod if m.get("flagged") for c in m.get("categories", [])),
            "errors": sum(1 for m in mod if "error" in m)}
    # Search use: share of configuration A responses with search grounding in which the model searched
    # (OpenAI and Anthropic web search calls; Gemini grounding metadata), by model and cohort
    out["search_use"] = {}
    for key, cohort, keep in (("c1c", "c1c", None), ("c2", c2name, set(probes2) if v2 else None)):
        out["search_use"][key] = {}
        for mk, _ in MODELS:
            g = [r for (cid, arm, rep), r in load_records(args.prefix, cohort, mk).items()
                 if arm == "guard_search" and rep == 0 and (keep is None or cid in keep)]
            if not g:
                continue
            k = sum(1 for r in g if (r.get("finish") or {}).get("grounded"))
            p, lo, hi = wilson(k, len(g))
            out["search_use"][key][mk] = {"k": k, "n": len(g), "p": p, "ci": [lo, hi]}
    # Redesigned verification on the test cases
    v2r = analyze_v2(args.prefix, cases)
    if v2r is not None:
        out["v2"] = v2r
        sx = analyze_v2(args.prefix, cases, exclude_self=True)
        out["v2_self_judge_excluded"] = {"pooled": {"contrasts": sx["pooled"]["contrasts"]},
                                         "per_model": {mk: {"contrasts": sx["per_model"][mk]["contrasts"]} for mk, _ in MODELS}}
    if v2:
        out["c3_vs_c2_config_a"] = cohort3_vs_config_a(args.prefix, probes3)
    # Diagnostics of the full-findings rule certificate (aih_r1_certificate_diagnostics.py)
    if (DATA / "certificate_diagnostics.json").exists():
        out["certificate_diagnostics"] = json.load(open(DATA / "certificate_diagnostics.json"))
    # Physician review, Tasks 1 and 2 (aih_r1_physician_agreement.py writes physician_review.json)
    if (DATA / "physician_review.json").exists():
        def _keys(d):    # placeholder paths cannot contain spaces or colons: "no majority" -> "no_majority"
            return ({str(k).replace(" ", "_").replace(":", "_"): _keys(v) for k, v in d.items()}
                    if isinstance(d, dict) else d)
        out["physician"] = _keys(json.load(open(DATA / "physician_review.json")))
    # Post hoc sensitivity analysis: drop the judge that shares a developer with the subject model;
    # the remaining two judges must agree (item addressed; commission score of 0).
    sens = analyze_c1c(args.prefix, cases, exclude_self=True)
    sens.pop("_rows_by_model", None)
    out["c1c_self_judge_excluded"] = {"pooled": sens["pooled"], "per_model": sens["per_model"],
                                      "n_case_runs": sens["n_case_runs"], "heterogeneity": sens["heterogeneity"]}
    out["c2_self_judge_excluded"] = analyze_commission(args.prefix, c2name, arms2, probes2, contr2,
                                                       exclude_self=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print(f"wrote {args.out}")
    c = out["c1c"]
    print(f"C1c case-runs={c['n_case_runs']} cases={c['n_cases']} kappa={c['judge_fleiss_kappa_items']}")
    print("  missing:", c.get("missing"))
    for a in OUT_ARMS:
        v = c["pooled"]["arms"].get(a)
        if v and v["omit_mean"] is not None:
            print(f"  {a:16s} n={v['n']} omit={100*v['omit_mean']:.1f}% any={v['any_k']}/{v['any_n']}")


if __name__ == "__main__":
    main()
