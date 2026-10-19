"""Redesigned verification for AIH026370137 R1: case-grounded rules, blind cross-model drafts, calibrated alerts.

Stages (each writes its own file in mira-results:/aih_r1/, one writer per file; all resume):
  facts   GPT-5.5 selects, from the fixed fact vocabulary of the rule library, the findings, history, diagnoses,
          and completed actions that the case states for this patient (one call per case), and the actions and
          the final or working diagnoses of each configuration A response, read in full (one call per response).
  drafts  each of the three subject models lists the actions that the case requires, without seeing any response.
  match   per case-run, the candidates are the drafts of the two models from other developers and the actions
          required by the rules that fire (Boolean evaluation of the rule library on the grounded facts); GPT-5.5
          groups candidates that describe the same action and decides whether the full response contains each.
  label   GPT-6 Astra maps each candidate group to the checklist item that it addresses, or to none.
Selection of alerts (scores, conformal threshold, at most K alerts) runs locally (aih_r1_v2_select.py).

Run: modal run --detach aih_r1_v2.py::<stage> [--cohort c1c] [--split development|test|all]
"""
import modal

app = modal.App("aih-r1-v2")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("openai>=1.50.0", "anthropic>=0.70.0", "requests"))
results_vol = modal.Volume.from_name("mira-results")
SECRETS = [modal.Secret.from_name("openai"), modal.Secret.from_name("anthropic"), modal.Secret.from_name("google")]
MODELS = {"gpt6": "gpt-6-astra", "gemini31": "gemini-3.1-pro-preview", "claude55": "claude-opus-5-5"}
CASE_FILES = {"c1c": "cohort1c_cases.json", "c2v2": "cohort2_probes_v2.json"}

CASE_FACTS_PROMPT = """You will read a clinical case and a fixed list of fact names.

Select every fact name that the case states about this patient: a current finding, a history item, a diagnosis \
that the case gives as established, or an action that the case says was already done for this patient. Do not \
select a name for a finding that the case negates, that appears only in a question or in answer options, that is \
hypothetical, or that belongs to someone other than the patient.

Return only JSON: {{"present": ["fact_name", ...]}}

FACT NAMES:
{names}

CASE:
{case}"""

RESPONSE_FACTS_PROMPT = """You will read a clinical response written for a patient and two fixed lists of names.

1. From ACTION NAMES, select every action that the response recommends, orders, or performs for this patient, \
anywhere in the response. Do not select an action that the response mentions only to reject it or as something to \
avoid.
2. From DIAGNOSIS NAMES, select the diagnoses that the response gives as its final or working diagnosis for this \
patient. Do not select diagnoses that appear only in a differential list or that the response rules out.

Return only JSON: {{"actions": ["name", ...], "final_diagnoses": ["name", ...]}}

ACTION NAMES:
{actions}

DIAGNOSIS NAMES:
{dx}

RESPONSE:
{response}"""

DRAFT_PROMPT = """Read this clinical case before any recommendation has been written for it.

List the actions that this patient requires now: diagnostic tests, treatments, escalation or referral, monitoring, \
and disposition. Include only actions that this specific patient requires, not routine care that the case does not \
call for. List at most 10 actions, the most important first. For each action, give a reason of at most 15 words and \
your confidence, from 0 to 1, that a careful clinician would consider the action required for this patient.

Return only JSON: {{"actions": [{{"action": "...", "reason": "...", "confidence": 0.0}}]}}

CASE:
{case}"""

MATCH_PROMPT = """You will read a clinical case, a clinical response written for it, and a numbered list of \
candidate actions that reviewers proposed for the case.

Group the candidates that describe the same action. For each group, decide whether the response already contains \
that action or an equivalent one ("yes"), contains part of it or a weaker form ("partly"), or does not contain it \
("no"). Every candidate number must appear in exactly one group.

Return only JSON: {{"groups": [{{"members": [1, 4], "action": "short wording of the action", "in_response": \
"yes|partly|no"}}]}}

CASE:
{case}

RESPONSE:
{response}

CANDIDATES:
{candidates}"""

LABEL_PROMPT = """You will read a clinical case, its numbered checklist of required actions, and a numbered list of \
proposed actions.

For each proposed action, give the number of the checklist item that it addresses, or 0 if it addresses none.

Return only JSON: {{"labels": [{{"n": 1, "item": 0}}]}}

CASE:
{case}

CHECKLIST:
{items}

PROPOSED ACTIONS:
{actions}"""


def _json(text):
    import json
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _gpt(model, prompt, effort="low", max_tokens=6000, json_mode=True, flex=False):
    import time
    import openai
    client = openai.OpenAI()
    kw = {"model": model, "input": prompt, "max_output_tokens": max_tokens, "reasoning": {"effort": effort}}
    if json_mode:
        kw["text"] = {"format": {"type": "json_object"}}
    if flex:
        kw["service_tier"] = "flex"
    for attempt in range(6):
        try:
            return client.responses.create(timeout=900, **kw).output_text or ""
        except Exception:
            if attempt == 5:
                raise
            time.sleep([5, 15, 45, 90, 180][attempt])


def _call(mk, prompt, max_tokens=4000):
    """One synchronous call to a subject model at low reasoning effort."""
    import os
    import time
    import requests
    if mk == "gpt6":
        return _gpt(MODELS[mk], prompt, effort="low", max_tokens=max_tokens, json_mode=True, flex=True)
    if mk == "claude55":
        import anthropic
        client = anthropic.Anthropic()
        for attempt in range(6):
            try:
                msg = client.messages.create(model=MODELS[mk], max_tokens=max_tokens, output_config={"effort": "low"},
                                             messages=[{"role": "user", "content": prompt}])
                return "".join(getattr(b, "text", "") or "" for b in msg.content if getattr(b, "type", "") == "text")
            except Exception:
                if attempt == 5:
                    raise
                time.sleep([5, 15, 45, 90, 180][attempt])
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS[mk]}:generateContent"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingLevel": "low"}}}
    for attempt in range(6):
        try:
            r = requests.post(url, params={"key": key}, json=body, timeout=300)
            r.raise_for_status()
            cand = (r.json().get("candidates") or [{}])[0]
            return "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
        except Exception:
            if attempt == 5:
                raise
            time.sleep([5, 15, 45, 90, 180][attempt])


def _io(cohort):
    import json
    from pathlib import Path
    base = Path("/results/aih_r1")
    cases = {c["case_id"]: c for c in json.load(open(base / CASE_FILES[cohort]))}
    split = json.load(open(base / "v2_split.json")) if cohort == "c1c" else None
    rf = json.load(open(base / "v2_rules_facts.json"))
    return base, cases, split, rf


def _ids(cases, split, which):
    if which.startswith("ids:"):
        return [c for c in which[4:].split(",") if c in cases]
    if split is None or which == "all":
        return sorted(cases)
    return [c for c in split[which] if c in cases]


def _guard(base, cohort, mk):
    import json
    src = base / (f"records_{cohort}_{mk}__fx.jsonl" if cohort == "c1c" else f"records_{cohort}_{mk}.jsonl")
    out = {}
    for line in open(src):
        r = json.loads(line)
        if r.get("ok") and r.get("arm") == "guard" and int(r.get("rep", 0)) == 0:
            out[r["case_id"]] = r.get("text") or ""
    return out


def _done(path, key):
    import json
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                if r.get("ok"):
                    done.add(r[key])
            except Exception:
                pass
    return done


def _pool(fn, items, workers, path):
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock, n = threading.Lock(), [0]
    fh = open(path, "a")

    def run(it):
        try:
            rec = fn(it)
        except Exception as e:
            rec = {**(it if isinstance(it, dict) else {"item": it}), "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n[0] += 1
            if n[0] % 50 == 0:
                results_vol.commit()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(run, it) for it in items]))
    fh.close()
    results_vol.commit()
    return n[0]


@app.function(image=img, secrets=SECRETS, volumes={"/results": results_vol}, timeout=86400)
def facts(cohort: str = "c1c", split: str = "all", workers: int = 16) -> dict:
    base, cases, sp, rf = _io(cohort)
    kinds = rf["facts"]
    names_all = ", ".join(sorted(kinds))
    actions = ", ".join(sorted(f for f, k in kinds.items() if k == "action"))
    dx = ", ".join(sorted(f for f, k in kinds.items() if k == "dx"))
    ids = _ids(cases, sp, split)
    out = {}
    # case facts (one writer: this function)
    cpath = base / f"v2_casefacts_{cohort}.jsonl"
    done = _done(cpath, "case_id")

    def one_case(cid):
        raw = _gpt("gpt-5.5", CASE_FACTS_PROMPT.format(names=names_all, case=cases[cid]["case_text"]))
        js = _json(raw) or {}
        present = [f for f in js.get("present", []) if f in kinds]
        return {"case_id": cid, "ok": isinstance(js.get("present"), list), "present": present, "raw": raw[:3000]}
    out["case"] = _pool(one_case, [c for c in ids if c not in done], workers, cpath)
    for mk in MODELS:
        guard = _guard(base, cohort, mk)
        rpath = base / f"v2_respfacts_{cohort}_{mk}.jsonl"
        done = _done(rpath, "case_id")

        def one_resp(cid, mk=mk, guard=guard):
            raw = _gpt("gpt-5.5", RESPONSE_FACTS_PROMPT.format(actions=actions, dx=dx, response=guard[cid]))
            js = _json(raw) or {}
            ok = isinstance(js.get("actions"), list) and isinstance(js.get("final_diagnoses"), list)
            return {"case_id": cid, "model_key": mk, "ok": ok,
                    "actions": [f for f in js.get("actions", []) if kinds.get(f) == "action"],
                    "final_diagnoses": [f for f in js.get("final_diagnoses", []) if kinds.get(f) == "dx"],
                    "raw": raw[:3000]}
        out[mk] = _pool(one_resp, [c for c in ids if c in guard and c not in done], workers, rpath)
    print("[v2 facts]", out)
    return out


@app.function(image=img, secrets=SECRETS, volumes={"/results": results_vol}, timeout=86400)
def drafts(cohort: str = "c1c", split: str = "all", workers: int = 12) -> dict:
    base, cases, sp, _ = _io(cohort)
    ids = _ids(cases, sp, split)
    out = {}
    for mk in MODELS:
        path = base / f"v2_drafts_{cohort}_{mk}.jsonl"
        done = _done(path, "case_id")

        def one(cid, mk=mk):
            raw = _call(mk, DRAFT_PROMPT.format(case=cases[cid]["case_text"]))
            js = _json(raw) or {}
            acts = []
            for a in js.get("actions", [])[:10]:
                try:
                    acts.append({"action": str(a["action"])[:300], "reason": str(a.get("reason", ""))[:200],
                                 "confidence": max(0.0, min(1.0, float(a.get("confidence", 0))))})
                except Exception:
                    continue
            return {"case_id": cid, "drafter": mk, "ok": bool(acts), "actions": acts, "raw": raw[:4000]}
        out[mk] = _pool(one, [c for c in ids if c not in done], workers, path)
    print("[v2 drafts]", out)
    return out


def rule_candidates(rf, case_present, resp_actions, resp_final_dx):
    """Boolean evaluation of the rule library on the grounded facts. Returns the fired rules as candidates."""
    kinds = rf["facts"]
    F = set(case_present) | set(resp_actions) | set(resp_final_dx)
    human = lambda f: f.replace("_ordered", "").replace("_prescribed", "").replace("_", " ")  # noqa: E731
    out = []
    for r in rf["rules"]:
        if not r["evaluable"]:
            continue
        if all(p in F for p in r["premise"]) and not any(a in F for a in r["absent"]):
            if r["category"] == "omission" or r["category"] == "anchoring":
                text = "Add " + " or ".join(human(a) for a in r["absent"]) if r["absent"] else "Review the plan"
            elif r["category"] == "undertriage":
                text = "Escalate triage to an emergency level"
            else:
                text = "Review " + " with ".join(human(p) for p in r["premise"])
            out.append({"source": "rule", "rule": r["name"], "category": r["category"], "severity": r["severity"],
                        "action": text, "reason": r["description"][:200],
                        "confidence": min(1.0, max(0.1, (r["severity"] or 5) / 10.0)), "kinds": [kinds.get(p) for p in r["premise"]]})
    return out


@app.function(image=img, secrets=SECRETS, volumes={"/results": results_vol}, timeout=86400)
def match(cohort: str = "c1c", split: str = "all", workers: int = 16, max_rule_candidates: int = 15) -> dict:
    import json
    base, cases, sp, rf = _io(cohort)
    ids = _ids(cases, sp, split)
    casefacts = {}
    for line in open(base / f"v2_casefacts_{cohort}.jsonl"):
        r = json.loads(line)
        if r.get("ok"):
            casefacts[r["case_id"]] = r["present"]
    drafts_by = {}
    for d in MODELS:
        for line in open(base / f"v2_drafts_{cohort}_{d}.jsonl"):
            r = json.loads(line)
            if r.get("ok"):
                drafts_by[(r["case_id"], d)] = r["actions"]
    out = {}
    for mk in MODELS:
        guard = _guard(base, cohort, mk)
        respf = {}
        for line in open(base / f"v2_respfacts_{cohort}_{mk}.jsonl"):
            r = json.loads(line)
            if r.get("ok"):
                respf[r["case_id"]] = r
        path = base / f"v2_match_{cohort}_{mk}.jsonl"
        done = _done(path, "case_id")
        others = [d for d in MODELS if d != mk]

        def one(cid, mk=mk, guard=guard, respf=respf, others=others):
            cands = []
            for d in others:
                for a in drafts_by.get((cid, d), []):
                    cands.append({"source": d, **a})
            rc = rule_candidates(rf, casefacts.get(cid, []), respf[cid]["actions"], respf[cid]["final_diagnoses"])
            rc.sort(key=lambda x: -(x["severity"] or 0))
            cands += rc[:max_rule_candidates]
            if not cands:
                return {"case_id": cid, "model_key": mk, "ok": True, "candidates": [], "groups": [], "n_rules_fired": 0}
            listing = "\n".join(f"{i + 1}. {c['action']} ({c['reason']})" for i, c in enumerate(cands))
            raw = _gpt("gpt-5.5", MATCH_PROMPT.format(case=cases[cid]["case_text"], response=guard[cid], candidates=listing))
            js = _json(raw) or {}
            groups = []
            for g in js.get("groups", []):
                try:
                    mem = [int(x) for x in g.get("members", []) if 1 <= int(x) <= len(cands)]
                except Exception:
                    continue
                if mem:
                    groups.append({"members": mem, "action": str(g.get("action", ""))[:300],
                                   "in_response": str(g.get("in_response", "")).lower()})
            covered = {m for g in groups for m in g["members"]}
            return {"case_id": cid, "model_key": mk, "ok": bool(groups) and len(covered) >= 0.8 * len(cands),
                    "candidates": cands, "groups": groups, "n_rules_fired": len(rc), "raw": raw[:3000]}
        out[mk] = _pool(one, [c for c in ids if c in guard and c in respf and c not in done], workers, path)
    print("[v2 match]", out)
    return out


@app.function(image=img, secrets=SECRETS, volumes={"/results": results_vol}, timeout=86400)
def label(cohort: str = "c1c", split: str = "development", workers: int = 16) -> dict:
    """GPT-6 Astra maps each candidate group to a checklist item number, or 0 (Cohort 1c only)."""
    import json
    base, cases, sp, _ = _io(cohort)
    checklists = json.load(open(base / "cohort1c_cases_checklists.json"))
    ids = set(_ids(cases, sp, split))
    out = {}
    for mk in MODELS:
        groups = {}
        for line in open(base / f"v2_match_{cohort}_{mk}.jsonl"):
            r = json.loads(line)
            if r.get("ok") and r["case_id"] in ids:
                groups[r["case_id"]] = r["groups"]
        path = base / f"v2_label_{cohort}_{mk}.jsonl"
        done = _done(path, "case_id")

        def one(cid, mk=mk, groups=groups):
            gs = groups[cid]
            if not gs:
                return {"case_id": cid, "model_key": mk, "ok": True, "labels": []}
            items = "\n".join(f"{i + 1}. {it['item']}" for i, it in enumerate(checklists[cid]["checklist"]))
            acts = "\n".join(f"{i + 1}. {g['action']}" for i, g in enumerate(gs))
            raw = _gpt("gpt-6-astra", LABEL_PROMPT.format(case=cases[cid]["case_text"], items=items, actions=acts),
                       effort="low", flex=True)
            js = _json(raw) or {}
            lab = {}
            for x in js.get("labels", []):
                try:
                    lab[int(x["n"])] = int(x.get("item", 0))
                except Exception:
                    continue
            labels = [lab.get(i + 1, 0) for i in range(len(gs))]
            return {"case_id": cid, "model_key": mk, "ok": len(lab) >= 0.8 * len(gs), "labels": labels, "raw": raw[:2000]}
        out[mk] = _pool(one, [c for c in groups if c not in done], workers, path)
    print("[v2 label]", out)
    return out


@app.local_entrypoint()
def run(stage: str = "facts", cohort: str = "c1c", split: str = "all", workers: int = 16):
    fn = {"facts": facts, "drafts": drafts, "match": match, "label": label}[stage]
    fc = fn.spawn(cohort=cohort, split=split, workers=workers)
    print(f"[aih_r1_v2] spawned {stage} {cohort}/{split}: {fc.object_id}")
