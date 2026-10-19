"""Diagnostics of the full-findings rule certificate (Note S12).

Counts the rules that can be evaluated on the fact vocabulary, the fact names listed twice, and the rules that fire on
an empty fact vector (no finding, no action) with the deployed trained weights and with untrained weights. A rule
that fires with no facts fires whatever the case and the response contain. Writes
results_aih_r1/certificate_diagnostics.json.

Run: python aih_r1_certificate_diagnostics.py [path/to/lnn_verifier.pt]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model.lnn_verifier as L  # noqa: E402

DATA = Path(__file__).resolve().parent / "results_aih_r1"
ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "lnn_verifier.pt"


CK = torch.load(ckpt_path, map_location="cpu")


def fired_on_empty(load_ckpt: bool, seed: int) -> int:
    """Rules that fire on an empty fact vector. Parameters absent from the checkpoint keep their random
    initialization, so the count depends on the seed of the process that builds the verifier."""
    torch.manual_seed(seed)
    v = L.ClinicalLNNVerifier()
    if load_ckpt:
        sd = v.state_dict()
        v.load_state_dict({k: t for k, t in CK.items() if k in sd and t.shape == sd[k].shape}, strict=False)
    v.eval()
    with torch.no_grad():
        return int(v._verify_from_facts(torch.zeros(L.N_FACTS))["n_violations"])


def summary(xs):
    xs = sorted(xs)
    return {"min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1], "n_seeds": len(xs)}


v = L.ClinicalLNNVerifier()
rules = v.rule_eval.rules
dups = sorted(f for f, c in Counter(L.CLINICAL_FACTS).items() if c > 1)
out = {"n_rules": len(rules), "n_facts_listed": len(L.CLINICAL_FACTS), "n_fact_names": len(L.FACT_INDEX),
       "duplicate_fact_names": dups, "n_duplicate_fact_names": len(dups),
       "n_rules_evaluable": sum(1 for r in rules if r.premise_facts and all(f in L.FACT_INDEX for f in r.premise_facts)),
       "n_rules_premise_outside_vocabulary": sum(1 for r in rules if not r.premise_facts
                                                 or not all(f in L.FACT_INDEX for f in r.premise_facts)),
       "n_parameter_tensors": len(v.state_dict()),
       "n_parameter_tensors_in_checkpoint": sum(1 for k, t in v.state_dict().items() if k in CK and CK[k].shape == t.shape)}
SEEDS = range(20)
trained = [fired_on_empty(True, s_) for s_ in SEEDS]
untrained = [fired_on_empty(False, s_) for s_ in SEEDS]
out["trained_rules_fired_empty"] = summary(trained)
out["untrained_rules_fired_empty"] = summary(untrained)
(DATA / "certificate_diagnostics.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
