"""Quote, verbatim, the passage of each Table 2 text that addresses its checklist item (GPT-6 Astra).

Reads results_aih_r1/table2_quote_requests.json (written by the Table 2 builder) and writes
results_aih_r1/table2_quotes.json. The builder keeps a quote only if it appears verbatim in the text.
Run: modal run aih_r1_table2_quotes.py
"""
import json
from pathlib import Path

import modal

app = modal.App("aih-r1-table2-quotes")
img = modal.Image.debian_slim(python_version="3.11").pip_install("openai>=1.50.0")
DATA = Path(__file__).resolve().parent / "results_aih_r1"
PROMPT = ("From the TEXT below, quote verbatim the shortest passage, at most 40 words, that addresses this required "
          "clinical action:\n\nACTION: {item}\n\nCopy the words exactly as they appear in the text, without adding or "
          "changing any word. If no passage addresses the action, reply exactly NONE.\n\nTEXT:\n{text}")


@app.function(image=img, secrets=[modal.Secret.from_name("openai")], timeout=900)
def extract(requests: dict) -> dict:
    import openai
    client = openai.OpenAI()
    out = {}
    for k, r in requests.items():
        resp = client.responses.create(model="gpt-6-astra", input=PROMPT.format(item=r["item"], text=r["text"][:16000]),
                                       max_output_tokens=2000, reasoning={"effort": "low"})
        q = (resp.output_text or "").strip().strip('"“”')
        out[k] = None if q.upper().startswith("NONE") else q
    return out


@app.local_entrypoint()
def main():
    req = json.load(open(DATA / "table2_quote_requests.json"))
    quotes = extract.remote(req)
    (DATA / "table2_quotes.json").write_text(json.dumps(quotes, indent=1, ensure_ascii=False))
    for k, v in quotes.items():
        print(k, "->", (v or "NONE")[:200])
