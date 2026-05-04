"""
Strip ANCHOR safety/operational annotations from a C5_anchor_full response,
preserving the original LLM clinical recommendation and ANCHOR-derived
clinical content (DDx augmentation, anchoring alternatives).

Per SAP §8.4.4 (locked) and HealthBench Pro / AMIE Nature 2025 SOTA convention,
the LLM judge ensemble grades the clinical content of the recommendation,
not the structural safety/operational annotations that ANCHOR appends.

Patterns STRIPPED (safety/operational metadata; not clinical content):
  - "TRIAGE OVERRIDE (ANCHOR LNN..." or "TRIAGE OVERRIDE (ANCHOR Specialists..." line
  - "--- ANCHOR LNN Verification ---" through next blank line or section break
  - "--- ANCHOR: Missing Clinical Items ---" through next section break
  - "--- ANCHOR: Safety Warnings ---" through next section break
  - "--- MORNING REPORT EXPERT CONSULTANT (multi-agent audit) ---" through end OR next "[ANCHOR DDx" marker
  - "[MR Drug Interaction Agent]" / "[MR Lab Interpretation Agent]" / etc.
  - "[MR Disagreement Flags ...]"
  - "[MR Schema Advisory ...]"
  - "[MR Concept Decomposition (Adebayo et al. 2025)]" through end
  - "[ANCHOR Einstein Arena: ensemble flag ...]"
  - "[ANCHOR ICD-10 (constrained decoder)]: ..."
  - "[ANCHOR RAG provenance]: ..."

Patterns PRESERVED (clinical content):
  - The original LLM clinical response (everything before the first ANCHOR
    annotation block)
  - "--- ANCHOR DDx augmentation ---" section (top-5 most-likely + don't-miss
    dual-list per SAP §8.18) — this IS clinical content the judge should see
  - "--- ANCHOR: Alternative Diagnoses to Consider ---" section (anchoring
    alternatives) — these are clinically-relevant DDx expansions

Usage:
    from strip_anchor_annotations import strip_anchor_annotations
    stripped = strip_anchor_annotations(c5_response_text)
    # Pass `stripped` to the judge instead of the full response
"""

from __future__ import annotations
import re

# Section markers that begin annotation blocks to STRIP entirely
STRIP_SECTION_MARKERS = (
    "--- ANCHOR LNN Verification ---",
    "--- ANCHOR: Missing Clinical Items ---",
    "--- ANCHOR: Safety Warnings ---",
    "--- MORNING REPORT EXPERT CONSULTANT (multi-agent audit) ---",
)

# Section markers that begin annotation blocks to KEEP
KEEP_SECTION_MARKERS = (
    "--- ANCHOR DDx augmentation ---",
    "--- ANCHOR: Alternative Diagnoses to Consider ---",
)

# Inline annotation prefixes (single-block lines or short paragraphs)
INLINE_STRIP_PREFIXES = (
    "[MR Drug Interaction Agent]",
    "[MR Lab Interpretation Agent]",
    "[MR Guideline Compliance Agent]",
    "[MR Citation Verification Agent]",
    "[MR Safety Net Agent]",
    "[MR Disagreement Flags",
    "[MR Schema Advisory]",
    "[MR Concept Decomposition",
    "[ANCHOR Einstein Arena:",
    "[ANCHOR ICD-10",
    "[ANCHOR RAG provenance]",
)

# TRIAGE OVERRIDE line — appears anywhere; strip wherever it occurs
TRIAGE_OVERRIDE_PATTERN = re.compile(
    r"^TRIAGE OVERRIDE \(ANCHOR (LNN|Specialists)[^\n]*\n?",
    re.M,
)


def _split_into_blocks(text: str) -> list[tuple[str, str]]:
    """Split a response into (block_type, content) tuples.

    block_type is one of: "clinical", "strip_section", "keep_section",
    "inline_strip", "triage_override".
    """
    if not text:
        return []
    # Mark all section boundaries
    all_markers = list(STRIP_SECTION_MARKERS) + list(KEEP_SECTION_MARKERS)
    # Sort markers by length desc so longer markers match first
    all_markers.sort(key=len, reverse=True)

    blocks: list[tuple[str, str]] = []
    # Strip TRIAGE OVERRIDE lines (anywhere they appear) before splitting
    text = TRIAGE_OVERRIDE_PATTERN.sub("", text)
    remaining = text

    # Iteratively split on section markers
    while remaining:
        # Find the earliest occurrence of any marker
        earliest = None
        earliest_marker = None
        earliest_kind = None
        for marker in STRIP_SECTION_MARKERS:
            idx = remaining.find("\n" + marker)
            if idx < 0:
                idx = remaining.find(marker) if remaining.startswith(marker) else -1
            if idx >= 0 and (earliest is None or idx < earliest):
                earliest = idx
                earliest_marker = marker
                earliest_kind = "strip_section"
        for marker in KEEP_SECTION_MARKERS:
            idx = remaining.find("\n" + marker)
            if idx < 0:
                idx = remaining.find(marker) if remaining.startswith(marker) else -1
            if idx >= 0 and (earliest is None or idx < earliest):
                earliest = idx
                earliest_marker = marker
                earliest_kind = "keep_section"

        if earliest is None:
            # No more section markers; remaining is clinical content
            blocks.append(("clinical", remaining))
            break

        # Adjust idx to absolute (handles "\n" prefix)
        if remaining[earliest:earliest + 1] == "\n":
            split_idx = earliest
            content_idx = earliest + 1
        else:
            split_idx = earliest
            content_idx = earliest

        # Pre-section content is clinical
        pre = remaining[:split_idx]
        if pre.strip():
            blocks.append(("clinical", pre))

        # Find end of this section: next section marker or end
        section_text = remaining[content_idx:]
        next_section_idx = None
        for marker in all_markers:
            for prefix in ("\n" + marker, marker):
                pos = section_text.find(prefix, len(earliest_marker))
                if pos >= 0 and (next_section_idx is None or pos < next_section_idx):
                    next_section_idx = pos
                    break
        if next_section_idx is not None:
            blocks.append((earliest_kind, section_text[:next_section_idx]))
            remaining = section_text[next_section_idx:]
            if remaining.startswith("\n"):
                remaining = remaining[1:]
        else:
            blocks.append((earliest_kind, section_text))
            remaining = ""

    return blocks


def _strip_inline_annotations(text: str) -> str:
    """Remove inline [MR ...] / [ANCHOR ...] one-off annotation paragraphs."""
    if not text:
        return text
    lines_out: list[str] = []
    skip_until_blank = False
    for line in text.splitlines():
        stripped_line = line.lstrip()
        if any(stripped_line.startswith(p) for p in INLINE_STRIP_PREFIXES):
            skip_until_blank = True
            continue
        if skip_until_blank:
            if stripped_line == "" or stripped_line.startswith("---"):
                skip_until_blank = False
                if stripped_line.startswith("---"):
                    lines_out.append(line)
                continue
            else:
                continue
        lines_out.append(line)
    return "\n".join(lines_out)


def strip_anchor_annotations(response: str) -> str:
    """Return the judge-visible clinical content of a C5_anchor_full response.

    The output preserves:
      - The original LLM clinical recommendation
      - Any ANCHOR DDx augmentation block (top-5 + don't-miss dual list)
      - Any ANCHOR Alternative Diagnoses to Consider block (anchoring alternatives)

    The output strips:
      - TRIAGE OVERRIDE annotation line(s)
      - ANCHOR LNN Verification block
      - ANCHOR Missing Clinical Items block
      - ANCHOR Safety Warnings block
      - MORNING REPORT EXPERT CONSULTANT multi-agent audit block + its inline
        [MR ...] sub-annotations (Drug Interaction, Lab, Guideline, Citation,
        Safety Net, Disagreement Flags, Schema Advisory, Concept Decomposition)
      - ANCHOR Einstein Arena flag
      - ANCHOR ICD-10 constrained decoder annotation
      - ANCHOR RAG provenance annotation
    """
    if not response:
        return response

    blocks = _split_into_blocks(response)
    kept_parts = []
    for kind, content in blocks:
        if kind in ("clinical", "keep_section"):
            kept_parts.append(content)
        # strip_section / triage_override / inline_strip are dropped
    out = "\n".join(p for p in kept_parts if p)
    # Apply inline-prefix stripping on what remains (e.g., a [ANCHOR ICD-10]
    # appended to a clinical line)
    out = _strip_inline_annotations(out)
    # Collapse multiple blank lines
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """1. **TRIAGE LEVEL: Emergent**

This patient presents with severe respiratory distress.

2. **TOP-3 DIFFERENTIAL DIAGNOSES**
1. Pulmonary embolism — sudden onset, hypoxia
2. Acute MI — chest pain, diaphoresis
3. Pneumothorax — sudden dyspnea

TRIAGE OVERRIDE (ANCHOR LNN): URGENT — clinical findings inconsistent

--- ANCHOR: Missing Clinical Items ---
[ANCHOR -- ECG without troponin]: add troponin ordered
[ANCHOR -- Suicidal ideation safety packet]: add toxicology screen

--- MORNING REPORT EXPERT CONSULTANT (multi-agent audit) ---

[MR Drug Interaction Agent]
No drug interactions flagged.

[MR Safety Net Agent]
Consider PE workup with CTPA.

[MR Concept Decomposition (Adebayo et al. 2025)]
  known: 3 agent-traceable findings
  discovered: 1 emergent (safety-net)
  epsilon: residual attribution gap disclosed

--- ANCHOR DDx augmentation ---
**Most likely (top 5):**
1. Pulmonary embolism — Wells-criteria features
2. Acute MI — troponin pending
3. Pneumothorax — sudden dyspnea, decreased breath sounds
4. Pneumonia — fever, leukocytosis
5. Aortic dissection — tearing pain, BP differential

**Don't miss (top 3):**
1. Tension pneumothorax — needs immediate decompression
2. Massive PE — hemodynamic compromise
3. STEMI — door-to-balloon < 90 min

[ANCHOR Einstein Arena: ensemble flag (>=2/3 voters agreed)]
"""
    print("=== STRIPPED ===")
    print(strip_anchor_annotations(sample))
    print("\n=== END ===")
