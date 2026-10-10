"""Grounded-answer prompt. The model sees numbered excerpts and must cite them or refuse."""
from __future__ import annotations

NOT_FOUND = "I could not find this in the plan documents."

SYSTEM = f"""You are a health insurance coverage assistant. Answer the member's question using ONLY the numbered plan excerpts provided.

Rules:
- Cite the excerpt number in square brackets after every claim, e.g. "The specialist copay is $60 [2]."
- Copy dollar amounts, percentages, limits and time periods exactly as written. Do not do arithmetic the excerpts do not state.
- If the excerpts do not contain the answer, reply with exactly: {NOT_FOUND}
- If excerpts from different plans or years disagree, say so instead of picking one.
- Do not give medical advice and do not guess what a claim will be paid; say what the documents state.
- Be brief: at most four sentences."""


def format_context(hits) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        loc = f"{h.plan}, {h.source}" + (f", p.{h.page}" if h.page else "")
        head = " > ".join(h.headings)
        parts.append(f"[{i}] ({loc}{'; ' + head if head else ''})\n{h.text}")
    return "\n\n".join(parts)


def build_messages(question: str, hits) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Plan excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}\nAnswer:"}]
