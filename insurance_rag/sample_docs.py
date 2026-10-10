"""Generate two FICTIONAL plan PDFs plus an evaluation set from the same fact table, so questions and gold answers
can never drift from the documents. All names and numbers are invented; this is test data, not real coverage.

  python -m insurance_rag.sample_docs --out data/sample
"""
from __future__ import annotations

import argparse
import json
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PLANS = {
    "Cedar Silver 3000": dict(
        file="cedar_silver_3000.pdf", ded="$3,000", fam_ded="$6,000", oop="$8,500", fam_oop="$17,000",
        primary="$30", specialist="$60", urgent="$75", er="$350", generic="$10", brand="$45", inpatient="30%",
        pa="5", appeal="180",
        acupuncture="Acupuncture is not covered.", dental="Adult dental care is not covered.",
        acu_fact="not covered", dental_fact="not covered"),
    "Cedar Gold 1000": dict(
        file="cedar_gold_1000.pdf", ded="$1,000", fam_ded="$2,000", oop="$5,000", fam_oop="$10,000",
        primary="$20", specialist="$40", urgent="$50", er="$250", generic="$5", brand="$30", inpatient="20%",
        pa="3", appeal="180",
        acupuncture="Acupuncture is covered up to 12 visits per calendar year.",
        dental="Adult dental care: two preventive cleanings per calendar year are covered at no cost.",
        acu_fact="12 visits", dental_fact="two preventive cleanings"),
}

UNANSWERABLE = ["Does the plan cover bariatric surgery?", "What is the copay for chiropractic care?",
                "Is there a separate deductible for vision exams?", "How much is the premium for a family of four?"]


def sections(p: dict, name: str):
    return [
        ("ABOUT THIS DOCUMENT", [
            f"This is a fictional Summary of Benefits and Coverage for the {name} plan, written for software testing only.",
            "It describes what the plan pays for covered services during a calendar year. Your actual plan documents control.",
            "Network providers are doctors and hospitals that have a contract with the plan."]),
        ("DEDUCTIBLES AND OUT-OF-POCKET LIMITS", [
            f"The individual deductible is {p['ded']} per calendar year.",
            f"The family deductible is {p['fam_ded']} per calendar year.",
            f"The individual out-of-pocket maximum is {p['oop']} per calendar year.",
            f"The family out-of-pocket maximum is {p['fam_oop']} per calendar year.",
            "Balance-billed charges and services this plan does not cover do not count toward the out-of-pocket maximum."]),
        ("COMMON MEDICAL EVENTS", [
            f"Primary care visit: {p['primary']} copay per visit, deductible does not apply.",
            f"Specialist visit: {p['specialist']} copay per visit, deductible does not apply.",
            f"Urgent care: {p['urgent']} copay per visit.",
            f"Emergency room: {p['er']} copay per visit after the deductible is met.",
            f"Generic drugs: {p['generic']} copay for a 30-day supply.",
            f"Preferred brand drugs: {p['brand']} copay for a 30-day supply.",
            f"Inpatient hospital stay: {p['inpatient']} coinsurance after the deductible is met.",
            f"Outpatient mental health visit: {p['primary']} copay per visit."]),
        ("PAGEBREAK", []),
        ("PRIOR AUTHORIZATION", [
            "Prior authorization is required for MRI, CT and PET scans, and for planned inpatient admissions.",
            f"Request authorization at least {p['pa']} business days before the service.",
            "Emergency care never requires prior authorization.",
            "If authorization is required and not obtained, the plan may reduce or deny payment for the service."]),
        ("EXCLUSIONS AND LIMITATIONS", [
            "This plan does not cover cosmetic surgery, weight loss programs or experimental treatments.",
            p["acupuncture"], p["dental"],
            "Services received outside the network are covered only for emergencies unless the plan approved them in advance."]),
        ("CLAIMS AND APPEALS", [
            f"You have {p['appeal']} days from the date of a denial notice to file an internal appeal.",
            "The plan decides standard pre-service appeals within 30 days.",
            "You may ask for an external review by an independent organization after the internal appeal is decided."]),
    ]


def write_pdf(path: str, name: str, p: dict) -> None:
    ss = getSampleStyleSheet()
    head = ParagraphStyle("h", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10.5, leading=14)
    doc = SimpleDocTemplate(path, pagesize=letter, title=name)
    story = [Paragraph(f"{name} - Summary of Benefits", ss["Title"]), Spacer(1, 8)]
    for title, lines in sections(p, name):
        if title == "PAGEBREAK":
            story.append(PageBreak()); continue
        story.append(Paragraph(title, head))
        story += [Paragraph(ln, body) for ln in lines] + [Spacer(1, 10)]
    doc.build(story)


def eval_items(plans=PLANS) -> list[dict]:
    items = []
    for name, p in plans.items():
        qs = [("What is the individual deductible?", [p["ded"]]),
              ("What is the family out-of-pocket maximum?", [p["fam_oop"]]),
              ("How much is the copay for a primary care visit?", [p["primary"], "primary care"]),
              ("How much do I pay to see a specialist?", [p["specialist"], "specialist"]),
              ("What is the urgent care copay?", [p["urgent"]]),
              ("How much do I pay for an emergency room visit?", [p["er"]]),
              ("What is the copay for generic drugs?", [p["generic"], "generic"]),
              ("What coinsurance applies to an inpatient hospital stay?", [p["inpatient"]]),
              ("How many business days before a service must I request prior authorization?", [f"{p['pa']} business days"]),
              ("How long do I have to file an appeal after a denial?", [f"{p['appeal']} days"]),
              ("Is acupuncture covered?", [p["acu_fact"]]),
              ("Is adult dental care covered?", [p["dental_fact"]])]
        items += [{"question": q, "plan": name, "facts": f, "answerable": True} for q, f in qs]
        items += [{"question": q, "plan": name, "facts": [], "answerable": False} for q in UNANSWERABLE]
    return items


def build(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, p in PLANS.items():
        paths[name] = os.path.join(out_dir, p["file"])
        write_pdf(paths[name], name, p)
    eval_path = os.path.join(out_dir, "eval.jsonl")
    with open(eval_path, "w") as f:
        for it in eval_items():
            f.write(json.dumps(it) + "\n")
    return {"pdfs": paths, "eval": eval_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/sample")
    print(json.dumps(build(ap.parse_args().out), indent=2))
