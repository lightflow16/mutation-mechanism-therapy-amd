"""Build LoRA SFT dataset from CIViC/ClinVar case JSONs (training data ONLY - no OncoKB)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "cases"
OUT = ROOT / "data" / "lora_train.jsonl"

PARAPHRASES = [
    "Provide mechanism and therapy reasoning as JSON.",
    "Return structured JSON with mechanism, therapy sensitivity/resistance, and confidence.",
    "As an oncology copilot, output JSON reasoning with citations.",
]

# CIViC/ClinVar-style public training rows (no OncoKB). ~50 variants x 3 paraphrases = 150.
CIVIC_TRAINING_ROWS = [
    {"gene": "BRAF", "change": "V600E", "mechanism": "V600E constitutively activates BRAF kinase.", "sensitivity": ["Vemurafenib", "Dabrafenib"], "resistance": [], "context": "BRAF-mutant melanoma"},
    {"gene": "KRAS", "change": "G12C", "mechanism": "G12C locks KRAS in active GTP-bound state.", "sensitivity": ["Sotorasib", "Adagrasib"], "resistance": [], "context": "KRAS G12C NSCLC"},
    {"gene": "KRAS", "change": "G12D", "mechanism": "G12D impairs GTP hydrolysis in KRAS.", "sensitivity": [], "resistance": ["EGFR inhibitors"], "context": "KRAS-mutant CRC"},
    {"gene": "ALK", "change": "F1174L", "mechanism": "F1174L activates ALK kinase signaling.", "sensitivity": ["Crizotinib", "Alectinib"], "resistance": [], "context": "ALK-rearranged NSCLC"},
    {"gene": "MET", "change": "exon14 skipping", "mechanism": "Exon 14 skipping removes CBL binding and activates MET.", "sensitivity": ["Capmatinib", "Tepotinib"], "resistance": [], "context": "MET ex14 NSCLC"},
    {"gene": "RET", "change": "fusion", "mechanism": "RET fusion drives constitutive kinase activity.", "sensitivity": ["Selpercatinib", "Pralsetinib"], "resistance": [], "context": "RET fusion NSCLC"},
    {"gene": "ERBB2", "change": "amplification", "mechanism": "HER2 amplification hyperactivates MAPK/PI3K.", "sensitivity": ["Trastuzumab", "T-DXd"], "resistance": [], "context": "HER2+ breast"},
    {"gene": "BRCA1", "change": "truncating", "mechanism": "Loss of BRCA1 impairs homologous recombination.", "sensitivity": ["Olaparib", "Rucaparib"], "resistance": [], "context": "BRCA-mutant ovarian"},
    {"gene": "BRCA2", "change": "truncating", "mechanism": "BRCA2 loss causes HR deficiency.", "sensitivity": ["Olaparib"], "resistance": [], "context": "BRCA-mutant prostate"},
    {"gene": "IDH1", "change": "R132H", "mechanism": "R132H neomorphically produces 2-HG.", "sensitivity": ["Ivosidenib"], "resistance": [], "context": "IDH1-mutant AML"},
    {"gene": "IDH2", "change": "R140Q", "mechanism": "R140Q produces oncometabolite 2-HG.", "sensitivity": ["Enasidenib"], "resistance": [], "context": "IDH2-mutant AML"},
    {"gene": "FLT3", "change": "ITD", "mechanism": "FLT3-ITD drives STAT5 and PI3K signaling.", "sensitivity": ["Midostaurin", "Gilteritinib"], "resistance": [], "context": "FLT3-ITD AML"},
    {"gene": "KIT", "change": "D816V", "mechanism": "D816V activates KIT in absence of ligand.", "sensitivity": ["Avapritinib"], "resistance": ["Imatinib"], "context": "SM-AHNMD"},
    {"gene": "PDGFRA", "change": "D842V", "mechanism": "D842V confers imatinib-resistant PDGFRA activation.", "sensitivity": ["Avapritinib"], "resistance": ["Imatinib"], "context": "GIST"},
    {"gene": "NTRK1", "change": "fusion", "mechanism": "NTRK fusion drives TRK signaling.", "sensitivity": ["Larotrectinib", "Entrectinib"], "resistance": [], "context": "NTRK fusion solid tumors"},
    {"gene": "FGFR2", "change": "fusion", "mechanism": "FGFR2 fusion activates RTK signaling.", "sensitivity": ["Pemigatinib", "Futibatinib"], "resistance": [], "context": "FGFR2 cholangiocarcinoma"},
    {"gene": "FGFR3", "change": "S249C", "mechanism": "S249C stabilizes disulfide-linked FGFR3 dimers.", "sensitivity": ["Erdafitinib"], "resistance": [], "context": "FGFR3 bladder"},
    {"gene": "ESR1", "change": "Y537S", "mechanism": "Y537S ligand-independent ER activation.", "sensitivity": ["Elacestrant"], "resistance": ["Aromatase inhibitors"], "context": "ESR1-mutant breast"},
    {"gene": "AR", "change": "L702H", "mechanism": "L702H confers antiandrogen resistance.", "sensitivity": [], "resistance": ["Enzalutamide"], "context": "CRPC"},
    {"gene": "JAK2", "change": "V617F", "mechanism": "V617F constitutively activates JAK-STAT.", "sensitivity": ["Ruxolitinib"], "resistance": [], "context": "MPN"},
    {"gene": "CALR", "change": "type1", "mechanism": "CALR mutation activates MPL signaling.", "sensitivity": ["Ruxolitinib"], "resistance": [], "context": "ET/PV"},
    {"gene": "MPL", "change": "W515L", "mechanism": "W515L activates thrombopoietin receptor.", "sensitivity": ["Ruxolitinib"], "resistance": [], "context": "MPN"},
    {"gene": "NF1", "change": "LOF", "mechanism": "NF1 loss hyperactivates RAS/MAPK.", "sensitivity": ["MEK inhibitors"], "resistance": [], "context": "NF1-mutant melanoma"},
    {"gene": "PTEN", "change": "LOF", "mechanism": "PTEN loss increases PI3K/AKT signaling.", "sensitivity": [], "resistance": ["PI3K inhibitors"], "context": "PTEN-null tumors"},
    {"gene": "RB1", "change": "LOF", "mechanism": "RB1 loss releases E2F-driven proliferation.", "sensitivity": [], "resistance": ["CDK4/6 inhibitors"], "context": "SCLC"},
    {"gene": "CDKN2A", "change": "LOF", "mechanism": "p16 loss removes CDK4/6 inhibition.", "sensitivity": ["Palbociclib"], "resistance": [], "context": "CDKN2A-null melanoma"},
    {"gene": "ATM", "change": "LOF", "mechanism": "ATM loss impairs DNA damage response.", "sensitivity": ["Olaparib"], "resistance": [], "context": "ATM-mutant prostate"},
    {"gene": "PALB2", "change": "LOF", "mechanism": "PALB2 loss causes HR deficiency.", "sensitivity": ["Olaparib"], "resistance": [], "context": "PALB2 breast"},
    {"gene": "MLH1", "change": "LOF", "mechanism": "MMR deficiency increases neoantigen load.", "sensitivity": ["Pembrolizumab"], "resistance": [], "context": "MSI-H CRC"},
    {"gene": "MSH2", "change": "LOF", "mechanism": "MMR loss enables hypermutation.", "sensitivity": ["Pembrolizumab"], "resistance": [], "context": "MSI-H endometrial"},
    {"gene": "ROS1", "change": "fusion", "mechanism": "ROS1 fusion drives oncogenic signaling.", "sensitivity": ["Crizotinib", "Entrectinib"], "resistance": [], "context": "ROS1 NSCLC"},
    {"gene": "NTRK3", "change": "fusion", "mechanism": "NTRK3 fusion activates TRK pathway.", "sensitivity": ["Larotrectinib"], "resistance": [], "context": "NTRK fusion"},
    {"gene": "EGFR", "change": "T790M", "mechanism": "T790M gatekeeper mutation confers TKI resistance.", "sensitivity": ["Osimertinib"], "resistance": ["Gefitinib", "Erlotinib"], "context": "EGFR-mutant NSCLC"},
    {"gene": "EGFR", "change": "C797S", "mechanism": "C797S confers osimertinib resistance.", "sensitivity": [], "resistance": ["Osimertinib"], "context": "EGFR-mutant NSCLC"},
    {"gene": "PIK3CA", "change": "H1047R", "mechanism": "H1047R disrupts inhibitory contacts in PI3K alpha.", "sensitivity": ["Alpelisib"], "resistance": [], "context": "PIK3CA-mutant breast"},
    {"gene": "PIK3CA", "change": "E542K", "mechanism": "E542K activates PI3K alpha kinase domain.", "sensitivity": ["Alpelisib"], "resistance": [], "context": "PIK3CA-mutant breast"},
    {"gene": "AKT1", "change": "E17K", "mechanism": "E17K recruits PI3K to membrane constitutively.", "sensitivity": ["Capivasertib"], "resistance": [], "context": "AKT1-mutant breast"},
    {"gene": "MTOR", "change": "E2419K", "mechanism": "MTOR activating mutation increases mTORC1.", "sensitivity": ["Everolimus"], "resistance": [], "context": "mTOR-mutant tumors"},
    {"gene": "SMO", "change": "W535L", "mechanism": "SMO mutation activates hedgehog pathway.", "sensitivity": ["Vismodegib"], "resistance": [], "context": "BCC"},
    {"gene": "CTNNB1", "change": "S45F", "mechanism": "S45F stabilizes beta-catenin.", "sensitivity": [], "resistance": [], "context": "HCC WNT activation"},
    {"gene": "APC", "change": "truncating", "mechanism": "APC loss activates WNT signaling.", "sensitivity": [], "resistance": [], "context": "CRC"},
    {"gene": "VHL", "change": "LOF", "mechanism": "VHL loss stabilizes HIF and angiogenesis.", "sensitivity": ["Sunitinib", "Pazopanib"], "resistance": [], "context": "ccRCC"},
    {"gene": "TSC1", "change": "LOF", "mechanism": "TSC1 loss hyperactivates mTOR.", "sensitivity": ["Everolimus"], "resistance": [], "context": "LAM/TSC"},
    {"gene": "TSC2", "change": "LOF", "mechanism": "TSC2 loss increases mTORC1 signaling.", "sensitivity": ["Everolimus"], "resistance": [], "context": "TSC"},
    {"gene": "STK11", "change": "LOF", "mechanism": "LKB1 loss rewires metabolism and immune evasion.", "sensitivity": [], "resistance": ["Immunotherapy"], "context": "KRAS/STK11 NSCLC"},
    {"gene": "KEAP1", "change": "LOF", "mechanism": "KEAP1 loss increases NRF2 antioxidant program.", "sensitivity": [], "resistance": ["Chemotherapy"], "context": "NSCLC"},
    {"gene": "MDM2", "change": "amplification", "mechanism": "MDM2 amplification degrades p53.", "sensitivity": ["Idasanutlin"], "resistance": [], "context": "MDM2-amp sarcoma"},
    {"gene": "CDK4", "change": "amplification", "mechanism": "CDK4 amplification drives cell cycle.", "sensitivity": ["Palbociclib"], "resistance": [], "context": "CDK4-amp melanoma"},
    {"gene": "CDK6", "change": "amplification", "mechanism": "CDK6 amplification supports G1-S transition.", "sensitivity": ["Abemaciclib"], "resistance": [], "context": "Breast"},
    {"gene": "EML4", "change": "ALK fusion", "mechanism": "EML4-ALK fusion activates ALK kinase.", "sensitivity": ["Alectinib", "Brigatinib"], "resistance": ["Crizotinib"], "context": "ALK+ NSCLC"},
]


def case_to_examples(case: dict) -> list[dict]:
    user_base = (
        f"Variant: {case['variant']['gene']} {case['variant']['protein_change']}\n"
        f"Structure: {json.dumps(case['structure'])}\n"
        f"Evidence: {json.dumps(case['evidence'])}\n"
    )
    assistant = json.dumps(case["target_reasoning"])
    return [
        {"messages": [
            {"role": "user", "content": user_base + suffix},
            {"role": "assistant", "content": assistant},
        ]}
        for suffix in PARAPHRASES
    ]


def civic_row_to_case(row: dict) -> dict:
    return {
        "variant": {"gene": row["gene"], "protein_change": row["change"]},
        "structure": {
            "residue": row["change"][:4],
            "pLDDT_at_residue": 60.0,
            "region": "kinase or functional domain (CIViC training row)",
        },
        "evidence": [{
            "source": "CIViC",
            "therapies": ",".join(row.get("sensitivity", []) + row.get("resistance", [])),
            "direction": "Sensitivity/Response",
            "level": "B",
        }],
        "target_reasoning": {
            "mechanism": row["mechanism"],
            "therapy": {
                "sensitivity": row.get("sensitivity", []),
                "resistance": row.get("resistance", []),
                "context": row.get("context", "cancer"),
            },
            "confidence": "0.70",
        },
    }


def build_dataset_from_traces(
    traces_dir: Path | None = None,
    *,
    metrics_dir_path: Path | None = None,
    out_path: Path | None = None,
) -> tuple[Path, int]:
    """Distill teacher (blackboard) traces into LoRA SFT rows. Returns (out_path, n_rows)."""
    from src.pipeline import extract_target_reasoning

    tdir = traces_dir or ROOT / "data" / "traces"
    md = metrics_dir_path or Path(__import__("os").environ.get("METRICS_DIR", ""))
    dest = out_path or OUT
    dest.parent.mkdir(parents=True, exist_ok=True)

    sources: list[Path] = list(tdir.glob("*_blackboard.json"))
    if md and Path(md).is_dir():
        sources.extend(Path(md).glob("trace_*_blackboard.json"))

    rows: list[dict] = []
    seen: set[str] = set()
    for p in sources:
        key = p.name
        if key in seen:
            continue
        seen.add(key)
        trace = json.loads(p.read_text())
        target = trace.get("target", {})
        structure = trace.get("structure", {})
        evidence = trace.get("evidence", [])
        tr = extract_target_reasoning(trace)
        if not tr or tr.get("raw"):
            continue
        label = tr if tr.get("therapy") else tr.get("target_reasoning", tr)
        case = {
            "variant": {"gene": target.get("gene"), "protein_change": target.get("mutation")},
            "structure": structure,
            "evidence": evidence,
            "target_reasoning": label,
        }
        rows.extend(case_to_examples(case))

    with open(dest, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return dest, len(rows)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-traces", action="store_true", help="Distill from blackboard trace JSON")
    ap.add_argument("--traces-only", action="store_true", help="Write only teacher-trace rows")
    ap.add_argument("--traces-dir", default=None)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0

    if args.from_traces or args.traces_only:
        _, n = build_dataset_from_traces(
            Path(args.traces_dir) if args.traces_dir else None,
        )
        print(f"[distill] {n} teacher-trace rows -> {OUT}")
        if args.traces_only:
            return

    mode = "a" if (args.from_traces and n) else "w"
    with open(OUT, mode) as f:
        for p in sorted(CASES.glob("*.json")):
            case = json.loads(p.read_text())
            for ex in case_to_examples(case):
                f.write(json.dumps(ex) + "\n")
                n += 1
        for row in CIVIC_TRAINING_ROWS:
            case = civic_row_to_case(row)
            for ex in case_to_examples(case):
                f.write(json.dumps(ex) + "\n")
                n += 1
    print(f"Wrote {n} examples -> {OUT}")


if __name__ == "__main__":
    main()
