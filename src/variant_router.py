"""Graded VUS routing: evidence tier → therapy confidence gate."""
from __future__ import annotations

from typing import Any, Literal

from src.evidence import score_evidence_tier

Classification = Literal["known_driver", "likely_driver", "vus", "unknown"]
EvidenceTier = Literal["strong", "weak", "none"]


def route_variant(
    target: dict,
    structure: dict,
    evidence: list[dict],
) -> dict[str, Any]:
    """Return routing metadata merged into target for downstream agents."""
    tier = score_evidence_tier(evidence)
    gene = target.get("gene", "")
    pathway = target.get("pathway", "inhibitor_rag")
    cls = target.get("class", "")

    if cls == "UNKNOWN" or not target.get("civic_profile"):
        tier = score_evidence_tier(evidence) if evidence else "none"
        if tier != "strong":
            return {
                "evidence_tier": tier if tier != "strong" else "none",
                "classification": "vus",
                "vus_branch": True,
                "allow_confident_therapy": False,
                "pathway": pathway,
                "gene": gene,
                "mutation": target.get("mutation", ""),
            }

    if tier == "strong" and cls in ("ONCOGENE_GOF", "TUMOR_SUPPRESSOR_LOF"):
        classification: Classification = "known_driver"
        allow_confident_therapy = True
        vus_branch = False
    elif tier == "weak":
        classification = "likely_driver" if evidence else "vus"
        allow_confident_therapy = bool(evidence) and pathway != "structural_rescue"
        vus_branch = not allow_confident_therapy
    else:
        classification = "unknown"
        allow_confident_therapy = False
        vus_branch = True

    if pathway == "structural_rescue" and tier != "strong":
        allow_confident_therapy = False

    return {
        "evidence_tier": tier,
        "classification": classification,
        "vus_branch": vus_branch,
        "allow_confident_therapy": allow_confident_therapy,
        "pathway": pathway,
        "gene": gene,
        "mutation": target.get("mutation", ""),
    }
