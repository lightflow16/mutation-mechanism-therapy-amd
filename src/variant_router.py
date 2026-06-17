"""Graded VUS routing: evidence tier → therapy confidence gate."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

from src.evidence import score_evidence_tier

logger = logging.getLogger("MMTR.VariantRouter")

Classification = Literal["known_driver", "likely_driver", "VUS-high", "VUS-mid", "likely_benign", "vus", "unknown"]
EvidenceTier = Literal["strong", "weak", "none"]


class VariantRouter:
    """
    Implements a graded evidence-inference routing decision tree for clinical variants.
    Differentiates known oncogenic driver/loss-of-function variants from Variants of
    Unknown Significance (VUS), enforcing clinical abstention or pivoting to
    research-only rescue pipelines based on structural/database indicators.
    """

    TSG_LIST = {"TP53", "PTEN", "BRCA1", "BRCA2", "APC", "RB1", "VHL"}

    def __init__(self, destabilizing_threshold: float = 1.5, core_burial_plddt: float = 70.0):
        self.destabilizing_threshold = destabilizing_threshold
        self.core_burial_plddt = core_burial_plddt

    def route_target(
        self,
        gene: str,
        mutation: str,
        evidence_hits: List[Dict[str, Any]],
        structure_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the graded decision routing tree.

        Args:
            gene: Target gene name (e.g. 'EGFR', 'TP53')
            mutation: Amino acid change (e.g. 'R175H', 'E545K')
            evidence_hits: Retrieved literature or database items (CIViC, ClinVar, OncoKB)
            structure_features: Extracted physical features (pLDDT, ddG, burial metrics)

        Returns:
            Dict containing routing decisions, evidence classification, and safety flags.
        """
        evidence_tier: EvidenceTier = "none"
        classification: str = "VUS"
        allow_confident_therapy = False
        vus_branch = "VUS_uncertain"
        pathway = "inhibitor_rag"
        recommendation_status = "insufficient_evidence"
        next_best_action = "tumor_board"
        confidence_scope = "mechanism"

        # Step 1: Graded Evidence Tiering — inline check supplements score_evidence_tier
        has_strong_kb_hits = False
        if evidence_hits:
            for hit in evidence_hits:
                level = str(hit.get("level", "D")).upper()
                clinical_significance = str(hit.get("clinical_significance", "")).lower()
                if level in {"A", "B", "C"} or "pathogenic" in clinical_significance or hit.get("therapies"):
                    has_strong_kb_hits = True
                    break

        if has_strong_kb_hits:
            evidence_tier = "strong"
            classification = "known_driver"
            vus_branch = "known"
            allow_confident_therapy = True
            recommendation_status = "approved"
            next_best_action = "none"

            if gene in self.TSG_LIST:
                pathway = "structural_rescue"
                confidence_scope = "rescue_success"
            else:
                pathway = "inhibitor_rag"
                confidence_scope = "therapy"

            logger.info("Routed %s %s as Known Driver -> Pathway: %s", gene, mutation, pathway)

        else:
            # Step 2: VUS / Unannotated Variant Path
            evidence_tier = "weak" if evidence_hits else "none"
            allow_confident_therapy = False  # strict safety abstention
            next_best_action = "functional_assay"

            # Structural proxies — map both naming conventions
            ddg = float(
                structure_features.get("thermo_ddg")
                or structure_features.get("mutant_ddg_kcal_mol")
                or 0.0
            )
            plddt = float(
                structure_features.get("residue_plddt")
                or structure_features.get("pLDDT_at_residue")
                or 0.0
            )
            neighbors = int(
                structure_features.get("neighbors_10a")
                or structure_features.get("neighboring_residues_within_10A")
                or 0
            )

            # Branch B: VUS Likely GOF
            # Activating mutations at surface-accessible regions of kinase/oncogenes
            if gene not in self.TSG_LIST and plddt > 80.0 and ddg < self.destabilizing_threshold and neighbors < 12:
                vus_branch = "VUS_likely_GOF"
                classification = "VUS-high"
                pathway = "inhibitor_rag"
                recommendation_status = "investigational"
                next_best_action = "literature_review"
                confidence_scope = "mechanism"
                logger.info("Routed %s %s as VUS Likely GOF", gene, mutation)

            # Branch C: VUS Likely LOF
            # Buried TSG residues with structural destabilization.
            # ddg == 0.0 means not yet computed (rescue runs post-routing); treat as
            # "unknown destabilization" and conservatively route to structural_rescue
            # for buried TSG residues (pLDDT proxy for core burial).
            elif gene in self.TSG_LIST and plddt >= self.core_burial_plddt and (
                ddg >= self.destabilizing_threshold or ddg == 0.0
            ):
                vus_branch = "VUS_likely_LOF"
                classification = "VUS-high"
                pathway = "structural_rescue"
                recommendation_status = "insufficient_evidence"
                next_best_action = "functional_assay"
                confidence_scope = "rescue_success"
                logger.info("Routed %s %s as VUS Destabilizing LOF -> Research Rescue Pathway", gene, mutation)

            # Branch D: Likely Benign
            elif ddg < 0.5 and plddt > 60.0 and neighbors < 8:
                vus_branch = "likely_benign"
                classification = "likely_benign"
                pathway = "inhibitor_rag"
                recommendation_status = "none_direct"
                next_best_action = "none"
                confidence_scope = "global_fold"
                logger.info("Routed %s %s as Likely Benign", gene, mutation)

            # Branch E: VUS Uncertain / Conflicting Signals
            else:
                vus_branch = "VUS_mid"
                classification = "VUS-mid"
                pathway = "inhibitor_rag"
                recommendation_status = "insufficient_evidence"
                next_best_action = "tumor_board"
                confidence_scope = "local_region"
                logger.info("Routed %s %s as VUS-mid Uncertain", gene, mutation)

        return {
            "gene": gene,
            "mutation": mutation,
            "evidence_tier": evidence_tier,
            "classification": classification,
            "vus_branch": vus_branch,
            "allow_confident_therapy": allow_confident_therapy,
            "pathway": pathway,
            "recommendation_status": recommendation_status,
            "next_best_action": next_best_action,
            "confidence_scope": confidence_scope,
            "mechanism_hypothesis": (
                f"VUS analysis for {gene} {mutation} under {vus_branch} branch."
                if classification != "known_driver"
                else "Established pathogenic driver pathway."
            ),
        }


def route_variant(
    target: dict,
    structure: dict,
    evidence: list[dict],
) -> dict[str, Any]:
    """
    Pipeline-compatible wrapper around VariantRouter.route_target().

    Preserves the call signature used by pipeline.py while delegating all
    routing logic to the full VariantRouter decision tree.
    """
    # Supplement inline evidence check with score_evidence_tier for consistency
    # with the existing evidence module (handles source/significance fields).
    tier_from_module = score_evidence_tier(evidence)

    gene = target.get("gene", "")
    mutation = target.get("mutation", "")

    router = VariantRouter()
    routing = router.route_target(gene, mutation, evidence, structure)

    # If the evidence module scores stronger than the inline check, honour it.
    if tier_from_module == "strong" and routing["evidence_tier"] != "strong":
        routing["evidence_tier"] = "strong"

    return routing
