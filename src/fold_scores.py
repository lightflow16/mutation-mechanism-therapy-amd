"""Parse pLDDT / pTM from fold PDBs and Boltz output JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def pdb_plddt_stats(pdb_path: Path | str | None) -> dict[str, float]:
    """Mean/min pLDDT from PDB B-factors (AlphaFold/ESMFold/Boltz convention)."""
    if not pdb_path or not Path(pdb_path).is_file():
        return {}
    bfactors: list[float] = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            bfactors.append(float(line[60:66].strip()))
        except (ValueError, IndexError):
            continue
    if not bfactors:
        return {}
    return {
        "mean_plddt": round(sum(bfactors) / len(bfactors), 2),
        "min_plddt": round(min(bfactors), 2),
        "n_residues": float(len(bfactors)),
    }


def parse_boltz_scores(out_dir: Path | str | None) -> dict[str, Any]:
    """Best-effort parse of Boltz confidence JSON under output tree."""
    if not out_dir:
        return {}
    root = Path(out_dir)
    if not root.is_dir():
        return {}
    out: dict[str, Any] = {}
    for pattern in ("**/confidence*.json", "**/scores.json", "**/*confidence*.json"):
        for p in root.glob(pattern):
            try:
                obj = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(obj, dict):
                for k in ("ptm", "iptm", "complex_plddt", "plddt", "pae_mean"):
                    if k in obj and obj[k] is not None:
                        out[f"boltz_{k}" if not k.startswith("boltz_") else k] = float(obj[k])
                if "confidence_score" in obj:
                    out["boltz_ptm"] = out.get("boltz_ptm", float(obj["confidence_score"]))
            if out:
                out["boltz_scores_path"] = str(p)
                return out
    return out


def attach_fold_scores(rescue: dict, *, boltz_out_dir: Path | None = None) -> dict:
    """Enrich rescue dict with parsed fold quality scores."""
    if rescue.get("esmfold_pdb"):
        stats = pdb_plddt_stats(rescue["esmfold_pdb"])
        if stats:
            rescue["esmfold_plddt"] = stats.get("mean_plddt")
            rescue["esmfold_min_plddt"] = stats.get("min_plddt")
    if rescue.get("boltz_pdb"):
        stats = pdb_plddt_stats(rescue["boltz_pdb"])
        if stats:
            rescue["boltz_plddt"] = stats.get("mean_plddt")
            rescue["boltz_min_plddt"] = stats.get("min_plddt")
    boltz_dir = boltz_out_dir
    if boltz_dir is None and rescue.get("boltz_pdb"):
        boltz_dir = Path(str(rescue["boltz_pdb"])).parent.parent
    parsed = parse_boltz_scores(boltz_dir)
    for k, v in parsed.items():
        if k != "boltz_scores_path":
            rescue.setdefault(k, v)
        else:
            rescue["boltz_scores_json"] = Path(v).read_text() if Path(v).is_file() else rescue.get("boltz_scores_json")
    return rescue
