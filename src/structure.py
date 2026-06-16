"""AlphaFold structure fetch, HGVS mapping, numeric features, py3Dmol render."""
from __future__ import annotations

import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any

from src import metrics
from src.helpers.structure_helpers import local_contact_density, ca_rmsd, resseq_to_mpnn_index

AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA1 = {v: k for k, v in AA3.items()}


def parse_mutation(mutation: str) -> tuple[int, str, str]:
    """Normalize L858R, p.Leu858Arg, Leu858Arg -> (858, L, R)."""
    s = mutation.strip().upper()
    s = re.sub(r"^P\.", "", s, flags=re.I)
    m = re.match(r"^([A-Z]{3})(\d+)([A-Z]{3})$", s)
    if m:
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        return pos, AA1.get(wt, wt[0]), AA1.get(mut, mut[0])
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", s)
    if m:
        return int(m.group(2)), m.group(1), m.group(3)
    m = re.match(r"^(\d+)$", s)
    if m:
        raise ValueError(f"Residue number only ({s}); need wt+mut e.g. L858R")
    raise ValueError(f"Cannot parse mutation: {mutation}")


def alphafold_api_url(uniprot: str) -> str:
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    entry = data[0] if isinstance(data, list) else data
    return entry["pdbUrl"]


def fetch_pdb(uniprot: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"AF-{uniprot}-F1-model_v6.pdb"
    if cached.exists():
        return cached
    pdb_url = alphafold_api_url(uniprot)
    with urllib.request.urlopen(pdb_url, timeout=120) as resp:
        cached.write_bytes(resp.read())
    return cached


def _parse_ca(pdb_path: Path, chain: str = "A") -> list[dict]:
    atoms = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain:
                continue
            resseq = int(line[22:26])
            if resseq in seen:
                continue
            seen.add(resseq)
            resname = line[17:20].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            b = float(line[60:66]) if len(line) >= 66 else 0.0
            atoms.append({"resseq": resseq, "resname": resname, "xyz": (x, y, z), "plddt": b})
    return sorted(atoms, key=lambda a: a["resseq"])


def validate_residue(atoms: list[dict], residue: int, wt_aa: str) -> None:
    hit = next((a for a in atoms if a["resseq"] == residue), None)
    if hit is None:
        raise ValueError(f"Residue {residue} not in PDB")
    expected = AA3.get(wt_aa.upper(), wt_aa.upper())[:3]
    if hit["resname"] != expected:
        raise ValueError(
            f"Residue mismatch at {residue}: PDB has {hit['resname']}, expected {expected} ({wt_aa})"
        )


def neighbor_count(atoms: list[dict], residue: int, radius: float = 10.0) -> int:
    target = next(a for a in atoms if a["resseq"] == residue)
    tx, ty, tz = target["xyz"]
    n = 0
    for a in atoms:
        if a["resseq"] == residue:
            continue
        dx, dy, dz = a["xyz"][0] - tx, a["xyz"][1] - ty, a["xyz"][2] - tz
        if math.sqrt(dx * dx + dy * dy + dz * dz) <= radius:
            n += 1
    return n


def infer_region(gene: str, residue: int) -> str:
    regions = {
        "EGFR": {(755, 790): "kinase activation loop (A-loop)", (745, 754): "ATP-binding P-loop"},
        "PIK3CA": {(540, 550): "activation loop", (520, 538): "helical domain"},
        "TP53": {(170, 180): "DNA-binding surface (beta-sandwich core)"},
    }
    for (lo, hi), name in regions.get(gene.upper(), {}).items():
        if lo <= residue <= hi:
            return name
    return "unannotated region"


def compute_features(
    pdb_path: Path,
    gene: str,
    mutation: str,
    wt_aa: str,
    chain: str = "A",
) -> dict[str, Any]:
    residue, wt, mut = parse_mutation(mutation)
    if wt != wt_aa.upper():
        wt = wt_aa.upper()
    atoms = _parse_ca(pdb_path, chain)
    validate_residue(atoms, residue, wt)
    hit = next(a for a in atoms if a["resseq"] == residue)
    mean_plddt = sum(a["plddt"] for a in atoms) / len(atoms)
    return {
        "residue": f"{wt}{residue}",
        "residue_index": residue,
        "mutation": f"{wt}{residue}{mut}",
        "pLDDT_at_residue": round(hit["plddt"], 1),
        "mean_pLDDT_protein": round(mean_plddt, 1),
        "neighboring_residues_within_10A": neighbor_count(atoms, residue),
        "local_contact_density_10A": local_contact_density(pdb_path, residue, chain),
        "mpnn_index": resseq_to_mpnn_index(pdb_path, chain, residue),
        "region": infer_region(gene, residue),
        "chain": chain,
        "pdb_path": str(pdb_path),
    }


def render_py3dmol_html(pdb_path: Path, highlight_residue: int, chain: str = "A") -> str:
    pdb_text = pdb_path.read_text()
    style = (
        f"{{'cartoon': {{'color': 'spectrum'}}, "
        f"'stick': {{'resi': {highlight_residue}, 'chain': '{chain}'}}}}"
    )
    return f"""
<div id="viewer" style="width:640px;height:480px;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{backgroundColor:'white'}});
  viewer.addModel({json.dumps(pdb_text)}, 'pdb');
  viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
  viewer.setStyle({{resi: {highlight_residue}, chain: '{chain}'}}, {{stick: {{colorscheme: 'yellowCarbon'}}}});
  viewer.zoomTo({{resi: {highlight_residue}, chain: '{chain}'}});
  viewer.render();
</script>
"""


def export_residue_neighborhood_png(
    pdb_path: Path,
    highlight_residue: int,
    out_path: Path,
    chain: str = "A",
    radius: float = 15.0,
) -> Path | None:
    """Export CA neighborhood scatter PNG for VL multimodal input."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    atoms = _parse_ca(pdb_path, chain)
    if not atoms:
        return None
    target = next((a for a in atoms if a["resseq"] == highlight_residue), None)
    if target is None:
        return None
    tx, ty, tz = target["xyz"]
    nearby = []
    for a in atoms:
        dx, dy, dz = a["xyz"][0] - tx, a["xyz"][1] - ty, a["xyz"][2] - tz
        if (dx * dx + dy * dy + dz * dz) ** 0.5 <= radius:
            nearby.append(a)

    xs = [a["xyz"][0] for a in nearby]
    ys = [a["xyz"][1] for a in nearby]
    cs = [a["plddt"] for a in nearby]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=40, vmin=0, vmax=100)
    ax.scatter([tx], [ty], c="red", marker="*", s=200, label=f"mut {highlight_residue}")
    ax.set_title(f"Residue neighborhood (pLDDT colormap)")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=ax, label="pLDDT")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def analyze_target(target: dict, cache_dir: Path) -> dict[str, Any]:
    with metrics.track("structure_analyze", agent_role="Structure", model="alphafold"):
        pdb = fetch_pdb(target["uniprot"], cache_dir)
        feats = compute_features(
            pdb, target["gene"], target["mutation"], target["wt_aa"]
        )
        feats["render_html"] = render_py3dmol_html(
            pdb, feats["residue_index"], feats.get("chain", "A")
        )
        feats["gene"] = target["gene"]
        png_path = cache_dir / "images" / f"{target['gene']}_{target['mutation']}_neighborhood.png"
        exported = export_residue_neighborhood_png(
            pdb, feats["residue_index"], png_path, feats.get("chain", "A")
        )
        if exported:
            feats["structure_image_path"] = str(exported)
        return feats
