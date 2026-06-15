"""Biologic Rescue: ProteinMPNN redesign + ThermoMPNN ddG + ESMFold/Boltz fold."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src import metrics
from src.config import load_config, shared_dir
from src.helpers.structure_helpers import (
    count_chain_residues,
    extract_local_shell_pdb,
    mpnn_index_map,
    thermompnn_index_for_resseq,
)

ROOT = Path(__file__).resolve().parents[1]
THERMOMPNN_DIR = ROOT / "external" / "ThermoMPNN"
MPNN_REPO = ROOT / "external" / "ProteinMPNN"


def _thermompnn_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(THERMOMPNN_DIR), str(THERMOMPNN_DIR / "analysis"), env.get("PYTHONPATH", "")]
    )
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return env


def patch_thermompnn_local_yaml(thermompnn_dir: Path | None = None) -> None:
    """Point platform.thermompnn_dir at the local clone (probe-night fix)."""
    import yaml
    d = thermompnn_dir or THERMOMPNN_DIR
    local_yaml = d / "local.yaml"
    if not local_yaml.exists():
        return
    cfg = yaml.safe_load(local_yaml.read_text()) or {}
    cfg.setdefault("platform", {})["thermompnn_dir"] = str(d.resolve())
    local_yaml.write_text(yaml.dump(cfg, default_flow_style=False))


def _thermompnn_model_path() -> Path:
    candidates = [
        THERMOMPNN_DIR / "models" / "thermoMPNN_default.pt",
        THERMOMPNN_DIR / "models" / "thermoMPNN_default.ckpt",
    ]
    for folder in ("models", "vanilla_model_weights"):
        d = THERMOMPNN_DIR / folder
        if d.is_dir():
            for fn in sorted(d.iterdir()):
                if fn.suffix in (".pt", ".ckpt"):
                    candidates.insert(0, fn)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"No ThermoMPNN checkpoint under {THERMOMPNN_DIR}; run setup_external.sh or pass weights to models/"
    )


def _ensure_thermompnn_deps() -> None:
    missing: list[str] = []
    for mod, pkg in (
        ("pytorch_lightning", "pytorch-lightning"),
        ("torchmetrics", "torchmetrics"),
        ("omegaconf", "omegaconf"),
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            "ThermoMPNN requires: pip install "
            + " ".join(missing)
            + "  (or pip install -r requirements.txt)"
        )


def score_ddg_site_saturation(
    pdb_path: Path,
    chain: str,
    out_dir: Path,
    *,
    center_residue: int | None = None,
    shell_radius: float | None = None,
    max_full_protein_residues: int = 80,
) -> Path:
    """Run ThermoMPNN site-saturation on a PDB; return CSV path."""
    _ensure_thermompnn_deps()
    patch_thermompnn_local_yaml()
    out_dir.mkdir(parents=True, exist_ok=True)
    script = ROOT / "scripts" / "thermompnn_ssm.py"
    if not script.exists():
        raise FileNotFoundError(f"ThermoMPNN wrapper not found at {script}")
    if not THERMOMPNN_DIR.exists():
        raise FileNotFoundError(f"ThermoMPNN not found at {THERMOMPNN_DIR}; run setup_external.sh")

    run_pdb = pdb_path
    if center_residue is not None and count_chain_residues(pdb_path, chain) > max_full_protein_residues:
        radius = shell_radius if shell_radius is not None else 15.0
        run_pdb = extract_local_shell_pdb(
            pdb_path,
            chain,
            center_residue,
            radius,
            out_dir / f"thermo_shell_r{center_residue}.pdb",
        )

    cmd = [
        sys.executable, str(script),
        "--pdb", str(run_pdb),
        "--chain", chain,
        "--model_path", str(_thermompnn_model_path()),
        "--out_dir", str(out_dir),
    ]
    with metrics.track("thermompnn_ssm", agent_role="Rescue", model="thermoMPNN"):
        proc = subprocess.run(
            cmd, check=False, env=_thermompnn_env(), cwd=str(ROOT),
            capture_output=True, text=True,
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        tail = detail[-4000:] if len(detail) > 4000 else detail
        raise RuntimeError(
            f"ThermoMPNN failed (exit {proc.returncode}) on {run_pdb}:\n{tail or '(no output)'}"
        )
    csvs = list(out_dir.glob("ThermoMPNN_inference_*.csv"))
    if not csvs:
        raise RuntimeError("ThermoMPNN produced no CSV")
    return csvs[0]


def mutation_ddg(
    csv_path: Path,
    wt: str,
    position: int,
    mut: str,
    *,
    resseq: int | None = None,
) -> float | None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    wt_u, mut_u = wt.upper(), mut.upper()
    if resseq is not None and "resseq" in df.columns:
        by_site = df[df["resseq"].astype(int) == int(resseq)]
        if not by_site.empty:
            wt_s = by_site["wildtype"].astype(str).str.upper()
            mut_s = by_site["mutation"].astype(str).str.upper()
            row = by_site[(wt_s == wt_u) & (mut_s == mut_u)]
            if row.empty:
                row = by_site[mut_s == mut_u]
            if not row.empty:
                return float(row.iloc[0]["ddG_pred"])
    pos_col = df["position"].astype(int)
    wt_col = df["wildtype"].astype(str).str.upper()
    mut_col = df["mutation"].astype(str).str.upper()
    for p in (int(position), int(position) + 1):
        row = df[(wt_col == wt_u) & (pos_col == p) & (mut_col == mut_u)]
        if not row.empty:
            return float(row.iloc[0]["ddG_pred"])
    return None


def fold_esmfold(sequence: str, out_pdb: Path) -> Path:
    setup = load_config()
    os.environ.setdefault("HF_HOME", setup.get("paths", {}).get("hf_cache", "/workspace/shared/hf_cache"))
    with metrics.track("esmfold_fold", agent_role="Rescue", model="facebook/esmfold_v1"):
        import torch
        from transformers import EsmForProteinFolding, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        if torch.cuda.is_available():
            model = model.cuda().eval()
            pdb = model.infer_pdb(sequence)
        else:
            model = model.eval()
            pdb = model.infer_pdb(sequence)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text(pdb)
    return out_pdb


def fold_boltz(sequence: str, yaml_path: Path, out_dir: Path, cache_dir: Path) -> Path | None:
    """Run Boltz in isolated env with --no_kernels (ROCm probe validated). Skips if not installed."""
    boltz_bin = os.environ.get("BOLTZ_BIN", "boltz")
    if not shutil.which(boltz_bin):
        return None
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {sequence}\n      msa: empty\n"
    )
    cmd = [
        boltz_bin, "predict", str(yaml_path.name),
        "--out_dir", str(out_dir),
        "--cache", str(cache_dir),
        "--accelerator", "gpu", "--devices", "1",
        "--recycling_steps", "1", "--diffusion_samples", "1",
        "--no_kernels", "--output_format", "pdb",
    ]
    with metrics.track("boltz_fold", agent_role="Rescue", model="boltz-2.2.1"):
        try:
            r = subprocess.run(cmd, cwd=str(yaml_path.parent), capture_output=True, text=True)
        except OSError:
            return None
        if r.returncode != 0:
            return None
    pdbs = list(out_dir.rglob("*.pdb"))
    return pdbs[0] if pdbs else None


def run_proteinmpnn_redesign(
    pdb_path: Path,
    chain: str,
    fixed_positions: list[int],
    out_dir: Path,
    *,
    temperature: float = 0.8,
    num_seqs: int = 8,
) -> list[dict]:
    """Shell out to dauparas/ProteinMPNN if cloned; else return stub for CPU dev."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_py = MPNN_REPO / "protein_mpnn_run.py"
    if not run_py.exists():
        return [{"sequence": "STUB", "score": 0.0, "note": "Clone ProteinMPNN to external/ProteinMPNN"}]
    pdb_name = pdb_path.stem
    fixed_json = out_dir / "fixed_positions.jsonl"
    # ProteinMPNN expects JSONL: {pdb_name: {chain: [1-based indices to keep fixed]}}
    fixed_json.write_text(json.dumps({pdb_name: {chain: sorted(fixed_positions)}}) + "\n")
    cmd = [
        sys.executable, str(run_py),
        "--pdb_path", str(pdb_path),
        "--pdb_path_chains", chain,
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(temperature),
        "--fixed_positions_jsonl", str(fixed_json),
        "--suppress_print", "1",
    ]
    with metrics.track("proteinmpnn_redesign", agent_role="Rescue", model="ProteinMPNN"):
        proc = subprocess.run(cmd, check=False, cwd=str(MPNN_REPO), capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        tail = detail[-4000:] if len(detail) > 4000 else detail
        raise RuntimeError(
            f"ProteinMPNN failed (exit {proc.returncode}) on {pdb_path}:\n{tail or '(no output)'}"
        )
    fastas = list(out_dir.rglob("*.fa"))
    designs = []
    for fa in fastas:
        lines = fa.read_text().splitlines()
        for i in range(2, len(lines), 2):
            hdr, seq = lines[i], lines[i + 1]
            score = 0.0
            if "score=" in hdr:
                score = float(hdr.split("score=")[1].split(",")[0])
            designs.append({"header": hdr, "sequence": seq, "score": score, "fasta": str(fa)})
    designs.sort(key=lambda d: d["score"])
    return designs


def run_rescue(
    target: dict,
    pdb_path: Path,
    *,
    chain: str = "A",
    work_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    rescue_cfg = cfg.get("rescue", {})
    work = work_dir or (shared_dir(cfg) / "rescue" / f"{target['gene']}_{target['mutation']}")
    work.mkdir(parents=True, exist_ok=True)

    with metrics.phase(f"rescue_{target['gene']}_{target['mutation']}", model="structural_stack"):
        pos = target["residue"]
        wt, mut = target["wt_aa"], target["mut_aa"]
        thermo_dir = work / "thermompnn"
        csv_path = score_ddg_site_saturation(
            pdb_path,
            chain,
            thermo_dir,
            center_residue=pos,
            shell_radius=rescue_cfg.get("thermompnn_shell_radius", 15.0),
            max_full_protein_residues=rescue_cfg.get("thermompnn_full_protein_max_residues", 80),
        )
        shell_pdb = thermo_dir / f"thermo_shell_r{pos}.pdb"
        thermo_pdb = shell_pdb if shell_pdb.exists() else pdb_path
        thermo_pos = thermompnn_index_for_resseq(thermo_pdb, chain, pos)
        if thermo_pos is None:
            thermo_pos = pos
        mut_ddg = mutation_ddg(csv_path, wt, thermo_pos, mut, resseq=pos)

        mpnn_pdb = thermo_pdb
        shell_resseqs = list(range(max(1, pos - 5), pos + 6))
        design_resseqs = [r for r in shell_resseqs if r != pos]
        idx_map = mpnn_index_map(mpnn_pdb, chain)
        design_idx = {idx_map[r] for r in design_resseqs if r in idx_map}
        n_res = count_chain_residues(mpnn_pdb, chain)
        fixed_mpnn = [i for i in range(1, n_res + 1) if i not in design_idx]
        designs = run_proteinmpnn_redesign(
            mpnn_pdb,
            chain,
            fixed_mpnn,
            work / "mpnn",
            temperature=rescue_cfg.get("proteinmpnn_temperature", 0.8),
            num_seqs=rescue_cfg.get("n_designs", 8),
        )

        best = designs[0] if designs else {}
        fold_pdb = None
        fold_method = None
        if best.get("sequence") and best["sequence"] != "STUB":
            fold_pdb = fold_esmfold(best["sequence"], work / "esmfold_candidate.pdb")
            fold_method = "esmfold"
            boltz_out = fold_boltz(
                best["sequence"],
                work / "boltz" / "input.yaml",
                work / "boltz" / "out",
                shared_dir(cfg) / "boltz_cache",
            )
            if boltz_out:
                fold_pdb = boltz_out
                fold_method = "boltz"

        return {
            "mutant_ddg_kcal_mol": mut_ddg,
            "destabilizing": mut_ddg is not None and mut_ddg > rescue_cfg.get("ddg_destabilizing_threshold", 1.0),
            "ddg_scope": "local_shell_15A",
            "ddg_note": (
                "Shell-local ThermoMPNN SSM; literature R175H is often >1 kcal/mol destabilizing on full domain."
                if mut_ddg is not None and mut_ddg <= rescue_cfg.get("ddg_destabilizing_threshold", 1.0)
                else None
            ),
            "designs": designs[:5],
            "folded_candidate_pdb": str(fold_pdb) if fold_pdb else None,
            "fold_method": fold_method,
            "thermompnn_csv": str(csv_path),
            "thermompnn_shell_pdb": str(shell_pdb) if shell_pdb.exists() else None,
            "proteinmpnn_pdb": str(mpnn_pdb),
            "ddg_engine": "ThermoMPNN",
        }
