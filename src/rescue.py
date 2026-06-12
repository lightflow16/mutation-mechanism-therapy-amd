"""Biologic Rescue: ProteinMPNN redesign + ThermoMPNN ddG + ESMFold/Boltz fold."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src import metrics
from src.config import load_config, shared_dir

ROOT = Path(__file__).resolve().parents[1]
THERMOMPNN_DIR = ROOT / "external" / "ThermoMPNN"
MPNN_REPO = ROOT / "external" / "ProteinMPNN"


def _thermompnn_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{THERMOMPNN_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
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


def score_ddg_site_saturation(pdb_path: Path, chain: str, out_dir: Path) -> Path:
    """Run ThermoMPNN custom_inference on a PDB; return CSV path."""
    patch_thermompnn_local_yaml()
    out_dir.mkdir(parents=True, exist_ok=True)
    script = THERMOMPNN_DIR / "analysis" / "custom_inference.py"
    if not script.exists():
        raise FileNotFoundError(f"ThermoMPNN not found at {THERMOMPNN_DIR}; git clone Kuhlman-Lab/ThermoMPNN")
    cmd = [
        sys.executable, str(script),
        "--pdb", str(pdb_path),
        "--chain", chain,
        "--model_path", str(THERMOMPNN_DIR / "models" / "thermoMPNN_default.pt"),
        "--out_dir", str(out_dir),
    ]
    with metrics.track("thermompnn_ssm", agent_role="Rescue", model="thermoMPNN"):
        subprocess.run(cmd, check=True, env=_thermompnn_env(), cwd=str(THERMOMPNN_DIR))
    csvs = list(out_dir.glob("ThermoMPNN_inference_*.csv"))
    if not csvs:
        raise RuntimeError("ThermoMPNN produced no CSV")
    return csvs[0]


def mutation_ddg(csv_path: Path, wt: str, position: int, mut: str) -> float | None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    row = df[(df["wildtype"] == wt) & (df["position"] == position) & (df["mutation"] == mut)]
    if row.empty:
        return None
    return float(row.iloc[0]["ddG_pred"])


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
    """Run Boltz in isolated env with --no_kernels (ROCm probe validated)."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {sequence}\n      msa: empty\n"
    )
    boltz_bin = os.environ.get("BOLTZ_BIN", "boltz")
    cmd = [
        boltz_bin, "predict", str(yaml_path.name),
        "--out_dir", str(out_dir),
        "--cache", str(cache_dir),
        "--accelerator", "gpu", "--devices", "1",
        "--recycling_steps", "1", "--diffusion_samples", "1",
        "--no_kernels", "--output_format", "pdb",
    ]
    with metrics.track("boltz_fold", agent_role="Rescue", model="boltz-2.2.1"):
        r = subprocess.run(cmd, cwd=str(yaml_path.parent), capture_output=True, text=True)
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
    fixed_json = out_dir / "fixed_positions.json"
    fixed_json.write_text(json.dumps({chain: {str(p): p for p in fixed_positions}}))
    cmd = [
        sys.executable, str(run_py),
        "--pdb_path", str(pdb_path),
        "--pdb_path_chains", chain,
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(temperature),
        "--fixed_positions_jsonl", str(fixed_json),
    ]
    with metrics.track("proteinmpnn_redesign", agent_role="Rescue", model="ProteinMPNN"):
        subprocess.run(cmd, check=True, cwd=str(MPNN_REPO))
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
        csv_path = score_ddg_site_saturation(pdb_path, chain, work / "thermompnn")
        pos = target["residue"]
        wt, mut = target["wt_aa"], target["mut_aa"]
        mut_ddg = mutation_ddg(csv_path, wt, pos, mut)

        residue, _, _ = __import__("src.structure", fromlist=["parse_mutation"]).parse_mutation(
            target["mutation"]
        )
        shell = list(range(max(1, pos - 5), pos + 6))
        designs = run_proteinmpnn_redesign(
            pdb_path, chain, [p for p in shell if p != pos],
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
            "designs": designs[:5],
            "folded_candidate_pdb": str(fold_pdb) if fold_pdb else None,
            "fold_method": fold_method,
            "thermompnn_csv": str(csv_path),
            "ddg_engine": "ThermoMPNN",
        }
