#!/usr/bin/env python3
"""ThermoMPNN site-saturation with correct .pt weight loading.

ThermoMPNN's custom_inference.py calls load_from_checkpoint on thermoMPNN_default.pt,
but that file is a plain state_dict (not a Lightning checkpoint). This wrapper loads
weights into TransferModelPL correctly, then runs SSM like custom_inference.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THERMOMPNN_DIR = ROOT / "external" / "ThermoMPNN"
THERMOMPNN_ANALYSIS = THERMOMPNN_DIR / "analysis"
sys.path.insert(0, str(THERMOMPNN_DIR))
sys.path.insert(0, str(THERMOMPNN_ANALYSIS))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def get_ssm_mutations(pdb: dict) -> list:
    """Site-saturation mutation codes (0-based index), from ThermoMPNN analysis/SSM.py."""
    mutation_list: list = []
    for seq_pos in range(len(pdb["seq"])):
        wt_aa = pdb["seq"][seq_pos]
        if wt_aa != "-":
            for mut_aa in ALPHABET[:-1]:
                mutation_list.append(wt_aa + str(seq_pos) + mut_aa)
        else:
            mutation_list.append(None)
    return mutation_list

INFERENCE_CONFIG = {
    "training": {
        "num_workers": 8,
        "learn_rate": 0.001,
        "epochs": 100,
        "lr_schedule": True,
    },
    "model": {
        "hidden_dims": [64, 32],
        "subtract_mut": True,
        "num_final_layers": 2,
        "freeze_weights": True,
        "load_pretrained": True,
        "lightattn": True,
        "lr_schedule": True,
    },
}


def load_thermompnn_model(model_path: Path, cfg):
    import torch
    from omegaconf import OmegaConf
    from train_thermompnn import TransferModelPL

    if not isinstance(cfg, OmegaConf):
        cfg = OmegaConf.create(cfg)

    raw = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "pytorch-lightning_version" in raw:
        return TransferModelPL.load_from_checkpoint(str(model_path), cfg=cfg).model

    pl_module = TransferModelPL(cfg)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(state, dict):
        raise RuntimeError(f"Unrecognized ThermoMPNN weights at {model_path}")

    if any(k.startswith("model.") for k in state):
        pl_module.load_state_dict(state, strict=False)
    else:
        pl_module.model.load_state_dict(state, strict=False)
    return pl_module.model.eval()


def run_ssm(pdb_path: Path, chain: str, model_path: Path, out_dir: Path) -> Path:
    import pandas as pd
    import torch
    from Bio.PDB import PDBParser
    from omegaconf import OmegaConf

    from datasets import Mutation
    from protein_mpnn_utils import alt_parse_PDB

    local_yaml = THERMOMPNN_DIR / "local.yaml"
    base_cfg = OmegaConf.load(str(local_yaml)) if local_yaml.exists() else OmegaConf.create({})
    merged = OmegaConf.merge(base_cfg, OmegaConf.create(INFERENCE_CONFIG))
    merged.platform = merged.get("platform") or {}
    merged.platform.thermompnn_dir = str(THERMOMPNN_DIR.resolve())

    model = load_thermompnn_model(model_path, merged)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pdb_str = str(pdb_path)
    if not chain:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("", pdb_str)
        chain = next(structure.get_chains()).id

    mut_pdb = alt_parse_PDB(pdb_str, chain)
    mutation_list = get_ssm_mutations(mut_pdb[0])
    final_mutation_list = []
    for m in mutation_list:
        if m is None:
            final_mutation_list.append(None)
            continue
        m = m.strip()
        wt_aa, position, mut_aa = str(m[0]), int(str(m[1:-1])), str(m[-1])
        final_mutation_list.append(
            Mutation(
                position=position,
                wildtype=wt_aa,
                mutation=mut_aa,
                ddG=None,
                pdb=mut_pdb[0]["name"],
            )
        )

    with torch.no_grad():
        pred, _ = model(mut_pdb, final_mutation_list)

    rows = []
    for mut, out in zip(final_mutation_list, pred):
        if mut is None:
            continue
        rows.append(
            {
                "ddG_pred": out["ddG"].cpu().item(),
                "position": mut.position,
                "wildtype": mut.wildtype,
                "mutation": mut.mutation,
                "pdb": mut.pdb.strip(".pdb"),
                "chain": chain,
                "Model": "ThermoMPNN",
                "Dataset": pdb_path.stem,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"ThermoMPNN_inference_{pdb_path.stem}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved ThermoMPNN output to {csv_path}")
    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    run_ssm(Path(args.pdb), args.chain, Path(args.model_path), Path(args.out_dir))


if __name__ == "__main__":
    main()
