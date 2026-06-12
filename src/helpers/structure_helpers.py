"""Reusable structure helpers adapted from mini_protein_pipeline_6a95."""
from __future__ import annotations

import math
from pathlib import Path


def parse_ca_coords(pdb_path: Path, chain: str = "A") -> dict[int, tuple[float, float, float]]:
    coords: dict[int, tuple[float, float, float]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            if line[21] != chain:
                continue
            resseq = int(line[22:26])
            if resseq in coords:
                continue
            coords[resseq] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return coords


def resseq_to_mpnn_index(pdb_path: Path, chain: str, resseq: int) -> int | None:
    keys = sorted(parse_ca_coords(pdb_path, chain).keys())
    for i, rs in enumerate(keys):
        if rs == resseq:
            return i + 1
    return None


def local_contact_density(pdb_path: Path, residue: int, chain: str = "A", radius: float = 10.0) -> int:
    coords = parse_ca_coords(pdb_path, chain)
    if residue not in coords:
        return 0
    tx, ty, tz = coords[residue]
    n = 0
    for rs, (x, y, z) in coords.items():
        if rs == residue:
            continue
        d = math.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        if d <= radius:
            n += 1
    return n


def ca_rmsd(pdb_a: Path, pdb_b: Path, chain: str = "A") -> float | None:
    ca = parse_ca_coords(pdb_a, chain)
    cb = parse_ca_coords(pdb_b, chain)
    common = sorted(set(ca) & set(cb))
    if len(common) < 3:
        return None
    sse = 0.0
    for rs in common:
        ax, ay, az = ca[rs]
        bx, by, bz = cb[rs]
        sse += (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
    return math.sqrt(sse / len(common))
