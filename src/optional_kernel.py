"""OPTIONAL stretch: Triton-on-ROCm neighbor-list kernel benchmark stub."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np


def numpy_neighbor_count(coords: np.ndarray, radius: float = 10.0) -> int:
    n = len(coords)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= radius:
                count += 1
    return count


def benchmark_contacts(pdb_path: Path, chain: str = "A") -> dict:
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    arr = np.array(coords)
    t0 = time.perf_counter()
    c = numpy_neighbor_count(arr)
    dt = time.perf_counter() - t0
    return {"contacts_within_10A": c, "numpy_seconds": round(dt, 4), "note": "Triton kernel TBD"}
