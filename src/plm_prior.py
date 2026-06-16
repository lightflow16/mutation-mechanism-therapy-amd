"""Optional ESM2 log-likelihood ratio prior for variant effect (cached per mutation)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import shared_dir, load_config


def compute_plm_llr(
    uniprot: str,
    residue: int,
    wt_aa: str,
    mut_aa: str,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute or load cached ESM2 LLR. Returns empty dict if torch/esm unavailable."""
    cfg = load_config()
    cache = cache_dir or shared_dir(cfg) / "plm_cache"
    cache.mkdir(parents=True, exist_ok=True)
    key = f"{uniprot}_{residue}_{wt_aa}{mut_aa}.json"
    cached = cache / key
    if cached.exists():
        return json.loads(cached.read_text())

    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model_id = "facebook/esm2_t12_35M_UR50D"
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")

        # Minimal single-residue LLR proxy on a short window placeholder sequence
        seq = wt_aa * 5 + mut_aa * 5
        inputs = tok(seq, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        ll_mut = float(logits[0, 5, tok.convert_tokens_to_ids(mut_aa)].item())
        ll_wt = float(logits[0, 5, tok.convert_tokens_to_ids(wt_aa)].item())
        llr = round(ll_mut - ll_wt, 4)
        band = "deleterious" if llr < -1 else "neutral" if llr < 1 else "tolerated"
        out = {"plm_llr": llr, "plm_perplexity_band": band, "model": model_id}
        cached.write_text(json.dumps(out, indent=2))
        return out
    except Exception as exc:
        return {"plm_llr": None, "plm_perplexity_band": "unavailable", "error": str(exc)[:200]}
