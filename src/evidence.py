"""ClinVar / CIViC / OncoKB / PubMed retrieval with cache + simple vector index."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src import metrics

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
INDEX = CACHE / "pubmed_index.json"


def _cache_read(name: str) -> Any | None:
    p = CACHE / name
    if p.exists():
        return json.loads(p.read_text())
    return None


def _cache_write(name: str, data: Any) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / name).write_text(json.dumps(data, indent=2))


def fetch_clinvar(gene: str, mutation: str) -> list[dict]:
    key = f"clinvar_{gene}_{mutation}.json"
    cached = _cache_read(key)
    if cached is not None:
        return cached
    term = urllib.parse.quote(f"{gene}[gene] AND {mutation}")
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=clinvar&term={term}&retmode=json&retmax=5"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read())
        ids = data.get("esearchresult", {}).get("idlist", [])
        out = [{"source": "ClinVar", "ids": ids, "term": term}]
        _cache_write(key, out)
        return out
    except Exception:
        return []


def fetch_oncokb_benchmark(gene: str, mutation: str) -> list[dict]:
    """Load cached OncoKB benchmark snapshot (validation ONLY - not for training)."""
    key = f"oncokb_{gene}_{mutation}.json"
    cached = _cache_read(key)
    if cached is not None:
        return cached
    return []


def fetch_pubmed_abstracts(query: str, max_results: int = 3) -> list[dict]:
    key = f"pubmed_{re.sub(r'[^a-zA-Z0-9]', '_', query)[:40]}.json"
    cached = _cache_read(key)
    if cached is not None:
        return cached
    q = urllib.parse.quote(query)
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={q}&retmode=json&retmax={max_results}"
    )
    try:
        with urllib.request.urlopen(search_url, timeout=20) as resp:
            ids = json.loads(resp.read()).get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fetch_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&id={','.join(ids)}&retmode=xml"
        )
        with urllib.request.urlopen(fetch_url, timeout=20) as resp:
            root = ET.fromstring(resp.read())
        out = []
        for article in root.findall(".//PubmedArticle"):
            title = (article.findtext(".//ArticleTitle") or "").strip()
            abstract = " ".join(
                t.text or "" for t in article.findall(".//AbstractText")
            ).strip()
            pmid = article.findtext(".//PMID") or ""
            out.append({
                "source": "PubMed",
                "pmid": pmid,
                "title": title,
                "abstract": abstract[:500],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        _cache_write(key, out)
        _update_vector_index(out)
        return out
    except Exception:
        return []


def _update_vector_index(docs: list[dict]) -> None:
    idx = _cache_read("pubmed_index.json") or []
    seen = {d.get("pmid") for d in idx}
    for d in docs:
        if d.get("pmid") not in seen:
            idx.append(d)
    _cache_write("pubmed_index.json", idx)


def search_literature(query: str, k: int = 3) -> list[dict]:
    """Simple keyword retrieval over cached PubMed index (fallback to live fetch)."""
    idx = _cache_read("pubmed_index.json") or []
    if not idx:
        return fetch_pubmed_abstracts(query, k)
    qtok = set(query.lower().split())
    scored = []
    for doc in idx:
        text = f"{doc.get('title', '')} {doc.get('abstract', '')}".lower()
        score = sum(1 for t in qtok if t in text)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]] or fetch_pubmed_abstracts(query, k)


def fetch_civic_evidence(gene: str, mutation: str, profile: str) -> list[dict]:
    key = f"civic_{gene}_{mutation}.json"
    cached = _cache_read(key)
    if cached is not None:
        return cached
    query = """
    query($name: String!) {
      molecularProfile(name: $name) {
        evidenceItems { id direction therapies { name } disease { name } significance citation { id } }
      }
    }"""
    payload = json.dumps({"query": query, "variables": {"name": profile}}).encode()
    req = urllib.request.Request(
        "https://civicdb.org/api/graphql",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        items = data.get("data", {}).get("molecularProfile", {}).get("evidenceItems") or []
        out = []
        for it in items:
            out.append({
                "source": "CIViC",
                "direction": it.get("direction"),
                "therapies": ",".join(t["name"] for t in (it.get("therapies") or [])),
                "disease": (it.get("disease") or {}).get("name"),
                "citation": f"CIViC evidence {it.get('id')}",
                "url": f"https://civicdb.org/links/evidence_items/{it.get('id')}",
            })
        _cache_write(key, out)
        return out
    except Exception:
        return []


def load_case_evidence(gene: str, mutation: str) -> list[dict]:
    case_path = Path(__file__).resolve().parents[1] / "data" / "cases" / f"{gene}_{mutation}.json"
    if case_path.exists():
        return json.loads(case_path.read_text()).get("evidence", [])
    return []


def score_evidence_tier(evidence: list[dict]) -> str:
    """Classify evidence strength: strong | weak | none."""
    if not evidence:
        return "none"
    strong_sources = {"civic", "oncokb", "clinvar"}
    has_therapy = False
    strong_hits = 0
    for item in evidence:
        src = (item.get("source") or "").lower()
        therapies = (item.get("therapies") or "").strip()
        if therapies:
            has_therapy = True
        if any(s in src for s in strong_sources):
            strong_hits += 1
        if item.get("level") in ("A", "B") or item.get("significance"):
            strong_hits += 1
    if strong_hits >= 1 and has_therapy:
        return "strong"
    if evidence:
        return "weak"
    return "none"


def gather_evidence(target: dict, live: bool = False) -> list[dict]:
    with metrics.track("evidence_gather", agent_role="Evidence", model="cache+api"):
        ev = load_case_evidence(target["gene"], target["mutation"])
        merged = list(ev)
        query = f"{target['gene']} {target['mutation']} cancer therapy mechanism"
        merged.extend(search_literature(query, k=2))
        if live:
            civic = fetch_civic_evidence(
                target["gene"], target["mutation"], target.get("civic_profile", "")
            )
            merged.extend(civic)
            merged.extend(fetch_clinvar(target["gene"], target["mutation"]))
        return merged
