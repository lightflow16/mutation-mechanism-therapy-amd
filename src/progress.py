"""Live console echo for metrics — video-demo friendly narrative."""
from __future__ import annotations

import json
import os
import re
from typing import Any

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)

_VERBOSE: bool | None = None
_DEMO_THINKING: bool | None = None
_PREVIEW = 300


def _cfg_bool(key: str, default: bool) -> bool:
    if os.environ.get("METRICS_ECHO") == "0":
        return False
    if os.environ.get("METRICS_ECHO") == "1":
        return True
    try:
        from src.config import load_config

        return bool(load_config().get("pipeline", {}).get(key, default))
    except Exception:
        return default


def verbose() -> bool:
    global _VERBOSE
    if _VERBOSE is None:
        _VERBOSE = _cfg_bool("verbose_progress", True)
    return _VERBOSE


def demo_thinking() -> bool:
    global _DEMO_THINKING
    if _DEMO_THINKING is None:
        _DEMO_THINKING = _cfg_bool("demo_echo_thinking", True)
    return _DEMO_THINKING


def log(stage: str, msg: str, **kv: Any) -> None:
    if not verbose():
        return
    extra = " | ".join(f"{k}={v}" for k, v in kv.items()) if kv else ""
    print(f"[{stage}] {msg}" + (f" | {extra}" if extra else ""), flush=True)


def banner(title: str) -> None:
    if not verbose():
        return
    print("\n" + "=" * 60 + f"\n  {title}\n" + "=" * 60, flush=True)


def _truncate(text: str, limit: int = _PREVIEW) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _extract_thinking(text: str) -> str:
    parts = _THINK_RE.findall(text or "")
    return " ".join(p.strip() for p in parts if p.strip())


def _extract_json_therapy(text: str) -> str:
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= start:
            return ""
        obj = json.loads(text[start:end])
        tr = obj.get("target_reasoning") or obj
        if isinstance(tr, dict) and "target_reasoning" in tr:
            tr = tr["target_reasoning"]
        therapy = tr.get("therapy") if isinstance(tr, dict) else None
        if isinstance(therapy, dict):
            sens = therapy.get("sensitivity") or []
            res = therapy.get("resistance") or []
            parts = []
            if sens:
                parts.append(f"sensitivity={sens}")
            if res:
                parts.append(f"resistance={res}")
            return " | ".join(parts)
    except Exception:
        pass
    return ""


def echo_llm_call(row: dict, completion_text: str = "") -> None:
    if not verbose():
        return
    arch = row.get("architecture") or "llm"
    role = row.get("agent_role") or row.get("label") or "agent"
    gene = row.get("gene") or ""
    mutation = row.get("mutation") or ""
    case = f"{gene} {mutation}".strip()
    prefix = f"[{arch}]"
    if case:
        prefix += f" {case} |"
    prefix += f" {role}"

    thinking = _extract_thinking(completion_text)
    if thinking and demo_thinking():
        log(
            arch,
            f"THINKING (reasoning_tokens={row.get('reasoning_tokens', 0)}, preview):",
        )
        print(f"  {_truncate(thinking, 500)}", flush=True)

    therapy_hint = _extract_json_therapy(completion_text)
    if therapy_hint:
        log(arch, f"CONCLUSION | {therapy_hint}")

    log(
        arch,
        "METRICS",
        ingress=row.get("ingress_tokens"),
        egress=row.get("egress_tokens"),
        reasoning=row.get("reasoning_tokens"),
        total=row.get("total_tokens"),
        latency_s=row.get("latency_s"),
        multimodal=row.get("multimodal_image", False),
    )


def echo_track_row(row: dict) -> None:
    if not verbose():
        return
    log(
        row.get("label", "track"),
        "done",
        cpu_s=row.get("cpu_time_s"),
        gpu_s=row.get("gpu_active_s"),
        vram=row.get("peak_vram_gib"),
        tokens=row.get("total_tokens"),
    )


def echo_self_correction(event: dict) -> None:
    if not verbose():
        return
    log(
        "reflexion",
        "mechanism self-correction",
        before=event.get("rubric_before"),
        after=event.get("rubric_after"),
        gene=event.get("gene"),
        mutation=event.get("mutation"),
    )


def echo_blackboard_step(
    gene: str,
    mutation: str,
    rnd: int,
    max_rounds: int,
    agent: str,
    content: str,
    *,
    early_exit: bool = False,
) -> None:
    if not verbose():
        return
    preview = _truncate(content, 200)
    log("blackboard", f"{gene} {mutation} | round {rnd}/{max_rounds} | {agent}", stance=preview)
    if early_exit:
        log("blackboard", "early_exit=true — skipping remaining rounds")


def echo_rescue(msg: str, **kv: Any) -> None:
    log("rescue", msg, **kv)
