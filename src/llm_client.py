"""LLM client: vLLM when available, transformers fallback on ROCm / when servers are down."""
from __future__ import annotations

import os
import re
import time
from typing import Any

from src import metrics
from src.config import is_rocm, load_config

_TEXT_MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
_MODEL_CACHE: dict[str, tuple] = {}

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)


def split_completion(completion_text: str) -> tuple[str, str]:
    """Return (reasoning_text, output_text) from a model completion."""
    text = completion_text or ""
    reasoning = "\n\n".join(m.strip() for m in _THINK_RE.findall(text) if m.strip())
    output = _THINK_RE.sub("", text).strip()
    return reasoning, output or text.strip()


def _store_llm_text() -> bool:
    if os.environ.get("STORE_LLM_TEXT") == "0":
        return False
    if os.environ.get("STORE_LLM_TEXT") == "1":
        return True
    try:
        return bool(load_config().get("pipeline", {}).get("store_llm_text", True))
    except Exception:
        return True


def _llm_response(
    completion: str,
    *,
    prompt: str,
    system_prompt: str | None,
    model: str,
    backend: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    reasoning, output = split_completion(completion)
    return {
        "content": completion,
        "prompt": prompt,
        "system_prompt": system_prompt or "",
        "reasoning": reasoning,
        "output": output or completion,
        "metadata": {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "backend": backend,
        },
    }


def trace_step_from_response(agent: str, step_type: str, resp: dict[str, Any]) -> dict[str, Any]:
    """Build a trace step dict with prompt / reasoning / output for blackboard or debate logs."""
    step: dict[str, Any] = {
        "agent": agent,
        "type": step_type,
        "content": resp.get("content", ""),
    }
    for key in ("prompt", "system_prompt", "reasoning", "output"):
        val = resp.get(key)
        if val:
            step[key] = val
    return step


def _llm_backend() -> str:
    return os.environ.get("LLM_BACKEND", "auto").lower()


def _vllm_endpoints_up() -> bool:
    try:
        from src.serving import check_all_endpoints, vllm_import_ok

        if not vllm_import_ok():
            return False
        cfg = load_config()
        eps = cfg.get("serving", {}).get("endpoints", {})
        return all(check_all_endpoints(eps).values())
    except Exception:
        return False


def use_vllm() -> bool:
    """True when live calls should go through vLLM OpenAI endpoints."""
    backend = _llm_backend()
    if backend == "transformers":
        return False
    if backend == "vllm":
        return True
    if is_rocm():
        return False
    return _vllm_endpoints_up()


def _get_text_model(model_id: str = _TEXT_MODEL_DEFAULT) -> tuple[Any, Any, bool]:
    if model_id in _MODEL_CACHE:
        tok, model = _MODEL_CACHE[model_id]
        return tok, model, True
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
    )
    _MODEL_CACHE[model_id] = (tok, model)
    return tok, model, False


def call_transformers(
    prompt: str,
    *,
    model_id: str = _TEXT_MODEL_DEFAULT,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    agent_role: str = "",
    round_idx: int | str = "",
    label: str = "llm_call",
    query_id: str = "",
    architecture: str = "",
    gene: str = "",
    mutation: str = "",
) -> dict[str, Any]:
    with metrics.track(label, agent_role=agent_role, model=model_id, round_idx=round_idx) as m:
        tok, model, weight_cache_hit = _get_text_model(model_id)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        in_len = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        gen_kwargs: dict = {"max_new_tokens": max_tokens}
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        out = model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - t0
        gen = tok.decode(out[0][in_len:], skip_special_tokens=True)
        out_tok = out[0].shape[0] - in_len
        m.set_tokens(ingress=in_len, egress=int(out_tok))
        metrics.log_llm_call(
            agent_role,
            model_id,
            round_idx,
            in_len,
            gen,
            int(out_tok),
            latency,
            query_id=query_id,
            architecture=architecture,
            label=label,
            gene=gene,
            mutation=mutation,
            weight_cache_hit=weight_cache_hit,
            prompt_text=prompt,
            system_prompt=system_prompt,
        )
        return _llm_response(
            gen,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model_id,
            backend="transformers",
            prompt_tokens=in_len,
            completion_tokens=int(out_tok),
        )


def call_vllm(
    prompt: str,
    *,
    base_url: str,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    agent_role: str = "",
    round_idx: int | str = "",
    label: str = "llm_call",
    query_id: str = "",
    architecture: str = "",
    gene: str = "",
    mutation: str = "",
) -> dict[str, Any]:
    from openai import OpenAI

    with metrics.track(label, agent_role=agent_role, model=model, round_idx=round_idx) as m:
        client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else len(prompt.split())
        out_tok = usage.completion_tokens if usage else len(text.split())
        m.set_tokens(ingress=in_tok, egress=out_tok)
        metrics.log_llm_call(
            agent_role,
            model,
            round_idx,
            in_tok,
            text,
            out_tok,
            latency,
            query_id=query_id,
            architecture=architecture,
            label=label,
            gene=gene,
            mutation=mutation,
            prompt_text=prompt,
            system_prompt=system_prompt,
        )
        return _llm_response(
            text,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            backend="vllm",
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
        )


def call_llm(
    prompt: str,
    *,
    base_url: str = "",
    model: str = "",
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    agent_role: str = "",
    round_idx: int | str = "",
    label: str = "llm_call",
    query_id: str = "",
    architecture: str = "",
    gene: str = "",
    mutation: str = "",
) -> dict[str, Any]:
    """Route to vLLM when up; otherwise transformers (Qwen2.5-7B text model)."""
    ctx = dict(
        query_id=query_id,
        architecture=architecture,
        gene=gene,
        mutation=mutation,
    )
    if use_vllm() and base_url:
        return call_vllm(
            prompt,
            base_url=base_url,
            model=model or "qwen2.5-7b-instruct",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_role=agent_role,
            round_idx=round_idx,
            label=label,
            **ctx,
        )
    model_id = _TEXT_MODEL_DEFAULT
    if model and "3b" in model.lower():
        model_id = "Qwen/Qwen2.5-3B-Instruct"
    elif model and "7b" in model.lower() and "vl" not in model.lower():
        model_id = _TEXT_MODEL_DEFAULT
    return call_transformers(
        prompt,
        model_id=model_id,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_role=agent_role,
        round_idx=round_idx,
        label=label,
        **ctx,
    )
