"""OpenAI-compatible vLLM client with metrics instrumentation."""
from __future__ import annotations

import os
import json
import time
from typing import Any

from src import metrics


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
        metrics.log_llm_call(agent_role, model, round_idx, in_tok, text, out_tok, latency)
        return {
            "content": text,
            "metadata": {
                "model": model,
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": (usage.total_tokens if usage else in_tok + out_tok),
            },
        }
