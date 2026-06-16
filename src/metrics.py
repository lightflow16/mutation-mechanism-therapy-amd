"""
src/metrics.py - continuous, downloadable metrics for every program call.

ALWAYS-MEASURE MANDATE (per the plan): every wrapped call/phase logs
  - CPU time  (process user+system seconds; captures multi-core work)
  - GPU time  (gpu_active_s = CUDA-event elapsed; gpu_attached_s = wall while attached)
  - peak VRAM (GiB) and mean gfx utilization (best-effort via amd-smi)
  - TOKEN TAXONOMY: ingress (input/prompt), egress (output/completion),
    reasoning (<think>...</think>) - per agent, per model, per round.

Two CSVs, both in METRICS_DIR, each row appended + fsync'd immediately so a
session kill never loses a row:
  - calls.csv   : one row per individual call  (level="call")
  - phases.csv  : one row per high-level phase (level="phase")
Nested calls roll their token counts up into the enclosing phase automatically.

Usage:
    import src.metrics as metrics
    with metrics.phase("blackboard_query", model="qwen2.5-vl-7b"):
        with metrics.track("structure_agent", agent_role="Structure",
                           model="qwen2.5-7b", round=1) as m:
            out_text = run_agent(...)
            m.add_text(prompt_tokens=in_tok, completion_text=out_text)

GPU fields are "NA" when no GPU is attached (degrade gracefully); they
auto-populate the moment a ROCm/CUDA device is visible.
"""
from __future__ import annotations

import contextlib
import contextvars
import csv
import json
import os
import re
import subprocess
import threading
import time

try:
    import torch
    _HAS_TORCH = True
except Exception:  # torch absent (pure-CPU dev box)
    _HAS_TORCH = False

try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None

METRICS_DIR = os.environ.get("METRICS_DIR", "/workspace/shared/metrics")
CALLS_CSV = os.path.join(METRICS_DIR, "calls.csv")
PHASES_CSV = os.path.join(METRICS_DIR, "phases.csv")
LLM_LOG = os.path.join(METRICS_DIR, "llm_calls.jsonl")
SYS_CSV = os.path.join(METRICS_DIR, "system_samples.csv")

FIELDS = [
    "timestamp", "call_id", "level", "label", "agent_role", "model", "round",
    "cpu_time_s", "gpu_active_s", "gpu_attached_s", "peak_vram_gib", "gfx_util_mean",
    "ingress_tokens", "egress_tokens", "reasoning_tokens", "total_tokens",
    "latency_s", "tok_per_s",
]

_LOCK = threading.Lock()
# stack of currently-open handles, so a phase can aggregate its child calls
_STACK: contextvars.ContextVar[tuple] = contextvars.ContextVar("metrics_stack", default=())

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)


def _split_completion(completion_text: str) -> tuple[str, str]:
    text = completion_text or ""
    reasoning = "\n\n".join(m.strip() for m in _THINK_RE.findall(text) if m.strip())
    output = _THINK_RE.sub("", text).strip()
    return reasoning, output or text


def _store_llm_text_enabled() -> bool:
    if os.environ.get("STORE_LLM_TEXT") == "0":
        return False
    if os.environ.get("STORE_LLM_TEXT") == "1":
        return True
    try:
        from src.config import load_config

        return bool(load_config().get("pipeline", {}).get("store_llm_text", True))
    except Exception:
        return True

_LAST_TRACK_ROW: dict | None = None


_GPU_OK: bool | None = None


def _gpu_present() -> bool:
    """True only if torch CUDA/HIP runtime actually initializes (not just is_available())."""
    global _GPU_OK
    if not _HAS_TORCH:
        return False
    if _GPU_OK is not None:
        return _GPU_OK
    try:
        if not torch.cuda.is_available():
            _GPU_OK = False
            return False
        torch.cuda.synchronize()
        _GPU_OK = True
        return True
    except Exception:
        _GPU_OK = False
        return False


def reset_gpu_probe() -> None:
    """Clear cached GPU probe (call after fixing a broken torch install)."""
    global _GPU_OK
    _GPU_OK = None


def set_metrics_dir(path: str) -> None:
    """Override the output directory at runtime (e.g. to /workspace/shared/metrics)."""
    global METRICS_DIR, CALLS_CSV, PHASES_CSV, LLM_LOG, SYS_CSV
    METRICS_DIR = path
    CALLS_CSV = os.path.join(METRICS_DIR, "calls.csv")
    PHASES_CSV = os.path.join(METRICS_DIR, "phases.csv")
    LLM_LOG = os.path.join(METRICS_DIR, "llm_calls.jsonl")
    SYS_CSV = os.path.join(METRICS_DIR, "system_samples.csv")


class _Handle:
    """Returned by track(); lets the caller attach token counts to the row."""

    def __init__(self):
        self.ingress = 0
        self.egress = 0
        self.reasoning = 0

    def set_tokens(self, ingress: int = 0, egress: int = 0, reasoning: int = 0) -> "_Handle":
        """Set token counts explicitly (when you already counted them)."""
        self.ingress += int(ingress)
        self.egress += int(egress)
        self.reasoning += int(reasoning)
        return self

    def add_text(self, prompt_tokens: int, completion_text: str,
                 completion_tokens: int | None = None) -> "_Handle":
        """Derive egress + reasoning tokens from raw completion text.

        reasoning = whitespace tokens inside <think>...</think>;
        egress    = explicit completion_tokens if given, else whitespace token count.
        """
        think = "".join(_THINK_RE.findall(completion_text or ""))
        reasoning = len(think.split())
        if completion_tokens is None:
            completion_tokens = len((completion_text or "").split())
        self.ingress += int(prompt_tokens)
        self.egress += int(completion_tokens)
        self.reasoning += int(reasoning)
        return self


class _GfxSampler(threading.Thread):
    """Best-effort background sampler of GPU gfx utilization via amd-smi.

    Robust to any amd-smi schema/availability issue -> returns 'NA' on failure.
    """

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self._halt = threading.Event()  # not _stop — shadows threading.Thread._stop()
        self._samples: list[float] = []

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["amd-smi", "metric", "-u", "--json"],
                    capture_output=True, text=True, timeout=2,
                )
                for m in re.finditer(r'"gfx"\s*:\s*\{[^}]*?"value"\s*:\s*([\d.]+)', out.stdout):
                    self._samples.append(float(m.group(1)))
                    break
                else:
                    m = re.search(r'gfx[^0-9]{0,12}([\d.]+)', out.stdout)
                    if m:
                        self._samples.append(float(m.group(1)))
            except Exception:
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2,
                    )
                    val = out.stdout.strip().split("\n")[0].strip()
                    if val.isdigit():
                        self._samples.append(float(val))
                except Exception:
                    pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)
        if not self._samples:
            return "NA"
        return round(sum(self._samples) / len(self._samples), 1)


def _cpu_seconds() -> float:
    if _PROC is not None:
        t = _PROC.cpu_times()
        return float(t.user + t.system)
    return time.process_time()


def _append(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with _LOCK, open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        try:
            os.fsync(f.fileno())  # continuous save: row survives a session kill
        except OSError:
            pass


@contextlib.contextmanager
def track(label: str, agent_role: str = "", model: str = "", round_idx=" ",
          level: str = "call"):
    """Context manager wrapping any program call. Yields a handle for token counts."""
    h = _Handle()
    call_id = f"{int(time.time() * 1000)}_{label}"
    wall0 = time.perf_counter()
    cpu0 = _cpu_seconds()

    gpu = _gpu_present()
    sampler = None
    ev0 = ev1 = None
    if gpu:
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            sampler = _GfxSampler()
            sampler.start()
        except Exception:
            gpu = False
            ev0 = ev1 = sampler = None

    token = _STACK.set(_STACK.get() + (h,))
    try:
        yield h
    finally:
        _STACK.reset(token)
        wall = time.perf_counter() - wall0
        cpu = _cpu_seconds() - cpu0

        if gpu and ev0 is not None and ev1 is not None:
            try:
                ev1.record()
                torch.cuda.synchronize()
                gpu_active = round(ev0.elapsed_time(ev1) / 1000.0, 4)
                gpu_attached = round(wall, 4)
                peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 3)
                gfx = sampler.stop() if sampler else "NA"
            except Exception:
                gpu_active = gpu_attached = peak_vram = gfx = "NA"
        else:
            gpu_active = gpu_attached = peak_vram = gfx = "NA"

        # bubble child tokens up into the enclosing phase (if any)
        parents = _STACK.get()
        if parents:
            p = parents[-1]
            p.ingress += h.ingress
            p.egress += h.egress
            p.reasoning += h.reasoning

        total = h.ingress + h.egress + h.reasoning
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "call_id": call_id, "level": level, "label": label,
            "agent_role": agent_role, "model": model, "round": round_idx,
            "cpu_time_s": round(cpu, 4),
            "gpu_active_s": gpu_active, "gpu_attached_s": gpu_attached,
            "peak_vram_gib": peak_vram, "gfx_util_mean": gfx,
            "ingress_tokens": h.ingress, "egress_tokens": h.egress,
            "reasoning_tokens": h.reasoning, "total_tokens": total,
            "latency_s": round(wall, 4),
            "tok_per_s": round(h.egress / wall, 1) if (wall > 0 and h.egress) else "NA",
        }
        _append(PHASES_CSV if level == "phase" else CALLS_CSV, row)
        global _LAST_TRACK_ROW
        _LAST_TRACK_ROW = row
        try:
            from src import progress

            if level == "call":
                progress.echo_track_row(row)
        except Exception:
            pass


@contextlib.contextmanager
def phase(label: str, agent_role: str = "", model: str = "", round_idx=" "):
    """Convenience wrapper for a high-level phase (writes to phases.csv)."""
    with track(label, agent_role=agent_role, model=model, round_idx=round_idx, level="phase") as h:
        yield h


def summary() -> dict:
    """Quick aggregate over calls.csv (for a live budget check)."""
    import json  # local import; keep module import-light
    out = {"n_calls": 0, "ingress": 0, "egress": 0, "reasoning": 0, "total": 0}
    if not os.path.exists(CALLS_CSV):
        return out
    with open(CALLS_CSV) as f:
        for r in csv.DictReader(f):
            out["n_calls"] += 1
            out["ingress"] += int(r.get("ingress_tokens", 0) or 0)
            out["egress"] += int(r.get("egress_tokens", 0) or 0)
            out["reasoning"] += int(r.get("reasoning_tokens", 0) or 0)
    out["total"] = out["ingress"] + out["egress"] + out["reasoning"]
    return out


def log_llm_call(
    agent_role: str,
    model_name: str,
    round_idx,
    prompt_tokens: int,
    completion_text: str,
    completion_tokens: int,
    latency_s: float,
    query_id: str = "",
    *,
    architecture: str = "",
    label: str = "",
    gene: str = "",
    mutation: str = "",
    weight_cache_hit: bool = False,
    multimodal_image: bool = False,
    prompt_text: str = "",
    system_prompt: str | None = None,
) -> None:
    think = "".join(_THINK_RE.findall(completion_text or ""))
    reasoning_tok = len(think.split())
    reasoning_text, output_text = _split_completion(completion_text or "")
    if not query_id and gene and mutation:
        query_id = f"{gene}_{mutation}"
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query_id": query_id,
        "gene": gene,
        "mutation": mutation,
        "architecture": architecture,
        "label": label,
        "agent_role": agent_role,
        "model": model_name,
        "round": round_idx,
        "ingress_tokens": prompt_tokens,
        "egress_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tok,
        "thinking_tokens": reasoning_tok,
        "total_tokens": prompt_tokens + completion_tokens + reasoning_tok,
        "latency_s": round(latency_s, 4),
        "tok_per_s": round(completion_tokens / latency_s, 1) if latency_s > 0 else "NA",
        "weight_cache_hit": bool(weight_cache_hit),
        "multimodal_image": bool(multimodal_image),
    }
    try:
        if _store_llm_text_enabled():
            if prompt_text:
                row["prompt_text"] = prompt_text
            if system_prompt:
                row["system_prompt"] = system_prompt
            if reasoning_text:
                row["reasoning_text"] = reasoning_text
            if output_text:
                row["output_text"] = output_text
            row["completion_text"] = completion_text or ""
    except Exception:
        pass
    if _LAST_TRACK_ROW:
        row["gpu_active_s"] = _LAST_TRACK_ROW.get("gpu_active_s")
        row["peak_vram_gib"] = _LAST_TRACK_ROW.get("peak_vram_gib")
    try:
        from src.platform import detect_platform

        row["platform_id"] = detect_platform().get("platform_id", "")
    except Exception:
        row["platform_id"] = ""
    os.makedirs(METRICS_DIR, exist_ok=True)
    with _LOCK, open(LLM_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    try:
        from src import progress

        progress.echo_llm_call(row, completion_text)
    except Exception:
        pass


def log_self_correction(
    *,
    gene: str,
    mutation: str,
    architecture: str = "blackboard",
    rubric_before: int | float = 0,
    rubric_after: int | float = 0,
    agent_role: str = "Mechanism",
    note: str = "",
) -> None:
    """Log reflexion / rubric self-correction events to llm_calls.jsonl."""
    event = {
        "event": "self_correction",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gene": gene,
        "mutation": mutation,
        "architecture": architecture,
        "agent_role": agent_role,
        "rubric_before": rubric_before,
        "rubric_after": rubric_after,
        "note": note,
    }
    os.makedirs(METRICS_DIR, exist_ok=True)
    with _LOCK, open(LLM_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    try:
        from src import progress

        progress.echo_self_correction(event)
    except Exception:
        pass


class SysSampler:
    """Background amd-smi + psutil sampler (Layer 1)."""

    SYS_FIELDS = [
        "timestamp", "label", "cpu_pct", "ram_pct", "gfx_util", "vram_gib",
        "socket_power_w", "torch_alloc_gib", "torch_peak_gib",
    ]

    def __init__(self, label: str, interval: float = 1.0):
        self.label = label
        self.interval = interval
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _sample_once(self) -> dict:
        row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": self.label}
        if _PROC is not None:
            row["cpu_pct"] = _PROC.cpu_percent()
            row["ram_pct"] = psutil.virtual_memory().percent
        else:
            row["cpu_pct"] = row["ram_pct"] = "NA"
        row["gfx_util"] = row["vram_gib"] = row["socket_power_w"] = "NA"
        try:
            out = subprocess.run(
                ["amd-smi", "metric", "-u", "--json"],
                capture_output=True, text=True, timeout=2,
            )
            m = re.search(r'"gfx"\s*:\s*\{[^}]*?"value"\s*:\s*([\d.]+)', out.stdout)
            if m:
                row["gfx_util"] = float(m.group(1))
        except Exception:
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                parts = [p.strip() for p in out.stdout.strip().split(",")]
                if parts:
                    row["gfx_util"] = float(parts[0]) if parts[0].replace(".", "").isdigit() else "NA"
                if len(parts) > 1 and parts[1].replace(".", "").isdigit():
                    row["vram_gib"] = round(float(parts[1]) / 1024.0, 3)
                if len(parts) > 2 and parts[2].replace(".", "").isdigit():
                    row["socket_power_w"] = float(parts[2])
            except Exception:
                pass
        if _gpu_present():
            row["torch_alloc_gib"] = round(torch.cuda.memory_allocated() / 1e9, 3)
            row["torch_peak_gib"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        else:
            row["torch_alloc_gib"] = row["torch_peak_gib"] = "NA"
        return row

    def _run(self):
        os.makedirs(METRICS_DIR, exist_ok=True)
        new = not os.path.exists(SYS_CSV)
        with open(SYS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.SYS_FIELDS)
            if new:
                w.writeheader()
            while not self._halt.is_set():
                w.writerow(self._sample_once())
                f.flush()
                self._halt.wait(self.interval)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        halt = getattr(self, "_halt", None)
        if halt is not None:
            halt.set()
        if self._thread:
            self._thread.join(timeout=3)
        return False


def aggregate_ablation(out_path: str | None = None) -> str:
    """Roll calls.csv + phases.csv + llm_calls.jsonl into ablation_summary.csv (Layer 3)."""
    out_path = out_path or os.path.join(METRICS_DIR, "ablation_summary.csv")
    rows = []
    if os.path.exists(PHASES_CSV):
        with open(PHASES_CSV) as f:
            for r in csv.DictReader(f):
                rows.append({
                    "label": r.get("label", ""),
                    "level": r.get("level", ""),
                    "cpu_time_s": r.get("cpu_time_s", ""),
                    "gpu_active_s": r.get("gpu_active_s", ""),
                    "gpu_attached_s": r.get("gpu_attached_s", ""),
                    "ingress_tokens": r.get("ingress_tokens", ""),
                    "egress_tokens": r.get("egress_tokens", ""),
                    "reasoning_tokens": r.get("reasoning_tokens", ""),
                    "latency_s": r.get("latency_s", ""),
                })
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        fields = ["label", "level", "cpu_time_s", "gpu_active_s", "gpu_attached_s",
                  "ingress_tokens", "egress_tokens", "reasoning_tokens", "latency_s"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out_path


if __name__ == "__main__":
    # self-test: run with no GPU, confirm rows + token roll-up land in the CSVs
    set_metrics_dir(os.environ.get("METRICS_DIR", "./metrics"))
    with phase("selftest_phase", model="demo"):
        with track("call_a", agent_role="Structure", model="demo", round_idx=1) as m:
            time.sleep(0.05)
            m.set_tokens(ingress=100, egress=20, reasoning=5)
        with track("call_b", agent_role="Therapy", model="demo", round_idx=1) as m:
            time.sleep(0.05)
            m.add_text(prompt_tokens=200, completion_text="answer <think>because reasons here</think> done")
    print("calls.csv  ->", CALLS_CSV)
    print("phases.csv ->", PHASES_CSV)
    print("summary:", summary())
