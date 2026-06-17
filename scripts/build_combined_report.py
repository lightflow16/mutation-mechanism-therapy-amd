#!/usr/bin/env python3
"""Stack the MMT-R evaluation report + AMD run dashboard + Colab run dashboard
into a single self-contained HTML.

The evaluation report stays the master document (its dark theme + CSS drive the
page). The two run-status dashboards are embedded as isolated <iframe srcdoc>
blocks so their own styling cannot collide with the report's. A sticky top nav
lets you jump between the three stacked sections.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "thesis" / "03_evaluation_report.html"
# Full AMD ROCm-run RoR dashboard extracted from the executed notebook output
# (see scripts/extract_amd_ror_dashboard.py). This supersedes the tiny
# throughput-only metrics/local/workflow_trace_dashboard.html.
AMD = ROOT / "metrics" / "local" / "ror_workflow_dashboard_amd.html"
COLAB = ROOT / "uploads" / "metrics_bundle_colab_cuda_20260617_121238" / "workflow_trace_dashboard.html"
OUT = ROOT / "thesis" / "04_combined_report.html"


def srcdoc_escape(html: str) -> str:
    """Escape a full HTML document so it survives as an attribute value.

    The browser HTML-decodes the srcdoc attribute once before parsing it as a
    document, so we encode & first (covers existing entities) then the quote.
    """
    return html.replace("&", "&amp;").replace('"', "&quot;")


def main() -> None:
    eval_html = EVAL.read_text(encoding="utf-8")
    amd_doc = srcdoc_escape(AMD.read_text(encoding="utf-8"))
    colab_doc = srcdoc_escape(COLAB.read_text(encoding="utf-8"))

    extra_css = """
    /* ── Combined-report nav + embedded dashboards ── */
    .combined-nav {
      position: sticky;
      top: 0;
      z-index: 50;
      display: flex;
      flex-wrap: wrap;
      gap: .5rem;
      align-items: center;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: .7rem 3rem;
    }
    .combined-nav .cn-brand { font-weight: 700; font-size: .85rem; margin-right: auto; color: var(--muted); letter-spacing: .03em; }
    .combined-nav a {
      font-size: .8rem;
      font-weight: 600;
      color: var(--text);
      text-decoration: none;
      padding: .35rem .8rem;
      border: 1px solid var(--border);
      border-radius: 5px;
      background: var(--surface2);
      transition: border-color .15s, color .15s;
    }
    .combined-nav a:hover { border-color: var(--accent); color: var(--accent); }
    .embed-frame {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #0f172a;
      display: block;
    }
    .embed-note { font-size: .82rem; color: var(--muted); margin: .2rem 0 1rem; }
  </style>"""

    nav = """
<nav class="combined-nav">
  <span class="cn-brand">MMT-R · Combined Report</span>
  <a href="#report">1 · Evaluation Report</a>
  <a href="#amd-run">2 · AMD Run Status</a>
  <a href="#colab-run">3 · Colab Run Status</a>
</nav>
<span id="report"></span>
"""

    embedded = f"""
  <!-- ─────────────────────────────────────
       EMBEDDED RUN DASHBOARDS
  ──────────────────────────────────────── -->
  <section class="section" id="amd-run">
    <div class="section-head">
      <div class="num"></div>
      <h2>AMD Run Status &mdash; Return on Reasoning Dashboard</h2>
      <span class="badge badge-40">AMD / ROCm</span>
    </div>
    <p class="embed-note">Full Return-on-Reasoning + infrastructure dashboard from the AMD MI300X / ROCm run,
      captured from the executed notebook cell output (<code>/workspace/shared/metrics/workflow_trace_dashboard.html</code>).
      Rendered in an isolated frame.</p>
    <iframe class="embed-frame" id="frame-amd" title="AMD run RoR dashboard"
            srcdoc="{amd_doc}"></iframe>
  </section>

  <section class="section" id="colab-run">
    <div class="section-head">
      <div class="num"></div>
      <h2>Colab Run Status &mdash; Return on Reasoning Dashboard</h2>
      <span class="badge badge-20">COLAB / CUDA</span>
    </div>
    <p class="embed-note">Full Return-on-Reasoning + infrastructure dashboard from the Colab CUDA run
      (<code>metrics_bundle_colab_cuda_20260617_121238</code>). Rendered in an isolated frame.</p>
    <iframe class="embed-frame" id="frame-colab" title="Colab run status dashboard"
            srcdoc="{colab_doc}"></iframe>
  </section>

</main>"""

    resize_script = """
<script>
  // Auto-size same-origin srcdoc iframes to their content height.
  function sizeFrame(f) {
    try {
      var doc = f.contentDocument || f.contentWindow.document;
      var h = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight);
      f.style.height = (h + 24) + 'px';
    } catch (e) { f.style.height = '900px'; }
  }
  window.addEventListener('load', function () {
    document.querySelectorAll('iframe.embed-frame').forEach(function (f) {
      sizeFrame(f);
      f.addEventListener('load', function () { sizeFrame(f); });
    });
  });
  window.addEventListener('resize', function () {
    document.querySelectorAll('iframe.embed-frame').forEach(sizeFrame);
  });
</script>
</body>"""

    out = eval_html

    # 1) update <title>
    out = out.replace(
        "<title>MMT-R — Hackathon Evaluation Report | TCS × AMD Jun 2026 (Latest Run)</title>",
        "<title>MMT-R — Combined Report · Evaluation + AMD Run + Colab Run | TCS × AMD Jun 2026</title>",
        1,
    )

    # 2) inject extra CSS before first </style>
    out = out.replace("  </style>", extra_css, 1)

    # 3) insert sticky nav right after <body>
    out = out.replace("<body>", "<body>" + nav, 1)

    # 4) inject resize script before </body> (do this BEFORE embedding the
    #    dashboards, whose srcdoc attributes also contain a literal </body>)
    out = out.replace("</body>", resize_script, 1)

    # 5) append embedded dashboards before the real </main>
    out = out.replace("</main>", embedded, 1)

    OUT.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
