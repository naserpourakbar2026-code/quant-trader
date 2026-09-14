"""Generate reports/progress.html — a static, self-contained page showing
which delivery phase the project is in and which files exist so far.

Intended for a non-technical viewer: open reports/progress.html in any
browser, no server or code execution required.

Run after each phase is committed:

    python scripts/generate_progress_report.py

PHASES below is maintained by hand (there is no reliable automatic way to
detect "phase complete" from the repo alone) — update the status for a
phase here when you finish and commit it.
"""
from __future__ import annotations

import html
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "reports" / "progress.html"


@dataclass
class Phase:
    number: int
    name: str
    status: str  # "done" | "pending"


PHASES: list[Phase] = [
    Phase(1, "Architecture + environment setup", "done"),
    Phase(2, "Data ingestion", "done"),
    Phase(3, "Data validation", "done"),
    Phase(4, "Feature engine", "done"),
    Phase(5, "Three strategies", "done"),
    Phase(6, "vectorbt research engine", "done"),
    Phase(7, "Backtrader validation engine", "done"),
    Phase(8, "Optuna optimization", "done"),
    Phase(9, "Walk-forward analysis", "done"),
    Phase(10, "Monte Carlo analysis", "done"),
    Phase(11, "Portfolio engine", "pending"),
    Phase(12, "MT5 broker adapter (code + mocked tests)", "pending"),
    Phase(13, "Generic REST/WebSocket broker adapter", "pending"),
    Phase(14, "Paper trading", "pending"),
    Phase(15, "Risk & kill switch", "pending"),
    Phase(16, "Reporting", "pending"),
    Phase(17, "Full integration tests", "pending"),
    Phase(18, "Final robustness evaluation", "pending"),
]


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def group_by_top_dir(paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in paths:
        top = path.split("/", 1)[0] if "/" in path else "(root)"
        groups.setdefault(top, []).append(path)
    return dict(sorted(groups.items()))


def render_html(phases: list[Phase], groups: dict[str, list[str]]) -> str:
    current = next((p for p in phases if p.status != "done"), None)
    done_count = sum(1 for p in phases if p.status == "done")
    total = len(phases)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    phase_rows = "\n".join(
        f'''      <tr class="{p.status}">
        <td>{p.number}</td>
        <td>{html.escape(p.name)}</td>
        <td><span class="badge {p.status}">{"Done" if p.status == "done" else "Pending"}</span></td>
      </tr>'''
        for p in phases
    )

    file_sections = "\n".join(
        f'''      <div class="group">
        <h3>{html.escape(top)}/</h3>
        <ul>
{chr(10).join(f"          <li>{html.escape(f)}</li>" for f in files)}
        </ul>
      </div>'''
        for top, files in groups.items()
    )

    current_label = (
        f"Phase {current.number} — {html.escape(current.name)}"
        if current
        else "All phases complete"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>quant-trader — progress</title>
<style>
  :root {{
    --bg: #f7f7f8; --fg: #1a1a1a; --muted: #6b7280; --card: #ffffff;
    --border: #e5e7eb; --accent: #2563eb; --done: #16a34a; --pending: #d97706;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #111318; --fg: #e5e7eb; --muted: #9ca3af; --card: #1a1d24;
      --border: #2a2e37; --accent: #60a5fa; --done: #4ade80; --pending: #fbbf24; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-top: 0; margin-bottom: 24px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; margin-bottom: 20px;
  }}
  .progress-bar {{
    height: 10px; border-radius: 6px; background: var(--border); overflow: hidden; margin: 12px 0;
  }}
  .progress-fill {{ height: 100%; background: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  tr.done td {{ color: var(--fg); }}
  tr.pending td {{ color: var(--muted); }}
  .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
  }}
  .badge.done {{ background: rgba(22,163,74,0.15); color: var(--done); }}
  .badge.pending {{ background: rgba(217,119,6,0.15); color: var(--pending); }}
  .group {{ margin-bottom: 14px; }}
  .group h3 {{ margin: 0 0 4px; font-size: 0.95rem; color: var(--accent); }}
  ul {{ margin: 0; padding-left: 20px; columns: 2; }}
  li {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85rem; }}
  footer {{ color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 24px; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>quant-trader — build progress</h1>
    <p class="subtitle">Static snapshot — open this file directly, no server or code execution needed.</p>

    <div class="card">
      <strong>Current phase:</strong> {current_label}
      <div class="progress-bar"><div class="progress-fill" style="width:{done_count / total * 100:.0f}%"></div></div>
      <div style="color:var(--muted); font-size:0.85rem;">{done_count} of {total} phases done</div>
    </div>

    <div class="card">
      <h2 style="margin-top:0;">Phases</h2>
      <table>
        <thead><tr><th>#</th><th>Phase</th><th>Status</th></tr></thead>
        <tbody>
{phase_rows}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2 style="margin-top:0;">Files created so far ({sum(len(f) for f in groups.values())})</h2>
{file_sections}
    </div>

    <footer>Generated {generated_at} by scripts/generate_progress_report.py</footer>
  </div>
</body>
</html>
"""


def main() -> None:
    files = tracked_files()
    groups = group_by_top_dir(files)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_html(PHASES, groups), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
