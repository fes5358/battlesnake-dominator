"""
status_page.py — Renders the /status dashboard as a self-contained HTML page.

No external JS/CSS dependencies (no CDN) — the line chart is drawn as an
inline SVG polyline so the page works even if the container has no outbound
access to third-party asset hosts.
"""

import html as html_escape
from typing import List, Optional


def _svg_chart(values: List[float], width: int = 640, height: int = 200,
               color: str = "#22c55e", label: str = "") -> str:
    if not values:
        return f'<p class="empty">No history yet for {html_escape.escape(label)}.</p>'

    pad = 24
    n = len(values)
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1

    def x(i):
        if n == 1:
            return pad
        return pad + (i / (n - 1)) * (width - 2 * pad)

    def y(v):
        return height - pad - ((v - lo) / (hi - lo)) * (height - 2 * pad)

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3" fill="{color}"><title>{v}</title></circle>'
        for i, v in enumerate(values)
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" class="chart">
      <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333" stroke-width="1"/>
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333" stroke-width="1"/>
      <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>
      {dots}
      <text x="{pad}" y="{pad-8}" fill="#888" font-size="11">{hi:.1f}</text>
      <text x="{pad}" y="{height-pad+16}" fill="#888" font-size="11">{lo:.1f}</text>
    </svg>
    """


def render_status_page(rating: Optional[dict], history: List[dict]) -> str:
    rating_html = "<p class='empty'>Could not fetch leaderboard rating.</p>"
    if rating:
        rating_html = f"""
        <div class="stat-grid">
          <div class="stat"><div class="stat-value">{rating['rating']:.2f}</div><div class="stat-label">Rating</div></div>
          <div class="stat"><div class="stat-value">#{rating['rank']}</div><div class="stat-label">Rank</div></div>
        </div>
        """

    death_turns = [h.get("avg_death_turn", 0) for h in history]
    chart_html = _svg_chart(death_turns, label="avg death turn")

    latest_constants = history[-1]["constants"] if history else {}
    constants_rows = "".join(
        f"<tr><td>{html_escape.escape(k)}</td><td>{v}</td></tr>"
        for k, v in latest_constants.items()
    )

    tuning_rows = []
    for entry in reversed(history[-30:]):
        applied = entry.get("applied") or []
        if not applied:
            continue
        changes = "; ".join(f"{a['constant']}: {a['old']} → {a['new']}" for a in applied)
        ts = html_escape.escape(entry.get("timestamp", ""))
        tuning_rows.append(f"<tr><td>{ts}</td><td>{html_escape.escape(changes)}</td></tr>")
    tuning_html = (
        "<table><thead><tr><th>Timestamp</th><th>Changes</th></tr></thead>"
        f"<tbody>{''.join(tuning_rows)}</tbody></table>"
        if tuning_rows else "<p class='empty'>No constant changes applied yet.</p>"
    )

    recent_rows = []
    for entry in reversed(history[-15:]):
        ts = html_escape.escape(entry.get("timestamp", ""))
        recent_rows.append(
            f"<tr><td>{ts}</td>"
            f"<td>{entry.get('total_games_checked', 0)}</td>"
            f"<td>{entry.get('deaths_analyzed', 0)}</td>"
            f"<td>{entry.get('avg_death_turn', 0)}</td>"
            f"<td>{entry.get('avg_health_at_death', 0)}</td></tr>"
        )
    recent_html = (
        "<table><thead><tr><th>Timestamp</th><th>Games</th><th>Deaths</th>"
        "<th>Avg death turn</th><th>Avg health @ death</th></tr></thead>"
        f"<tbody>{''.join(recent_rows)}</tbody></table>"
        if recent_rows else "<p class='empty'>No analysis runs recorded yet.</p>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>The Dominator — Status</title>
<style>
  body {{ background:#0f1115; color:#e5e7eb; font-family: system-ui, sans-serif; margin:0; padding:32px; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 15px; color:#9ca3af; text-transform: uppercase; letter-spacing: .05em; margin-top: 40px; }}
  .sub {{ color:#9ca3af; margin-top:0; margin-bottom: 24px; }}
  .card {{ background:#171a21; border:1px solid #262b36; border-radius:10px; padding:20px; }}
  .stat-grid {{ display:flex; gap:32px; }}
  .stat-value {{ font-size: 32px; font-weight: 700; color:#22c55e; }}
  .stat-label {{ color:#9ca3af; font-size:13px; margin-top:4px; }}
  table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align:left; padding: 8px 10px; border-bottom: 1px solid #262b36; }}
  th {{ color:#9ca3af; font-weight:600; }}
  .empty {{ color:#6b7280; font-style: italic; }}
  .chart {{ display:block; margin-top: 8px; }}
</style>
</head>
<body>
  <h1>The Dominator</h1>
  <p class="sub">Battlesnake tournament status &amp; self-improvement dashboard</p>

  <div class="card">{rating_html}</div>

  <h2>Avg Death Turn Over Time</h2>
  <div class="card">{chart_html}</div>

  <h2>Current Constants</h2>
  <div class="card"><table><thead><tr><th>Constant</th><th>Value</th></tr></thead>
    <tbody>{constants_rows or "<tr><td colspan=2 class='empty'>No data yet.</td></tr>"}</tbody></table></div>

  <h2>Tuning History (applied changes)</h2>
  <div class="card">{tuning_html}</div>

  <h2>Recent Analysis Runs</h2>
  <div class="card">{recent_html}</div>
</body>
</html>"""
