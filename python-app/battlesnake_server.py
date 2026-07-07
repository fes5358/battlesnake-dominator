import logging
import os
import sys
import time

from flask import Flask, jsonify
from flask import request

from battlesnake_types import GameState, BaseAgent


@staticmethod
def start_server(agent: BaseAgent, port):
    if port is None:
        raise ValueError('please select your port')

    app = Flask("Battlesnake")

    @app.get("/")
    def on_info():
        # TIP: If you open your Battlesnake URL in browser you should see this data
        data = {
            "author": agent.get_author(),
            "color": agent.get_color(),
            "head": agent.get_head(),
            "tail": agent.get_tail(),
        }

        # filter None values
        data = {k: v for k, v in data.items() if v is not None}

        if 'kilab' in request.args:
            name = agent.get_name()
            data['name'] = name

        return data

    @app.post("/start")
    def on_start():
        """start is called when your Battlesnake begins a game"""
        data = request.get_json()
        game_state = GameState(**data)
        agent.start(game_state)
        print("START")
        return "ok"

    @app.post("/move")
    def on_move():
        """move is called on every turn and returns your next move"""
        start = time.time()
        data = request.get_json()
        game_state = GameState(**data)
        move = agent.move(game_state)

        print(f"MOVE: {move}, {time.time() - start}")
        return move.model_dump()

    @app.post("/end")
    def on_end():
        """end is called when your Battlesnake finishes a game"""
        data = request.get_json()
        game_state = GameState(**data)
        agent.end(game_state)
        print("END")
        return "ok"

    @app.get("/analyze")
    def on_analyze():
        """
        Analyze recent tournament games and return a self-improvement report.

        Query params:
          games=N     — number of recent games to analyze (default 30)
          apply=true  — apply recommended constant adjustments (default false)

        Example:
          GET /analyze
          GET /analyze?games=50
          GET /analyze?apply=true
        """
        sys.path.insert(0, os.path.dirname(__file__))
        from self_improve import TournamentAnalyzer

        n_games      = request.args.get("games", 30, type=int)
        apply_changes = request.args.get("apply", "false").lower() == "true"

        try:
            analyzer = TournamentAnalyzer()
            report   = analyzer.run_analysis(games=n_games)

            applied = []
            if apply_changes:
                adjs    = report.get("recommended_adjustments", [])
                applied = analyzer.apply_adjustments(adjs)
                report["applied_adjustments"] = [
                    {"constant": a["constant"], "old": a["current"], "new": a["new"]}
                    for a in applied
                ]
                if applied:
                    report["note"] = (
                        "Constants updated. Restart the Python App workflow "
                        "to activate changes."
                    )
            analyzer.record_history(report, applied)
            return jsonify(report)

        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/analyze/trigger")
    def on_analyze_trigger():
        """
        Manually trigger one full auto-improvement cycle right now
        (analyze -> apply -> commit to GitHub -> restart if changes were made).
        Runs synchronously and will not return if a restart is triggered.
        """
        sys.path.insert(0, os.path.dirname(__file__))
        from auto_improve import run_once

        try:
            result = run_once()
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/status")
    def on_status():
        """
        Human-readable dashboard: current leaderboard rating/rank, current
        tuning constants, and a chart of avg-death-turn improvement over time
        drawn from tuning_history.json.
        """
        sys.path.insert(0, os.path.dirname(__file__))
        from self_improve import TournamentAnalyzer, load_history
        from status_page import render_status_page

        try:
            analyzer = TournamentAnalyzer()
            rating   = analyzer.get_leaderboard_rating()
        except Exception:
            rating = None

        history = load_history()
        html = render_status_page(rating, history)
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.get("/compare")
    def on_compare():
        """
        Side-by-side before/after comparison of two game windows.
        Fetches window*2 recent games, splits in half (newest = current,
        older = prior), analyzes each, and returns a JSON + HTML report.

        Query params:
          window=N   — games per window (default 25)
          format=json — return raw JSON instead of HTML
        """
        sys.path.insert(0, os.path.dirname(__file__))
        from self_improve import TournamentAnalyzer

        window  = request.args.get("window", 25, type=int)
        as_json = request.args.get("format", "html").lower() == "json"

        try:
            analyzer = TournamentAnalyzer()
            all_results = analyzer.get_recent_game_results(window * 2)
            if not all_results:
                return jsonify({"error": "No game results found"}), 500

            current_results = all_results[:window]
            prior_results   = all_results[window:]

            current_ids = [r["game_id"] for r in current_results]
            prior_ids   = [r["game_id"] for r in prior_results]

            def placement_stats(results):
                counts = {1: 0, 2: 0, 3: 0, 4: 0}
                for r in results:
                    p = r.get("placement", 0)
                    if p in counts:
                        counts[p] += 1
                total = max(1, len(results))
                return {
                    "1st_pct": round(counts[1] / total * 100, 1),
                    "2nd_pct": round(counts[2] / total * 100, 1),
                    "3rd_pct": round(counts[3] / total * 100, 1),
                    "4th_pct": round(counts[4] / total * 100, 1),
                    "podium_pct": round((counts[1] + counts[2]) / total * 100, 1),
                    "avg_rating_change": round(
                        sum(r.get("rating_change", 0) for r in results) / total, 3
                    ),
                    "counts": counts,
                }

            log.info(f"/compare: analyzing {len(prior_ids)} prior + {len(current_ids)} current games")
            prior_death   = analyzer.analyze_game_list(prior_ids,   label="prior")
            current_death = analyzer.analyze_game_list(current_ids, label="current")

            prior_place   = placement_stats(prior_results)
            current_place = placement_stats(current_results)

            def delta(cur, pri, key):
                c, p = cur.get(key, 0), pri.get(key, 0)
                return round(c - p, 3)

            report = {
                "window":  window,
                "prior": {
                    "games":      prior_results[0]["timestamp"] if prior_results else "",
                    "to":         prior_results[-1]["timestamp"] if prior_results else "",
                    "placements": prior_place,
                    "deaths":     prior_death,
                },
                "current": {
                    "from":       current_results[-1]["timestamp"] if current_results else "",
                    "to":         current_results[0]["timestamp"] if current_results else "",
                    "placements": current_place,
                    "deaths":     current_death,
                },
                "delta": {
                    "podium_pct":     delta(current_place, prior_place, "podium_pct"),
                    "1st_pct":        delta(current_place, prior_place, "1st_pct"),
                    "avg_death_turn": delta(current_death, prior_death, "avg_death_turn"),
                    "avg_health":     delta(current_death, prior_death, "avg_health"),
                    "avg_rating":     delta(current_place, prior_place, "avg_rating_change"),
                },
            }

            if as_json:
                return jsonify(report)

            # ── HTML rendering ─────────────────────────────────────────────
            import html as he

            def sign(v):
                return f"+{v}" if v > 0 else str(v)

            def delta_color(v, positive_good=True):
                if v == 0:
                    return "#9ca3af"
                good = (v > 0) == positive_good
                return "#22c55e" if good else "#ef4444"

            def cause_row(cause, cur_d, pri_d):
                c = cur_d.get("death_causes", {}).get(cause, {})
                p = pri_d.get("death_causes", {}).get(cause, {})
                c_pct = c.get("pct", 0.0)
                p_pct = p.get("pct", 0.0)
                diff  = round(c_pct - p_pct, 1)
                col   = delta_color(diff, positive_good=False)
                bar_c = f'<div style="background:#22c55e;height:8px;border-radius:4px;width:{min(c_pct,100):.0f}%"></div>'
                bar_p = f'<div style="background:#6b7280;height:8px;border-radius:4px;width:{min(p_pct,100):.0f}%"></div>'
                return (
                    f"<tr>"
                    f"<td><code>{he.escape(cause)}</code></td>"
                    f"<td>{p_pct:.1f}%{bar_p}</td>"
                    f"<td>{c_pct:.1f}%{bar_c}</td>"
                    f'<td style="color:{col};font-weight:600">{sign(diff)}%</td>'
                    f"</tr>"
                )

            all_causes = sorted(
                set(prior_death.get("death_causes", {}).keys()) |
                set(current_death.get("death_causes", {}).keys())
            )

            d = report["delta"]
            summary_cards = "".join(
                f'<div class="card stat">'
                f'<div class="stat-val" style="color:{delta_color(v, pg)}">{sign(round(v,1))}</div>'
                f'<div class="stat-lbl">{lbl}</div>'
                f'</div>'
                for lbl, v, pg in [
                    ("Podium % Δ",      d["podium_pct"],      True),
                    ("1st % Δ",         d["1st_pct"],         True),
                    ("Avg death turn Δ",d["avg_death_turn"],  True),
                    ("Avg health Δ",    d["avg_health"],      False),
                    ("Avg rating Δ",    d["avg_rating"],      True),
                ]
            )

            def place_row(label, pri_pct, cur_pct, pg=True):
                diff = round(cur_pct - pri_pct, 1)
                col  = delta_color(diff, pg)
                return (
                    f"<tr><td>{he.escape(label)}</td>"
                    f"<td>{pri_pct:.1f}%</td>"
                    f"<td>{cur_pct:.1f}%</td>"
                    f'<td style="color:{col};font-weight:600">{sign(diff)}%</td></tr>'
                )

            pp, cp = prior_place, current_place

            html_body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>The Dominator — Before/After Compare</title>
<style>
  body{{background:#0f1115;color:#e5e7eb;font-family:system-ui,sans-serif;margin:0;padding:32px;}}
  h1{{font-size:22px;margin-bottom:4px;}}h2{{font-size:14px;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;margin-top:36px;}}
  .sub{{color:#9ca3af;margin-top:0;margin-bottom:24px;}}
  .card{{background:#171a21;border:1px solid #262b36;border-radius:10px;padding:20px;margin-bottom:16px;}}
  .summary{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px;}}
  .stat{{flex:1;min-width:120px;text-align:center;}}
  .stat-val{{font-size:28px;font-weight:700;}}
  .stat-lbl{{color:#9ca3af;font-size:12px;margin-top:4px;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #262b36;}}
  th{{color:#9ca3af;font-weight:600;}}
  .win-rate{{font-size:18px;font-weight:700;color:#22c55e;}}
  .label-prior{{color:#6b7280;font-size:11px;}}
  .label-cur{{color:#22c55e;font-size:11px;}}
</style></head>
<body>
<h1>The Dominator — Before/After Compare</h1>
<p class="sub">Each window = {window} games &nbsp;·&nbsp;
  <span class="label-prior">■ Prior</span> &nbsp;
  <span class="label-cur">■ Current (newest)</span>
</p>

<h2>Δ Summary</h2>
<div class="summary">{summary_cards}</div>

<h2>Placement Breakdown</h2>
<div class="card">
<table>
  <thead><tr><th>Placement</th><th>Prior</th><th>Current</th><th>Δ</th></tr></thead>
  <tbody>
    {place_row("🥇 1st place",  pp["1st_pct"], cp["1st_pct"], True)}
    {place_row("🥈 2nd place",  pp["2nd_pct"], cp["2nd_pct"], True)}
    {place_row("3rd place",     pp["3rd_pct"], cp["3rd_pct"], False)}
    {place_row("4th place",     pp["4th_pct"], cp["4th_pct"], False)}
    {place_row("🏆 Podium (1st+2nd)", pp["podium_pct"], cp["podium_pct"], True)}
  </tbody>
</table>
<p style="font-size:12px;color:#6b7280;margin-top:8px;">
  Avg rating change: prior {pp['avg_rating_change']:+.3f} → current {cp['avg_rating_change']:+.3f}
</p>
</div>

<h2>Death Cause Breakdown</h2>
<div class="card">
<table>
  <thead><tr><th>Cause</th><th>Prior</th><th>Current</th><th>Δ</th></tr></thead>
  <tbody>
    {"".join(cause_row(c, current_death, prior_death) for c in all_causes)}
  </tbody>
</table>
<p style="font-size:12px;color:#6b7280;margin-top:8px;">
  Avg death turn: prior {prior_death['avg_death_turn']} → current {current_death['avg_death_turn']} &nbsp;|&nbsp;
  Avg health @ death: prior {prior_death['avg_health']} → current {current_death['avg_health']}
</p>
</div>

<h2>Windows</h2>
<div class="card" style="font-size:13px;color:#9ca3af;">
  <b>Prior:</b> {he.escape(str(prior_results[-1]['timestamp'] if prior_results else ''))}
  → {he.escape(str(prior_results[0]['timestamp'] if prior_results else ''))}
  ({len(prior_ids)} games)<br>
  <b>Current:</b> {he.escape(str(current_results[-1]['timestamp'] if current_results else ''))}
  → {he.escape(str(current_results[0]['timestamp'] if current_results else ''))}
  ({len(current_ids)} games)
</div>
<p style="font-size:12px;color:#4b5563;">
  <a href="/compare?window={window}&format=json" style="color:#4b5563;">JSON</a> &nbsp;·&nbsp;
  <a href="/compare?window=10" style="color:#4b5563;">window=10</a> &nbsp;·&nbsp;
  <a href="/compare?window=25" style="color:#4b5563;">window=25</a> &nbsp;·&nbsp;
  <a href="/compare?window=50" style="color:#4b5563;">window=50</a>
</p>
</body></html>"""

            return html_body, 200, {"Content-Type": "text/html; charset=utf-8"}

        except Exception as exc:
            log.exception("/compare failed")
            return jsonify({"error": str(exc)}), 500

    @app.get("/status.json")
    def on_status_json():
        """Raw JSON version of the /status dashboard data."""
        sys.path.insert(0, os.path.dirname(__file__))
        from self_improve import TournamentAnalyzer, load_history

        try:
            analyzer = TournamentAnalyzer()
            rating   = analyzer.get_leaderboard_rating()
        except Exception:
            rating = None

        return jsonify({"rating": rating, "history": load_history()})

    host = "0.0.0.0"

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    sys.path.insert(0, os.path.dirname(__file__))
    from auto_improve import start_scheduler
    start_scheduler()

    from keep_alive import start_keep_alive
    start_keep_alive()

    print(f"\nRunning Battlesnake at http://{host}:{port}")
    app.run(host=host, port=port)
