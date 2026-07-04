"""
auto_improve.py — Background scheduler that closes the self-improvement loop.

Every SELF_IMPROVE_INTERVAL_HOURS, this:
  1. Analyzes the last SELF_IMPROVE_GAMES tournament games (self_improve.py).
  2. Applies any recommended constant adjustments to dominator_agent.py.
  3. Commits the change straight to GitHub (github_push.py) so tuning history
     is tracked in version control.
  4. Restarts this process in place so the new constants take effect
     immediately, without needing a manual workflow restart.

Disabled automatically if GITHUB_PERSONAL_ACCESS_TOKEN is not configured.
"""

import logging
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

from self_improve import TournamentAnalyzer, AGENT_FILE  # noqa: E402
from github_push import push_files  # noqa: E402

log = logging.getLogger("auto_improve")

DEFAULT_INTERVAL_HOURS = float(os.environ.get("SELF_IMPROVE_INTERVAL_HOURS", "6"))
GAMES_PER_RUN = int(os.environ.get("SELF_IMPROVE_GAMES", "30"))


def run_once() -> dict:
    """Run one analyze -> apply -> push -> restart cycle. Returns a summary dict."""
    analyzer = TournamentAnalyzer()
    log.info("Auto self-improvement cycle starting...")
    report = analyzer.run_analysis(games=GAMES_PER_RUN)

    adjustments = report.get("recommended_adjustments", [])
    if not adjustments:
        analyzer.record_history(report, [])
        log.info("No adjustments recommended this cycle.")
        return {"status": "no_adjustments", "report": report}

    applied = analyzer.apply_adjustments(adjustments)
    if not applied:
        analyzer.record_history(report, [])
        log.info("Adjustments computed but none could be applied.")
        return {"status": "no_applied", "report": report}

    analyzer.record_history(report, applied)

    log.info(f"Applied {len(applied)} constant change(s). Pushing to GitHub...")
    with open(AGENT_FILE) as f:
        content = f.read()

    summary = "; ".join(f"{a['constant']} {a['current']}->{a['new']}" for a in applied)
    commit_message = (
        f"Auto-tune: {summary}\n\n"
        f"Applied automatically based on {report.get('deaths_analyzed', 0)} analyzed "
        f"deaths across {report.get('total_games_checked', 0)} tournament games."
    )

    try:
        sha = push_files(
            {"python-app/dominator_agent.py": content},
            commit_message=commit_message,
        )
        log.info(f"Pushed updated constants to GitHub (commit {sha[:7]}).")
    except Exception:
        log.exception("GitHub push failed; constants were still applied locally.")
        return {"status": "applied_push_failed", "applied": applied}

    log.info("Restarting process to activate new constants...")
    os.execv(sys.executable, [sys.executable] + sys.argv)
    return {"status": "restarting", "applied": applied}  # unreachable after execv


def _scheduler_loop(interval_hours: float):
    log.info(f"Self-improvement scheduler running every {interval_hours}h "
              f"({GAMES_PER_RUN} games/cycle).")
    while True:
        time.sleep(interval_hours * 3600)
        try:
            run_once()
        except Exception:
            log.exception("Self-improvement cycle raised an unexpected error")


def start_scheduler(interval_hours: float = None) -> bool:
    """Start the background scheduler thread. Returns True if started."""
    if not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
        log.warning(
            "GITHUB_PERSONAL_ACCESS_TOKEN not set — auto-improvement scheduler "
            "disabled (manual /analyze endpoint still works)."
        )
        return False

    interval = interval_hours if interval_hours is not None else DEFAULT_INTERVAL_HOURS
    t = threading.Thread(target=_scheduler_loop, args=(interval,), daemon=True)
    t.start()
    log.info(f"Self-improvement scheduler started (every {interval}h).")
    return True
