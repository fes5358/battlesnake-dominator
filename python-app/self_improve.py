#!/usr/bin/env python3
"""
self_improve.py  —  Analyze recent tournament games and auto-tune The Dominator.

Tournament:  https://www.tnt.uni-hannover.de/bs-blackout-2026
Our snake:   name="14", page=/snake/14
Replay API:  GET /api/replay/{game_id}

Usage
-----
  python self_improve.py                   # dry run: analysis only, no file changes
  python self_improve.py --apply           # apply recommended constant adjustments
  python self_improve.py --games 50        # analyze last 50 games (default 30)
  python self_improve.py --apply --json    # JSON output + apply

How it works
------------
1. Scrape /snake/14 HTML to get recent game IDs and results.
2. For every loss, fetch /api/replay/{id} and determine the death cause.
3. Categorize deaths: invisible_body, wall_collision, body_collision,
   head_collision, starvation, other.
4. Apply tuning rules: if a death category exceeds a threshold, adjust the
   corresponding constant in dominator_agent.py (bounded, ±1 step per run).
5. Also apply relaxation rules: if a category drops well below its threshold,
   slightly ease off the penalty so the snake doesn't over-correct.
6. Write changes to dominator_agent.py (--apply) and log everything.
7. Restart the Python App workflow to pick up new constants (--apply only).
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

# ── Config ─────────────────────────────────────────────────────────────────
TOURNAMENT_BASE = "https://www.tnt.uni-hannover.de/bs-blackout-2026"
SNAKE_PAGE_ID   = 14          # numeric ID on the tournament site
AGENT_FILE      = os.path.join(os.path.dirname(__file__), "dominator_agent.py")
LOG_FILE        = os.path.join(os.path.dirname(__file__), "self_improve.log")

# ── Tuning rules ────────────────────────────────────────────────────────────
# Each rule: (death_cause, pct_threshold, const_name, delta, lo, hi)
#   Applied when: deaths[cause] / total_deaths  >=  pct_threshold
TUNING_RULES: List[Tuple] = [
    # cause             threshold  constant                  delta    lo     hi
    ("invisible_body",  0.30,      "BLACKOUT_BORDER_COST",  +1.0,   2.0,  15.0),
    ("wall_collision",  0.25,      "WALL_EDGE_COST",        +0.5,   1.5,   8.0),
    ("head_collision",  0.35,      "HUNT_HEALTH_THRESHOLD", +5.0,  20.0,  70.0),
    ("starvation",      0.25,      "URGENCY_FULL",          -5.0,  30.0,  80.0),
]

# Relaxation rules — slightly ease off when a category becomes very rare
RELAX_RULES: List[Tuple] = [
    # cause             threshold  constant                  delta    lo     hi
    ("invisible_body",  0.05,      "BLACKOUT_BORDER_COST",  -0.5,   2.0,  15.0),
    ("wall_collision",  0.05,      "WALL_EDGE_COST",        -0.25,  1.5,   8.0),
]

# Names of all constants we can read / report from dominator_agent.py
READABLE_CONSTANTS = [
    "BLACKOUT_BORDER_COST",
    "WALL_EDGE_COST",
    "WALL_NEAR_COST",
    "WALL_BUFFER_COST",
    "GHOST_MAX_AGE",
    "HUNT_HEALTH_THRESHOLD",
    "URGENCY_FULL",
    "URGENCY_NONE",
    "MIN_AREA_RATIO",
    "HAZARD_A_STAR_COST",
]

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────
class DeathAnalysis:
    """Everything we know about one loss game."""

    def __init__(
        self,
        game_id: int,
        total_turns: int,
        death_turn: int,
        cause: str,
        my_health: int,
        my_length: int,
        invisible_by: bool,
    ):
        self.game_id      = game_id
        self.total_turns  = total_turns
        self.death_turn   = death_turn
        self.cause        = cause
        self.my_health    = my_health
        self.my_length    = my_length
        self.invisible_by = invisible_by  # True → "by" snake was outside vision radius

    def categorize(self) -> str:
        if self.cause == "snake-self-collision" and self.invisible_by:
            return "invisible_body"
        if self.cause == "wall-collision":
            return "wall_collision"
        if self.cause == "head-collision":
            return "head_collision"
        if self.cause in ("snake-collision", "snake-self-collision"):
            return "body_collision"
        if self.cause in ("starvation", "hazard"):
            return "starvation"
        return "other"

    def to_dict(self) -> dict:
        return {
            "game_id":      self.game_id,
            "death_turn":   self.death_turn,
            "cause":        self.cause,
            "category":     self.categorize(),
            "my_health":    self.my_health,
            "my_length":    self.my_length,
            "invisible_by": self.invisible_by,
        }


# ── Main analyser ─────────────────────────────────────────────────────────────
class TournamentAnalyzer:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "battlesnake-self-improve/1.0"

    # ── Fetch helpers ──────────────────────────────────────────────────────

    def get_recent_game_ids(self, n: int = 30) -> List[int]:
        """Scrape /snake/{id} HTML and return the last n game IDs."""
        url  = f"{TOURNAMENT_BASE}/snake/{SNAKE_PAGE_ID}"
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        all_ids = re.findall(r'/game/(\d+)', resp.text)
        seen, unique = set(), []
        for gid in all_ids:
            if gid not in seen:
                seen.add(gid)
                unique.append(int(gid))
        return unique[:n]

    def get_replay(self, game_id: int) -> Optional[dict]:
        url = f"{TOURNAMENT_BASE}/api/replay/{game_id}"
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            log.warning(f"Failed to fetch replay {game_id}: {exc}")
        return None

    # ── Replay analysis ────────────────────────────────────────────────────

    def analyze_game(self, game_id: int) -> Optional[DeathAnalysis]:
        """
        Return a DeathAnalysis if we died in this game, or None if we survived.
        """
        replay = self.get_replay(game_id)
        if not replay:
            return None

        moves = replay.get("moves") or []
        if not moves:
            return None

        # Find our snake ID (our snake's name == str(SNAKE_PAGE_ID))
        our_name = str(SNAKE_PAGE_ID)
        our_id: Optional[str] = None
        for turn_data in moves:
            for s in (turn_data.get("snakes") or []):
                if s.get("name") == our_name:
                    our_id = s["id"]
                    break
            if our_id:
                break
        if not our_id:
            return None

        # Find our elimination event in dead_snakes
        for turn_data in moves:
            for s in (turn_data.get("dead_snakes") or []):
                if s["id"] != our_id:
                    continue
                ev         = s.get("elimination_event") or {}
                cause      = ev.get("cause", "unknown")
                by_id      = ev.get("by")
                death_turn = ev.get("turn", 0)

                # Recover our last health/length from the preceding turn snapshot
                my_health, my_length = 0, 0
                if death_turn > 0 and death_turn - 1 < len(moves):
                    prev = moves[death_turn - 1]
                    us   = next(
                        (x for x in (prev.get("snakes") or []) if x["id"] == our_id),
                        None,
                    )
                    if us:
                        my_health = us.get("health", 0)
                        my_length = us.get("length", 0) or len(us.get("body", []))

                # Was the killer snake invisible at the time of death?
                invisible_by = False
                if by_id and by_id != our_id and cause == "snake-self-collision":
                    idx = max(0, death_turn - 1)
                    if idx < len(moves):
                        vis_ids = {x["id"] for x in (moves[idx].get("snakes") or [])}
                        invisible_by = by_id not in vis_ids

                return DeathAnalysis(
                    game_id      = game_id,
                    total_turns  = replay.get("total_turns", 0),
                    death_turn   = death_turn,
                    cause        = cause,
                    my_health    = my_health,
                    my_length    = my_length,
                    invisible_by = invisible_by,
                )

        return None  # survived

    # ── Constants I/O ─────────────────────────────────────────────────────

    def read_constants(self) -> Dict[str, float]:
        with open(AGENT_FILE) as f:
            content = f.read()
        out = {}
        for name in READABLE_CONSTANTS:
            m = re.search(rf'^{name}\s*=\s*([\d.]+)', content, re.MULTILINE)
            if m:
                out[name] = float(m.group(1))
        return out

    # ── Tuning logic ──────────────────────────────────────────────────────

    def compute_adjustments(
        self,
        death_pct: Dict[str, float],
        current: Dict[str, float],
    ) -> List[dict]:
        adjustments = []

        def _check(rules, require_above: bool):
            for cause, threshold, const_name, delta, lo, hi in rules:
                cur = current.get(const_name)
                if cur is None:
                    continue
                pct = death_pct.get(cause, 0.0)
                if require_above and pct < threshold:
                    continue
                if not require_above and pct > threshold:
                    continue
                new_val = round(max(lo, min(hi, cur + delta)), 2)
                if new_val == cur:
                    continue
                direction = "↑" if delta > 0 else "↓"
                reason = (
                    f"{cause} deaths = {pct*100:.1f}% "
                    f"{'≥' if require_above else '≤'} {threshold*100:.0f}% threshold"
                )
                adjustments.append(
                    dict(constant=const_name, current=cur, new=new_val,
                         delta=delta, direction=direction, reason=reason)
                )

        _check(TUNING_RULES, require_above=True)
        _check(RELAX_RULES,  require_above=False)
        return adjustments

    def apply_adjustments(self, adjustments: List[dict]) -> List[dict]:
        """Write constant changes to dominator_agent.py. Returns applied list."""
        if not adjustments:
            return []
        with open(AGENT_FILE) as f:
            content = f.read()

        applied = []
        for adj in adjustments:
            const   = adj["constant"]
            new_val = adj["new"]
            fmt     = str(int(new_val)) if new_val == int(new_val) else f"{new_val:.1f}"
            pattern = rf'^({re.escape(const)}\s*=\s*)[\d.]+(\s*(#.*)?)$'
            new_content, count = re.subn(pattern, rf'\g<1>{fmt}\g<2>', content, flags=re.MULTILINE)
            if count:
                content = new_content
                applied.append(adj)
                log.info(f"  {adj['direction']} {const}: {adj['current']} → {new_val}  ({adj['reason']})")

        if applied:
            with open(AGENT_FILE, "w") as f:
                f.write(content)
            log.info(f"Wrote {len(applied)} change(s) to {AGENT_FILE}")

        return applied

    # ── Full analysis run ─────────────────────────────────────────────────

    def run_analysis(self, games: int = 30) -> dict:
        log.info(f"Fetching last {games} game IDs for snake {SNAKE_PAGE_ID}...")
        game_ids = self.get_recent_game_ids(games)
        if not game_ids:
            return {"error": "Could not fetch game list from tournament page"}

        log.info(f"Found {len(game_ids)} game IDs. Fetching replays...")
        analyses: List[DeathAnalysis] = []
        survived = 0
        failed   = 0

        for i, gid in enumerate(game_ids):
            result = self.analyze_game(gid)
            if result is None:
                # Could be a win (survived) or a failed fetch — count both as survived
                survived += 1
            else:
                analyses.append(result)
            if (i + 1) % 5 == 0:
                log.info(f"  Progress: {i+1}/{len(game_ids)}")
            time.sleep(0.15)  # polite rate-limiting

        total_games  = len(game_ids)
        total_deaths = len(analyses)
        win_count    = survived  # approximate — includes failed fetches

        cats    = Counter(a.categorize() for a in analyses)
        total_d = max(1, total_deaths)
        pct     = {k: v / total_d for k, v in cats.items()}

        avg_turn   = sum(a.death_turn for a in analyses) / total_d if analyses else 0
        avg_health = sum(a.my_health  for a in analyses) / total_d if analyses else 0

        current_consts = self.read_constants()
        adjustments    = self.compute_adjustments(pct, current_consts)

        report = {
            "timestamp":             datetime.now().isoformat(),
            "total_games_checked":   total_games,
            "deaths_analyzed":       total_deaths,
            "survived_or_no_replay": survived,
            "win_rate_approx":       round(survived / max(1, total_games), 3),
            "avg_death_turn":        round(avg_turn,   1),
            "avg_health_at_death":   round(avg_health, 1),
            "death_causes": {
                k: {"count": cats[k], "pct": round(v * 100, 1)}
                for k, v in sorted(pct.items(), key=lambda x: -x[1])
            },
            "individual_deaths":     [a.to_dict() for a in analyses[:20]],
            "current_constants":     current_consts,
            "recommended_adjustments": adjustments,
        }

        log.info(
            f"Analysis complete: {total_deaths} deaths in {total_games} games. "
            f"Recommendations: {len(adjustments)}"
        )
        return report


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> dict:
    parser = argparse.ArgumentParser(
        description="Analyze tournament games and auto-tune The Dominator"
    )
    parser.add_argument(
        "--games", type=int, default=30,
        help="Number of recent games to analyze (default: 30)"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply recommended constant adjustments to dominator_agent.py"
    )
    parser.add_argument(
        "--json", dest="json_out", action="store_true",
        help="Print full report as JSON"
    )
    args = parser.parse_args()

    analyzer = TournamentAnalyzer()

    log.info("=" * 60)
    log.info(f"Self-Improvement Run  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    report = analyzer.run_analysis(games=args.games)

    if args.json_out:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)

    if args.apply:
        adjs = report.get("recommended_adjustments", [])
        if adjs:
            log.info(f"Applying {len(adjs)} adjustment(s)...")
            applied = analyzer.apply_adjustments(adjs)
            report["applied_adjustments"] = applied
            if applied:
                print(f"\n✓ Applied {len(applied)} change(s) to dominator_agent.py")
                print("  Restart the Python App workflow to activate new constants.")
        else:
            print("\nNo adjustments to apply — constants look well-tuned.")

    log.info("Done.")
    return report


def _print_report(report: dict):
    print(f"\n{'='*55}")
    print(f"  DOMINATOR SELF-IMPROVEMENT REPORT")
    print(f"  {report.get('timestamp','')}")
    print(f"{'='*55}")
    print(f"  Games checked:   {report.get('total_games_checked', 0)}")
    print(f"  Deaths analyzed: {report.get('deaths_analyzed', 0)}")
    print(f"  Win rate (approx): {report.get('win_rate_approx', 0)*100:.1f}%")
    print(f"  Avg death turn:  {report.get('avg_death_turn', 0)}")
    print(f"  Avg health @ death: {report.get('avg_health_at_death', 0)}")

    print(f"\n  Death causes:")
    for cause, info in (report.get("death_causes") or {}).items():
        bar = "█" * int(info["pct"] / 5)
        print(f"    {cause:22s} {info['count']:3d}  ({info['pct']:5.1f}%)  {bar}")

    print(f"\n  Current constants:")
    for name, val in (report.get("current_constants") or {}).items():
        print(f"    {name}: {val}")

    adjs = report.get("recommended_adjustments") or []
    if adjs:
        print(f"\n  Recommended adjustments ({len(adjs)}):")
        for a in adjs:
            print(f"    {a['direction']} {a['constant']}: {a['current']} → {a['new']}")
            print(f"       reason: {a['reason']}")
    else:
        print(f"\n  No adjustments recommended — constants look well-tuned.")
    print()


if __name__ == "__main__":
    main()
