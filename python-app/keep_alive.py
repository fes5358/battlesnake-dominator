"""
keep_alive.py — Prevent the Render free-tier dyno from spinning down.

Root cause found via self_improve.py --games 200 analysis:
  93.6% of recent losses were "wall-collision" at turn 2-14, with health
  still at 87-99% and the snake driving in a dead-straight line into the
  edge. That is the signature of the game engine's "keep last direction"
  fallback after a /move request times out — NOT a pathfinding bug.

Render's free plan spins the service down after ~15 minutes with no
inbound HTTP traffic. The tournament runs games in scheduled batches with
long idle gaps between them (observed ~12h gaps), so almost every batch's
first game(s) hit a cold dyno: /start and the first /move calls arrive
while the container is still booting, blow through the ruleset's 500ms
move timeout, and the engine just continues straight until it hits a wall.

Fix: run a background thread that pings our own public URL every
PING_INTERVAL_SECONDS. Regular inbound HTTP traffic resets Render's idle
timer, so the dyno never goes to sleep and is always warm when a real
game starts.
"""

import logging
import os
import threading
import time

import requests

log = logging.getLogger(__name__)

PING_INTERVAL_SECONDS = 10 * 60  # well under Render's ~15 min idle timeout
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://battlesnake-dominator.onrender.com")


def _ping_loop():
    # Small initial delay so the very first ping doesn't race the server
    # binding to its port on boot.
    time.sleep(15)
    while True:
        try:
            resp = requests.get(SELF_URL + "/", timeout=20)
            log.info(f"keep_alive: pinged {SELF_URL}/ -> {resp.status_code}")
        except Exception as exc:
            log.warning(f"keep_alive: ping failed: {exc}")
        time.sleep(PING_INTERVAL_SECONDS)


def start_keep_alive():
    """Start the self-ping background thread (idempotent per process)."""
    if os.environ.get("_KEEP_ALIVE_STARTED"):
        return
    os.environ["_KEEP_ALIVE_STARTED"] = "1"
    t = threading.Thread(target=_ping_loop, daemon=True, name="keep-alive")
    t.start()
    log.info(f"keep_alive: started, pinging {SELF_URL}/ every {PING_INTERVAL_SECONDS}s")
