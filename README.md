# Battlesnake Dominator

A tournament Battlesnake server powered by **The Dominator** — an advanced AI
agent built for the Blackout + Royale ruleset (limited vision, hazard damage).

## Features

- **A\* pathfinding** with wall-proximity and Blackout vision-boundary penalties
- **Ghost snake memory** — tracks opponents that vanish outside the vision
  radius and treats their last-known body as a soft/hard obstacle
- **Voronoi territory control** — flood-fill based space evaluation, cut-off
  and squeeze maneuvers against opponents
- **Minimax 2-step lookahead** to avoid unsafe head-to-head collisions
- **Survival filter** — never enters a corridor smaller than its own length
- **Self-improvement system** (`self_improve.py`) — analyzes tournament replay
  data and auto-tunes constants based on real death-cause statistics
- **Vision-based food memory** — remembers food seen outside the current view
  radius in Blackout mode

## Snake Profile

| Field  | Value       |
|--------|-------------|
| Color  | `#FF0000`   |
| Head   | villain     |
| Tail   | sharp       |
| Author | The Dominator |

## Project layout

```
python-app/
├── main.py                # Entry point (reads PORT from env)
├── battlesnake_server.py  # Flask server: /, /start, /move, /end, /analyze
├── battlesnake_types.py   # Pydantic models for the Battlesnake API
├── dominator_agent.py     # Core agent: A*, Voronoi, ghost snakes, tuning constants
├── self_improve.py        # Tournament replay analysis + auto-tuning CLI
└── requirements.txt
```

## Running locally

```bash
cd python-app
pip install -r requirements.txt
PORT=8000 python main.py
```

## Self-improvement

Analyze recent tournament games and auto-tune constants based on real
death-cause statistics:

```bash
cd python-app
python self_improve.py                # dry run report
python self_improve.py --apply        # apply recommended constant changes
python self_improve.py --games 50     # analyze the last 50 games
```

Or trigger it over HTTP while the server is running:

```
GET /analyze?games=30
GET /analyze?games=30&apply=true
```

## Deploying (Render, Railway, Fly.io)

Set the `PORT` environment variable — `main.py` reads it automatically.
Start command: `python python-app/main.py`
