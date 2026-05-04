# Battlesnake Dominator

A Battlesnake server powered by **The Dominator** agent.

## Features

- **A\* pathfinding** to nearest food
- **Vision-based food memory** — remembers food outside the view radius
- **Tail-chase fallback** when food is unreachable
- **Random move fallback** as last resort

## Snake Profile

| Field  | Value       |
|--------|-------------|
| Color  | `#FF0000`   |
| Head   | villain     |
| Tail   | sharp       |
| Author | The Dominator |

## Running locally

```bash
pip install -r requirements.txt
PORT=8000 python main.py
```

## Deploying (Render, Railway, Fly.io)

Set the `PORT` environment variable — `main.py` reads it automatically.
Start command: `python main.py`

