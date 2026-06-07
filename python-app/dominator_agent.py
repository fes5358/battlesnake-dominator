from collections import deque
from dataclasses import dataclass, field
import heapq
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point

# ── Tunable constants ─────────────────────────────────────────────────────────
HUNT_HEALTH_THRESHOLD          = 40    # Hunt only when health > this
HUNT_MAX_DISTANCE              = 8     # Max A* steps to chase a hunt target
SQUEEZE_HEALTH_THRESHOLD       = 40    # Squeeze only when health > this
SQUEEZE_FOOD_INTERRUPT_HEALTH  = 55    # Within squeeze, eat first when health ≤ this
SQUEEZE_TERRITORY_THRESHOLD    = 0.45  # Squeeze when we own > 45% in a 1v1
COIL_HEALTH_THRESHOLD          = 40    # Coil only when health > this
COIL_TERRITORY_THRESHOLD       = 0.50  # Coil when we control ≥ this share of territory
URGENCY_FULL                   = 70    # ≥ this health → full food-path penalties
URGENCY_NONE                   = 20    # ≤ this health → zero penalties (desperation)
TERRITORY_PRESSURE_LOW         = 0.30  # Below this ratio → extra food urgency applied

# Hazard avoidance
HAZARD_OBSTACLE_HEALTH         = 40    # Treat hazards as hard obstacles when health > this
HAZARD_A_STAR_COST             = 8.0   # Extra A* step cost through a hazard cell

# Minimum-area survival filter — never move to a cell with area < snake_length × this.
# This is the single most important constant: it prevents the snake from entering
# corridors it cannot escape.
MIN_AREA_RATIO                 = 1.0

# Lookahead weights — how much the minimax lookahead influences the final score.
# Higher values make the snake more conservative (prefers moves safe against
# worst-case opponent responses, even when that costs a little area).
LOOKAHEAD_AREA_WEIGHT          = 0.3   # weight of min-future-area in food cost
LOOKAHEAD_MIN_EXITS            = 2     # soft-prefer moves with ≥ this many future exits

# Territory control — flood-fill Voronoi per move + active cut-off
TERRITORY_GAIN_WEIGHT          = 0.4   # tiebreaker weight for territory gain in rank_key
CUTOFF_OPP_AREA_THRESHOLD      = 0.55  # attempt cut-off when opponent area < this × their_length
CUTOFF_MAX_DISTANCE            = 7     # max A* steps allowed to reach a cut-off target
CUTOFF_HEALTH_THRESHOLD        = 40    # only attempt cut-off when health > this
TERRITORY_TREND_WINDOW         = 5     # turns of territory ratio history to track
TERRITORY_DECLINING_THRESHOLD  = -0.04 # per-turn ratio drop → treat as extra food urgency


# ---------------------------------------------------------
# Health-aware food penalty helpers
# ---------------------------------------------------------
def food_penalty_scale(health: int, territory_ratio: float) -> float:
    health_scale = max(0.0, min(1.0, (health - URGENCY_NONE) / max(1, URGENCY_FULL - URGENCY_NONE)))
    if territory_ratio < TERRITORY_PRESSURE_LOW:
        territory_urgency = (TERRITORY_PRESSURE_LOW - territory_ratio) / TERRITORY_PRESSURE_LOW
    else:
        territory_urgency = 0.0
    return max(0.0, health_scale - territory_urgency)


def compute_food_penalties(
    health: int, area: int, snake_length: int, risky: bool, territory_ratio: float
) -> float:
    scale    = food_penalty_scale(health, territory_ratio)
    area_pen = 1.0 + 1.0 * scale if area < snake_length else 1.0
    risk_pen = 1.0 + 2.0 * scale if risky else 1.0
    return area_pen * risk_pen


def wall_proximity_penalty(x: int, y: int, width: int, height: int) -> float:
    """
    Extra A* step cost for cells near the board edge.  Makes paths through the
    open centre naturally preferred over wall-hugging routes — the primary cause
    of the early deaths seen in replay analysis (median death turn 14).
    """
    dist = min(x, width - 1 - x, y, height - 1 - y)
    if dist == 0:
        return 1.5
    if dist == 1:
        return 0.8
    if dist == 2:
        return 0.3
    return 0.0


# ---------------------------------------------------------
# Core map helpers
# ---------------------------------------------------------
def get_obstacle_map(game_state: GameState, include_hazards: bool = False) -> np.ndarray:
    """
    Build the occupancy grid.  All snake body segments except the tail are
    marked True.  When include_hazards=True the visible hazard cells are also
    marked, so A* and flood-fill treat them as hard walls.
    """
    h = game_state.board.height
    w = game_state.board.width
    obstacle_map = np.zeros((h, w), dtype=bool)

    for snake in game_state.board.snakes:
        for body_part in snake.body[:-1]:
            if body_part is None:
                continue
            if 0 <= body_part.y < h and 0 <= body_part.x < w:
                obstacle_map[body_part.y, body_part.x] = True

    if include_hazards:
        for hz in game_state.board.hazards:
            if 0 <= hz.y < h and 0 <= hz.x < w:
                obstacle_map[hz.y, hz.x] = True

    return obstacle_map


def get_hazard_grid(game_state: GameState) -> np.ndarray:
    """Boolean grid — True where a hazard exists (used for cost-weighted A*)."""
    h = game_state.board.height
    w = game_state.board.width
    grid = np.zeros((h, w), dtype=bool)
    for hz in game_state.board.hazards:
        if 0 <= hz.y < h and 0 <= hz.x < w:
            grid[hz.y, hz.x] = True
    return grid


def get_vision_mask(width: int, height: int, center: Point, radius: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    return (abs(x - center.x) + abs(y - center.y)) <= radius


def get_safe_moves(game_state: GameState, obstacle_map: np.ndarray) -> List[Direction]:
    """
    Hard filter — three inviolable rules before every other decision:
      1. Within board boundaries
      2. Not the neck cell (prevents instant self-reversal)
      3. Not an obstacle cell (any snake segment except the tail)
    """
    head = game_state.you.head
    if head is None:
        return list(Direction)

    neck: Optional[Point] = None
    body = game_state.you.body
    if len(body) > 1 and body[1] is not None:
        neck = body[1]

    w = game_state.board.width
    h = game_state.board.height
    safe = []
    for d in Direction:
        nx, ny = head.x + d.dx, head.y + d.dy
        if nx < 0 or nx >= w:                                     continue
        if ny < 0 or ny >= h:                                     continue
        if neck is not None and nx == neck.x and ny == neck.y:    continue
        if obstacle_map[ny, nx]:                                   continue
        safe.append(d)
    return safe


def flood_fill_cells(grid: np.ndarray, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """DFS from start; returns full set of reachable (row, col) cells."""
    h, w = grid.shape
    r0, c0 = start
    if not (0 <= r0 < h and 0 <= c0 < w) or grid[r0, c0]:
        return set()
    visited: Set[Tuple[int, int]] = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not grid[nr, nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return visited


def score_moves_by_area(
    safe_moves: List[Direction], obstacle_map: np.ndarray, head: Point
) -> Dict[Direction, int]:
    return {d: len(flood_fill_cells(obstacle_map, (head.y + d.dy, head.x + d.dx))) for d in safe_moves}


def filter_survival_moves(
    safe_moves: List[Direction],
    area_scores: Dict[Direction, int],
    snake_length: int,
) -> List[Direction]:
    """
    Drop moves where the flood-fill area is smaller than the snake itself.

    Without this filter, A* routes us along walls toward food (shortest path).
    After eating, the body fills the corridor and we have no escape — causing
    the self-collision and wall-collision deaths seen in every loss game.

    Falls back to all safe moves only when every direction fails the filter
    (genuine end-game / full-board situations).
    """
    min_area = max(1, int(snake_length * MIN_AREA_RATIO))
    survival = [d for d in safe_moves if area_scores.get(d, 0) >= min_area]
    return survival if survival else safe_moves


def get_head_collision_risks(game_state: GameState) -> Set[Tuple[int, int]]:
    """(x, y) cells an equal-or-larger opponent could step into next turn."""
    my_id     = game_state.you.id
    my_length = game_state.you.length or len(game_state.you.body)
    risky: Set[Tuple[int, int]] = set()
    for snake in game_state.board.snakes:
        if snake.id == my_id or snake.head is None:
            continue
        if (snake.length or len(snake.body)) < my_length:
            continue
        for d in Direction:
            nx, ny = snake.head.x + d.dx, snake.head.y + d.dy
            if 0 <= nx < game_state.board.width and 0 <= ny < game_state.board.height:
                risky.add((nx, ny))
    return risky


# ---------------------------------------------------------
# Minimax 2-step lookahead
# ---------------------------------------------------------
def _tail_not_duplicated(body, tail) -> bool:
    """Return True if the tail segment appears only once (it will slide away)."""
    if tail is None:
        return False
    return sum(1 for b in body if b and b.x == tail.x and b.y == tail.y) == 1


def lookahead_eval(
    candidate_move: Direction,
    head: Point,
    game_state: GameState,
    obstacle_map: np.ndarray,
    snake_length: int,
) -> Tuple[int, int, bool]:
    """
    Simulate one step forward for us and one step forward for each visible
    opponent (worst-case — the opponent move that hurts us the most).

    Returns:
        min_future_area  — worst-case flood-fill area from our new head after
                           the best possible opponent response
        min_exits        — worst-case number of free neighbours from our new
                           head (0 = instant death next turn, 1 = bottleneck)
        head_safe        — True iff NO equal-or-larger opponent can land on
                           our new head next turn

    These three values are used to break ties in move ranking and as a small
    discount factor in food-path costs, biasing the snake toward moves that
    remain safe regardless of what opponents do.
    """
    w, h      = game_state.board.width, game_state.board.height
    my_length = snake_length
    new_hx    = head.x + candidate_move.dx
    new_hy    = head.y + candidate_move.dy

    # ── Simulate our move ───────────────────────────────────────────────────
    sim_after_us = obstacle_map.copy()
    body = game_state.you.body
    tail = body[-1] if body else None
    # The tail slides away unless we just ate (body[-1] == body[-2])
    if tail and _tail_not_duplicated(body, tail):
        if 0 <= tail.y < h and 0 <= tail.x < w:
            sim_after_us[tail.y, tail.x] = False
    # Mark our new head so opponents "see" us there
    if 0 <= new_hy < h and 0 <= new_hx < w:
        sim_after_us[new_hy, new_hx] = True

    # ── Evaluate opponent responses ─────────────────────────────────────────
    opponents = [
        s for s in game_state.board.snakes
        if s.id != game_state.you.id and s.head is not None
    ]

    if not opponents:
        area  = len(flood_fill_cells(sim_after_us, (new_hy, new_hx)))
        exits = _count_exits(sim_after_us, new_hx, new_hy, w, h)
        return (area, exits, True)

    worst_area  = float('inf')
    worst_exits = float('inf')
    head_safe   = True

    for opp in opponents:
        opp_length = opp.length or len(opp.body)
        opp_neck   = opp.body[1] if len(opp.body) > 1 else None
        opp_tail   = opp.body[-1] if opp.body else None

        for opp_dir in Direction:
            onx = opp.head.x + opp_dir.dx
            ony = opp.head.y + opp_dir.dy
            if not (0 <= onx < w and 0 <= ony < h):
                continue
            if opp_neck and onx == opp_neck.x and ony == opp_neck.y:
                continue
            # Opponent cannot move into their own body (pre-slide map)
            if obstacle_map[ony, onx]:
                continue

            # Head-on collision check against our new position
            if onx == new_hx and ony == new_hy and opp_length >= my_length:
                head_safe = False

            # Simulate the opponent taking this step
            sim_both = sim_after_us.copy()
            sim_both[ony, onx] = True
            if opp_tail and _tail_not_duplicated(opp.body, opp_tail):
                if 0 <= opp_tail.y < h and 0 <= opp_tail.x < w:
                    sim_both[opp_tail.y, opp_tail.x] = False

            # Our area and exit count in this simulated world
            area  = len(flood_fill_cells(sim_both, (new_hy, new_hx)))
            exits = _count_exits(sim_both, new_hx, new_hy, w, h)

            if area  < worst_area:  worst_area  = area
            if exits < worst_exits: worst_exits = exits

    return (
        int(worst_area)  if worst_area  < float('inf') else 0,
        int(worst_exits) if worst_exits < float('inf') else 0,
        head_safe,
    )


def _count_exits(grid: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """Number of free neighbouring cells — 0 means instant death next turn."""
    count = 0
    for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and not grid[ny, nx]:
            count += 1
    return count


def compute_lookahead_scores(
    survival_moves: List[Direction],
    head: Point,
    game_state: GameState,
    obstacle_map: np.ndarray,
    snake_length: int,
) -> Dict[Direction, Tuple[int, int, bool]]:
    """
    Returns {direction: (min_future_area, min_exits, head_safe)} for each
    candidate move.  Runs in ~0.02 ms on a 15×15 board with 3 opponents.
    """
    return {
        d: lookahead_eval(d, head, game_state, obstacle_map, snake_length)
        for d in survival_moves
    }


# ---------------------------------------------------------
# Voronoi territory map
# ---------------------------------------------------------
CONTESTED = "__contested__"

def compute_voronoi(
    game_state: GameState,
    obstacle_map: np.ndarray,
) -> Tuple[Dict[str, int], Dict[str, Set[Tuple[int, int]]]]:
    """
    Multi-source BFS from every visible snake head simultaneously.
    Each open cell is claimed by whichever snake reaches it first.
    Equidistant cells are marked contested and belong to no one.
    """
    h, w = obstacle_map.shape
    cell_owner: Dict[Tuple[int, int], str] = {}
    cell_dist:  Dict[Tuple[int, int], int] = {}
    q: deque[Tuple[int, int]] = deque()

    for snake in game_state.board.snakes:
        if snake.head is None:
            continue
        r, c = snake.head.y, snake.head.x
        if not (0 <= r < h and 0 <= c < w) or obstacle_map[r, c]:
            continue
        pos = (r, c)
        if pos not in cell_dist:
            cell_dist[pos]  = 0
            cell_owner[pos] = snake.id
            q.append(pos)
        elif cell_owner[pos] != snake.id:
            cell_owner[pos] = CONTESTED

    while q:
        r, c = q.popleft()
        current_dist = cell_dist[(r, c)]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or obstacle_map[nr, nc]:
                continue
            new_dist = current_dist + 1
            pos = (nr, nc)
            if pos not in cell_dist:
                cell_dist[pos]  = new_dist
                cell_owner[pos] = cell_owner[(r, c)]
                q.append(pos)
            elif cell_dist[pos] == new_dist and cell_owner[pos] != cell_owner[(r, c)]:
                cell_owner[pos] = CONTESTED

    territory_counts: Dict[str, int] = {}
    territory_cells:  Dict[str, Set[Tuple[int, int]]] = {}
    for pos, owner in cell_owner.items():
        if owner == CONTESTED:
            continue
        territory_counts[owner] = territory_counts.get(owner, 0) + 1
        territory_cells.setdefault(owner, set()).add(pos)

    return territory_counts, territory_cells


# ---------------------------------------------------------
# Territory gain per move — post-move Voronoi simulation
# ---------------------------------------------------------
def _voronoi_bfs(
    grid: np.ndarray,
    seeds: List[Tuple[Tuple[int, int], str]],
) -> Dict[Tuple[int, int], str]:
    """
    Multi-source BFS Voronoi.  seeds = [(row, col), snake_id] list.
    Returns {(row,col): owner_id} for non-contested cells.
    """
    h, w = grid.shape
    cell_owner: Dict[Tuple[int, int], str] = {}
    cell_dist:  Dict[Tuple[int, int], int] = {}
    q: deque = deque()

    for (r, c), sid in seeds:
        if not (0 <= r < h and 0 <= c < w) or grid[r, c]:
            continue
        pos = (r, c)
        if pos not in cell_dist:
            cell_dist[pos]  = 0
            cell_owner[pos] = sid
            q.append(pos)
        elif cell_owner[pos] != sid:
            cell_owner[pos] = CONTESTED

    while q:
        r, c = q.popleft()
        owner = cell_owner.get((r, c), CONTESTED)
        if owner == CONTESTED:
            continue
        dist = cell_dist[(r, c)]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc]:
                continue
            new_dist = dist + 1
            pos = (nr, nc)
            if pos not in cell_dist:
                cell_dist[pos]  = new_dist
                cell_owner[pos] = owner
                q.append(pos)
            elif cell_dist[pos] == new_dist and cell_owner[pos] != owner:
                cell_owner[pos] = CONTESTED

    return cell_owner


def compute_territory_per_move(
    survival_moves: List[Direction],
    head: Point,
    game_state: GameState,
    obstacle_map: np.ndarray,
) -> Dict[Direction, int]:
    """
    For each candidate move, simulate our head at the new position and run
    a fast Voronoi BFS.  Returns how many cells we would own after each move.

    Used as a tiebreaker in move ranking (prefer moves that expand our territory)
    and to detect whether a move actively cuts into an opponent's space.
    Runs ≤4 BFS passes of O(board_size) — typically <0.5 ms total.
    """
    my_id  = game_state.you.id
    body   = game_state.you.body
    tail   = body[-1] if body else None
    h, w   = obstacle_map.shape

    # Opponent seed positions (constant across all candidate moves)
    opp_seeds = [
        ((s.head.y, s.head.x), s.id)
        for s in game_state.board.snakes
        if s.id != my_id and s.head is not None
        and 0 <= s.head.y < h and 0 <= s.head.x < w
    ]

    scores: Dict[Direction, int] = {}
    for d in survival_moves:
        new_hx = head.x + d.dx
        new_hy = head.y + d.dy

        sim = obstacle_map.copy()
        # Tail slides away unless we just ate
        if tail and _tail_not_duplicated(body, tail):
            if 0 <= tail.y < h and 0 <= tail.x < w:
                sim[tail.y, tail.x] = False
        # Our new head is free (we occupy it, not an obstacle for BFS)
        if 0 <= new_hy < h and 0 <= new_hx < w:
            sim[new_hy, new_hx] = False

        seeds = [((new_hy, new_hx), my_id)] + opp_seeds
        cell_owner = _voronoi_bfs(sim, seeds)
        scores[d] = sum(1 for v in cell_owner.values() if v == my_id)

    return scores


def territory_trend(history: List[float]) -> float:
    """
    Mean per-turn change in our Voronoi territory ratio over the last
    TERRITORY_TREND_WINDOW turns.  Negative = we are losing territory.
    Zero or positive = holding / expanding.
    """
    if len(history) < 2:
        return 0.0
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    return sum(diffs) / len(diffs)


def find_cutoff_move(
    game_state: GameState,
    obstacle_map: np.ndarray,
    hazard_grid: np.ndarray,
    head: Point,
    safe_moves: List[Direction],
    territory_counts: Dict[str, int],
    territory_cells: Dict[str, Set[Tuple[int, int]]],
    snake_length: int,
    risk_cells: Set[Tuple[int, int]],
) -> Optional[Direction]:
    """
    Active territory cut-off: if an opponent's Voronoi area is below
    CUTOFF_OPP_AREA_THRESHOLD × their snake length, they are already squeezed.
    Route toward the boundary of their territory to close off their remaining
    escape routes — the "boxing in" manoeuvre.

    Boundary cells are scored by depth (surrounded by more opponent cells =
    better cut-off position) divided by distance from our head.
    """
    my_id     = game_state.you.id
    our_cells = territory_cells.get(my_id, set())
    best_direction: Optional[Direction] = None
    best_priority  = -1.0

    for snake in game_state.board.snakes:
        if snake.id == my_id or snake.head is None:
            continue
        opp_length = snake.length or len(snake.body)
        opp_area   = territory_counts.get(snake.id, 0)
        opp_cells  = territory_cells.get(snake.id, set())

        if opp_area > CUTOFF_OPP_AREA_THRESHOLD * opp_length:
            continue  # not squeezed enough — skip
        if not opp_cells:
            continue

        # Find boundary cells: opponent cells adjacent to our territory or open space
        boundary: List[Tuple[int, int]] = []
        for r, c in opp_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb in our_cells or nb not in opp_cells:
                    boundary.append((r, c))
                    break

        if not boundary:
            boundary = list(opp_cells)  # fallback: any opponent cell

        def cell_priority(pos: Tuple[int, int]) -> float:
            r, c = pos
            # depth = how many opp-cell neighbours → prefer cutting deeper
            depth = sum(
                1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if (r + dr, c + dc) in opp_cells
            )
            dist = max(1, abs(r - head.y) + abs(c - head.x))
            return (depth + 1) / dist

        target_rc = max(boundary, key=cell_priority)
        target    = Point(x=target_rc[1], y=target_rc[0])

        direction, dist = a_star_wrapper(obstacle_map, hazard_grid, head, target)
        if direction is None or direction not in safe_moves or dist > CUTOFF_MAX_DISTANCE:
            continue
        if (head.x + direction.dx, head.y + direction.dy) in risk_cells:
            continue

        priority = cell_priority(target_rc) / max(1, dist)
        if priority > best_priority:
            best_direction = direction
            best_priority  = priority

    return best_direction


# ---------------------------------------------------------
# Hunting
# ---------------------------------------------------------
def find_hunt_move(
    game_state: GameState,
    obstacle_map: np.ndarray,
    hazard_grid: np.ndarray,
    head: Point,
    safe_moves: List[Direction],
    area_scores: Dict[Direction, int],
    lookahead_scores: Dict[Direction, Tuple[int, int, bool]],
    snake_length: int,
    risk_cells: Set[Tuple[int, int]],
    health: int,
    territory_ratio: float,
) -> Optional[Direction]:
    """Route toward a cell adjacent to a smaller opponent's head."""
    my_id = game_state.you.id
    best_direction: Optional[Direction] = None
    best_cost = float('inf')

    for snake in game_state.board.snakes:
        if snake.id == my_id or snake.head is None:
            continue
        if (snake.length or len(snake.body)) >= snake_length:
            continue

        for d in Direction:
            tx, ty = snake.head.x + d.dx, snake.head.y + d.dy
            if not (0 <= tx < game_state.board.width and 0 <= ty < game_state.board.height):
                continue
            if obstacle_map[ty, tx]:
                continue

            direction, length = a_star_wrapper(obstacle_map, hazard_grid, head, Point(x=tx, y=ty))
            if direction is None or direction not in safe_moves or length > HUNT_MAX_DISTANCE:
                continue

            area     = area_scores.get(direction, 0)
            risky    = (head.x + direction.dx, head.y + direction.dy) in risk_cells
            eff_cost = length * compute_food_penalties(health, area, snake_length, risky, territory_ratio)

            # Lookahead discount: prefer moves that keep exits open
            la = lookahead_scores.get(direction)
            if la:
                _, exits, head_safe = la
                if not head_safe:          eff_cost *= 1.5
                if exits < LOOKAHEAD_MIN_EXITS: eff_cost *= 1.2

            if eff_cost < best_cost:
                best_direction = direction
                best_cost      = eff_cost

    return best_direction


# ---------------------------------------------------------
# Late-game squeeze — contested cell priority
# ---------------------------------------------------------
def find_squeeze_move(
    game_state: GameState,
    obstacle_map: np.ndarray,
    hazard_grid: np.ndarray,
    head: Point,
    safe_moves: List[Direction],
    area_scores: Dict[Direction, int],
    risk_cells: Set[Tuple[int, int]],
    territory_cells: Dict[str, Set[Tuple[int, int]]],
    snake_length: int,
    health: int,
    territory_ratio: float,
) -> Optional[Direction]:
    """
    Route toward the highest-value opponent territory cell.
    Value = adjacent_opponent_neighbours / distance.
    """
    my_id = game_state.you.id
    opp_cells: Set[Tuple[int, int]] = set()
    for sid, cells in territory_cells.items():
        if sid != my_id:
            opp_cells |= cells
    if not opp_cells:
        return None

    def priority(pos: Tuple[int, int]) -> float:
        r, c = pos
        adj = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)] if (r+dr, c+dc) in opp_cells)
        dist = abs(r - head.y) + abs(c - head.x)
        return (adj + 1) / max(1, dist)

    target_rc = max(opp_cells, key=priority)
    target = Point(x=target_rc[1], y=target_rc[0])

    direction, _ = a_star_wrapper(obstacle_map, hazard_grid, head, target)
    if direction is None or direction not in safe_moves:
        candidates = sorted(opp_cells, key=lambda p: abs(p[0]-head.y)+abs(p[1]-head.x))
        direction = None
        for rc in candidates:
            d, _ = a_star_wrapper(obstacle_map, hazard_grid, head, Point(x=rc[1], y=rc[0]))
            if d is not None and d in safe_moves:
                direction = d
                break
        if direction is None:
            return None

    area    = area_scores.get(direction, 0)
    risky   = (head.x + direction.dx, head.y + direction.dy) in risk_cells
    penalty = compute_food_penalties(health, area, snake_length, risky, territory_ratio)
    if penalty > 4.0:
        return None
    return direction


# ---------------------------------------------------------
# Coil (territory-denial) — Voronoi-aware
# ---------------------------------------------------------
def find_coil_move(
    game_state: GameState,
    obstacle_map: np.ndarray,
    head: Point,
    safe_moves: List[Direction],
    risk_cells: Set[Tuple[int, int]],
    lookahead_scores: Dict[Direction, Tuple[int, int, bool]],
    snake_length: int,
    our_territory_cells: Set[Tuple[int, int]],
) -> Optional[Direction]:
    """
    Score each candidate by:
      1. tail_reachable  — can we still reach our own tail?
      2. not_risky       — not in equal/larger opponent's next-turn reach
      3. lookahead_safe  — lookahead head_safe flag
      4. exits_ok        — lookahead exits ≥ LOOKAHEAD_MIN_EXITS
      5. our_territory   — Voronoi-controlled cells reachable from landing
      6. total_area      — tiebreaker
    """
    tail = game_state.you.body[-1]
    best_direction: Optional[Direction] = None
    best_score: Tuple = (-1,) * 7

    for d in safe_moves:
        nx, ny = head.x + d.dx, head.y + d.dy
        reachable  = flood_fill_cells(obstacle_map, (ny, nx))
        total_area = len(reachable)
        our_reachable = len(reachable & our_territory_cells) if our_territory_cells else total_area

        tail_reachable = 1 if (tail is not None and (tail.y, tail.x) in reachable) \
                         else (1 if total_area >= snake_length else 0)
        not_risky = 0 if (nx, ny) in risk_cells else 1

        la = lookahead_scores.get(d)
        la_area, la_exits, la_head_safe = la if la else (total_area, 4, True)
        la_safe_flag = 1 if la_head_safe else 0
        la_exits_ok  = 1 if la_exits >= LOOKAHEAD_MIN_EXITS else 0

        score = (tail_reachable, not_risky, la_safe_flag, la_exits_ok,
                 our_reachable, total_area, la_area)
        if score > best_score:
            best_score     = score
            best_direction = d

    return best_direction


# ---------------------------------------------------------
# Battlesnake Agent
# ---------------------------------------------------------
@dataclass
class AgentState:
    possible_food: list[Food] = field(default_factory=list)
    territory_history: list[float] = field(default_factory=list)


class DominatorAgent(BaseAgent):
    def __init__(self):
        self.agent_states: dict[str, AgentState] = {}

    def get_name(self):   return "The Dominator"
    def get_color(self):  return "#FF0000"
    def get_author(self): return "The Dominator"
    def get_head(self):   return "villain"
    def get_tail(self):   return "sharp"

    def start(self, game_state: GameState):
        self.agent_states[game_state.game.id] = AgentState()

    def move(self, game_state: GameState) -> MoveAction:
        head = game_state.you.head
        if head is None:
            return MoveAction(move=Direction.UP)

        if game_state.game.id not in self.agent_states:
            self.agent_states[game_state.game.id] = AgentState()
        agent_state = self.agent_states[game_state.game.id]

        health        = game_state.you.health
        snake_length  = game_state.you.length or len(game_state.you.body)
        hazard_damage = game_state.game.ruleset.settings.hazardDamagePerTurn
        width         = game_state.board.width
        height        = game_state.board.height

        # ── 1. Obstacle map ──────────────────────────────────────────────────
        # Use hazard cells as hard walls when we have enough health to
        # honour them.  The A* cost penalty keeps routing around them even
        # when they're not hard walls (desperate / low-health situations).
        include_hazards = (
            hazard_damage > 0
            and health > HAZARD_OBSTACLE_HEALTH
            and len(game_state.board.hazards) > 0
        )
        obstacle_map = get_obstacle_map(game_state, include_hazards=include_hazards)
        hazard_grid  = get_hazard_grid(game_state)

        # ── 2. Safe moves — absolute hard filter ─────────────────────────────
        safe_moves = get_safe_moves(game_state, obstacle_map)
        if not safe_moves:
            # Retry without hazard blocking (no valid moves otherwise)
            obstacle_map = get_obstacle_map(game_state, include_hazards=False)
            safe_moves   = get_safe_moves(game_state, obstacle_map)
        if not safe_moves:
            return MoveAction(move=Direction.UP)

        # ── 3. Flood-fill area score for each safe move ───────────────────────
        area_scores = score_moves_by_area(safe_moves, obstacle_map, head)

        # ── 4. SURVIVAL FILTER — never enter a dead-end ───────────────────────
        #
        # The #1 cause of early death in the replay data (median turn 14):
        # A* routes the snake along a wall toward food.  After eating, the body
        # fills the corridor and there is no escape.  By requiring flood-fill
        # area ≥ snake_length before any move we guarantee a viable corridor out.
        survival_moves = filter_survival_moves(safe_moves, area_scores, snake_length)

        # ── 5. Head-collision risk cells ──────────────────────────────────────
        risk_cells = get_head_collision_risks(game_state)

        def is_risky(d: Direction) -> bool:
            return (head.x + d.dx, head.y + d.dy) in risk_cells

        # ── 6. Minimax 2-step lookahead ───────────────────────────────────────
        lookahead_scores = compute_lookahead_scores(
            survival_moves, head, game_state, obstacle_map, snake_length
        )

        # ── 7. Voronoi territory map + per-move territory gain ────────────────
        #
        # Run Voronoi BFS once for the current board state (territory_counts /
        # territory_cells), then run a post-move simulation for each candidate
        # move (territory_gain_scores) so we can prefer moves that expand our
        # zone over moves that cede cells to opponents.
        territory_counts, territory_cells = compute_voronoi(game_state, obstacle_map)
        our_cells       = territory_cells.get(game_state.you.id, set())
        our_count       = territory_counts.get(game_state.you.id, 0)
        total_claimed   = sum(territory_counts.values())
        territory_ratio = our_count / max(1, total_claimed)

        territory_gain_scores = compute_territory_per_move(
            survival_moves, head, game_state, obstacle_map
        )

        # ── 7a. Territory trend — extra food urgency when being squeezed ──────
        agent_state.territory_history.append(territory_ratio)
        if len(agent_state.territory_history) > TERRITORY_TREND_WINDOW:
            agent_state.territory_history = agent_state.territory_history[-TERRITORY_TREND_WINDOW:]
        trend = territory_trend(agent_state.territory_history)
        # Declining territory ratio → treat snake as hungrier than it is so A*
        # routes toward food more aggressively to maintain board presence.
        trend_health_penalty = max(0.0, -trend / max(0.001, abs(TERRITORY_DECLINING_THRESHOLD))) * 15.0
        effective_health = max(0, health - int(trend_health_penalty))

        # ── 8. Ranked survival moves ──────────────────────────────────────────
        #
        # Sort criteria (descending):
        #   not_risky → la_head_safe → la_exits_ok → area → la_future_area → territory_gain
        def rank_key(d: Direction) -> Tuple:
            la = lookahead_scores.get(d)
            la_area, la_exits, la_safe = la if la else (area_scores[d], 4, True)
            tg = territory_gain_scores.get(d, 0)
            return (
                0 if is_risky(d)                          else 1,
                0 if (not la or not la_safe)               else 1,
                0 if la_exits < LOOKAHEAD_MIN_EXITS        else 1,
                area_scores[d],
                la_area,
                tg,
            )

        ranked_safe_moves = sorted(survival_moves, key=rank_key, reverse=True)

        # ── 9. Food memory (Blackout limited-vision support) ──────────────────
        view_radius = game_state.game.ruleset.settings.viewRadius
        if view_radius is not None:
            vision_mask = get_vision_mask(
                width=width, height=height, center=head, radius=view_radius,
            )
            updated_food: list[Food] = []
            for food in agent_state.possible_food:
                if not vision_mask[food.y, food.x]:
                    updated_food.append(food)
            for food in game_state.board.food:
                if food not in updated_food:
                    updated_food.append(food)
            agent_state.possible_food = updated_food
        else:
            agent_state.possible_food = list(game_state.board.food)

        visible_opponents = [
            s for s in game_state.board.snakes
            if s.id != game_state.you.id and s.head is not None
        ]

        # ── 10. HUNT: intercept a visible smaller opponent ────────────────────
        if health > HUNT_HEALTH_THRESHOLD and len(survival_moves) > 1:
            hunt_dir = find_hunt_move(
                game_state=game_state, obstacle_map=obstacle_map,
                hazard_grid=hazard_grid, head=head,
                safe_moves=survival_moves, area_scores=area_scores,
                lookahead_scores=lookahead_scores,
                snake_length=snake_length, risk_cells=risk_cells,
                health=effective_health, territory_ratio=territory_ratio,
            )
            if hunt_dir is not None:
                return MoveAction(move=hunt_dir)

        # ── 11. CUT-OFF: actively box in a squeezed opponent ─────────────────
        #
        # When an opponent's Voronoi territory < CUTOFF_OPP_AREA_THRESHOLD ×
        # their snake length they are already losing the space war.  Route to
        # the boundary of their zone to close off remaining escape routes.
        # Unlike SQUEEZE (1v1 only, requires us to be winning), CUT-OFF fires
        # whenever any visible opponent is squeezed — including multi-snake games.
        if health > CUTOFF_HEALTH_THRESHOLD and len(survival_moves) > 1:
            cutoff_dir = find_cutoff_move(
                game_state=game_state, obstacle_map=obstacle_map,
                hazard_grid=hazard_grid, head=head,
                safe_moves=survival_moves,
                territory_counts=territory_counts,
                territory_cells=territory_cells,
                snake_length=snake_length, risk_cells=risk_cells,
            )
            if cutoff_dir is not None:
                return MoveAction(move=cutoff_dir)

        # ── 12. SQUEEZE: late-game boundary compression (1v1) ─────────────────
        should_squeeze = (
            len(visible_opponents) == 1
            and health > SQUEEZE_HEALTH_THRESHOLD
            and territory_ratio > SQUEEZE_TERRITORY_THRESHOLD
        )
        if should_squeeze:
            if health <= SQUEEZE_FOOD_INTERRUPT_HEALTH:
                interrupt_dir = _best_food_direction(
                    agent_state, obstacle_map, hazard_grid, head,
                    survival_moves, area_scores, lookahead_scores,
                    snake_length, risk_cells, effective_health, territory_ratio,
                    width, height,
                )
                if interrupt_dir is not None:
                    return MoveAction(move=interrupt_dir)
            else:
                squeeze_dir = find_squeeze_move(
                    game_state=game_state, obstacle_map=obstacle_map,
                    hazard_grid=hazard_grid, head=head,
                    safe_moves=survival_moves, area_scores=area_scores,
                    risk_cells=risk_cells, territory_cells=territory_cells,
                    snake_length=snake_length, health=effective_health,
                    territory_ratio=territory_ratio,
                )
                if squeeze_dir is not None:
                    return MoveAction(move=squeeze_dir)

        # ── 13. COIL: Voronoi-driven territory patrol ─────────────────────────
        should_coil = (
            health > COIL_HEALTH_THRESHOLD
            and territory_ratio >= COIL_TERRITORY_THRESHOLD
            and len(visible_opponents) > 0
        )
        if should_coil:
            coil_dir = find_coil_move(
                game_state=game_state, obstacle_map=obstacle_map, head=head,
                safe_moves=survival_moves, risk_cells=risk_cells,
                lookahead_scores=lookahead_scores,
                snake_length=snake_length, our_territory_cells=our_cells,
            )
            if coil_dir is not None:
                return MoveAction(move=coil_dir)

        # ── 14. FOOD: health + territory + lookahead-aware food seeking ────────
        #
        # Uses effective_health (= health − trend_health_penalty) so a declining
        # territory ratio makes the snake act hungrier and seek food earlier.
        result_direction = _best_food_direction(
            agent_state, obstacle_map, hazard_grid, head,
            survival_moves, area_scores, lookahead_scores,
            snake_length, risk_cells, effective_health, territory_ratio,
            width, height,
        )
        if result_direction is not None:
            return MoveAction(move=result_direction)

        # ── 14. Follow own tail ───────────────────────────────────────────────
        tail = game_state.you.body[-1]
        if tail is not None and ranked_safe_moves:
            direction, _ = a_star_wrapper(obstacle_map, hazard_grid, head, tail)
            if direction is not None and direction in survival_moves:
                if not is_risky(direction):
                    return MoveAction(move=direction)
                if is_risky(ranked_safe_moves[0]):
                    return MoveAction(move=direction)
                return MoveAction(move=ranked_safe_moves[0])

        # ── 15. Final fallback: best-ranked survival move ─────────────────────
        if ranked_safe_moves:
            return MoveAction(move=ranked_safe_moves[0])
        return MoveAction(move=safe_moves[0])

    def end(self, game_state: GameState):
        if game_state.game.id in self.agent_states:
            del self.agent_states[game_state.game.id]


# ---------------------------------------------------------
# Food seeking helper (shared by SQUEEZE interrupt + normal food)
# ---------------------------------------------------------
def _best_food_direction(
    agent_state,
    obstacle_map: np.ndarray,
    hazard_grid: np.ndarray,
    head: Point,
    survival_moves: List[Direction],
    area_scores: Dict[Direction, int],
    lookahead_scores: Dict[Direction, Tuple[int, int, bool]],
    snake_length: int,
    risk_cells: Set[Tuple[int, int]],
    health: int,
    territory_ratio: float,
    width: int,
    height: int,
) -> Optional[Direction]:
    """
    Returns the best direction to reach food, or None if no food path qualifies.

    Cost formula:
        path_length
        × food_penalty(area, risk, health)
        × (1 + 0.5 × wall_proximity_of_food)   ← discounts wall-corner food
        × lookahead_factor                        ← discounts bottleneck routes
    """
    safe_set = set(survival_moves)
    result_direction: Optional[Direction] = None
    best_cost = float('inf')

    for food in agent_state.possible_food:
        direction, length = a_star_wrapper(obstacle_map, hazard_grid, head, food)
        if direction is None or direction not in safe_set:
            continue

        area    = area_scores.get(direction, 0)
        risky   = (head.x + direction.dx, head.y + direction.dy) in risk_cells
        penalty = compute_food_penalties(
            health=health, area=area, snake_length=snake_length,
            risky=risky, territory_ratio=territory_ratio,
        )

        # Wall-proximity: food near edges is discounted (costs more to reach)
        food_wall_pen = wall_proximity_penalty(food.x, food.y, width, height)

        # Lookahead factor: routes that narrow our future options cost more
        la = lookahead_scores.get(direction)
        if la:
            la_area, la_exits, la_safe = la
            lookahead_factor = 1.0
            if not la_safe:
                lookahead_factor *= 1.4          # large danger: avoid if possible
            if la_exits < LOOKAHEAD_MIN_EXITS:
                lookahead_factor *= (1.0 + LOOKAHEAD_AREA_WEIGHT)  # bottleneck
        else:
            lookahead_factor = 1.0

        eff_cost = length * penalty * (1.0 + 0.5 * food_wall_pen) * lookahead_factor

        if eff_cost < best_cost:
            result_direction = direction
            best_cost        = eff_cost

    return result_direction


# ---------------------------------------------------------
# A* with hazard + wall-proximity costs
# ---------------------------------------------------------
def a_star_wrapper(
    grid: np.ndarray,
    hazard_grid: np.ndarray,
    start: Point,
    goal: Point,
) -> Tuple[Optional[Direction], int]:
    if start.x == goal.x and start.y == goal.y:
        return None, 0
    h, w = grid.shape
    path = a_star(grid, hazard_grid, (start.y, start.x), (goal.y, goal.x), w, h)
    if path is None or len(path) < 2:
        return None, 9_999_999
    next_pos  = path[1]
    direction = Direction.from_board_delta((next_pos[1] - start.x, next_pos[0] - start.y))
    return direction, len(path)


def a_star(
    grid: np.ndarray,
    hazard_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    board_width: int,
    board_height: int,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* with per-cell costs:
      • Hazard cells          → +HAZARD_A_STAR_COST   (routes around Royale zones)
      • Wall-adjacent cells   → +wall_proximity_penalty (avoids hugging edges)

    The wall penalty is the key fix for the replay-data deaths: food near a
    corner is now effectively further away than the same food in open board
    space, so the snake routes there instead of into a wall corridor.
    """
    h, w = grid.shape
    open_set: list[tuple[float, Tuple[int, int]]] = [(0.0, start)]
    g_score: dict[Tuple[int, int], float] = {start: 0.0}
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc]:
                continue
            neighbor: Tuple[int, int] = (nr, nc)

            step_cost = 1.0
            if hazard_grid[nr, nc]:
                step_cost += HAZARD_A_STAR_COST
            step_cost += wall_proximity_penalty(nc, nr, board_width, board_height)

            new_g = g_score[current] + step_cost
            if new_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor]   = new_g
                heuristic           = abs(nr - goal[0]) + abs(nc - goal[1])
                heapq.heappush(open_set, (new_g + heuristic, neighbor))

    return None
