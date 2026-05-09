from collections import deque
from dataclasses import dataclass, field
import heapq
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point

# ── Tunable constants ────────────────────────────────────────────────────────
HUNT_HEALTH_THRESHOLD     = 50   # Hunt more often (was 40)
HUNT_MAX_DISTANCE         = 6  # Chase further (was 10)
SQUEEZE_HEALTH_THRESHOLD       = 30   # Squeeze more (was 40)
SQUEEZE_FOOD_INTERRUPT_HEALTH  = 45   # was 55
SQUEEZE_TERRITORY_THRESHOLD    = 0.42  # Squeeze earlier (was 0.45)
COIL_HEALTH_THRESHOLD     = 30    # was 40
COIL_TERRITORY_THRESHOLD  = 0.47  # Coil less, be more aggressive (was 0.50)
URGENCY_FULL              = 65    # Start food urgency earlier (was 70)
URGENCY_NONE              = 15    # was 20
TERRITORY_PRESSURE_LOW    = 0.35  # was 0.30


# ---------------------------------------------------------
# Health-aware food penalty helpers
# ---------------------------------------------------------
def food_penalty_scale(health: int, territory_ratio: float) -> float:
    """
    Returns a scale factor in [0.0, 1.0] governing how much area/risk
    penalties apply to food paths.

    Two independent drivers reduce the scale (increase urgency):
      • Health dropping below URGENCY_FULL toward URGENCY_NONE
      • Territory ratio below TERRITORY_PRESSURE_LOW (we're losing space → grow)

    The two urgencies combine additively and are clamped to [0, 1].

    Examples (health-only, no territory pressure):
      health 70 → scale 1.00  area_pen 2.0  risk_pen 3.0
      health 45 → scale 0.50  area_pen 1.5  risk_pen 2.0
      health 20 → scale 0.00  area_pen 1.0  risk_pen 1.0  (desperation)
    """
    health_scale = max(0.0, min(1.0, (health - URGENCY_NONE) / (URGENCY_FULL - URGENCY_NONE)))

    # Extra urgency when we're losing the territory war
    if territory_ratio < TERRITORY_PRESSURE_LOW:
        territory_urgency = (TERRITORY_PRESSURE_LOW - territory_ratio) / TERRITORY_PRESSURE_LOW
    else:
        territory_urgency = 0.0

    return max(0.0, health_scale - territory_urgency)


def compute_food_penalties(
    health: int, area: int, snake_length: int, risky: bool, territory_ratio: float
) -> float:
    """Combined area × risk multiplier, scaled by health + territory urgency."""
    scale    = food_penalty_scale(health, territory_ratio)
    area_pen = 1.0 + 1.0 * scale if area < snake_length else 1.0
    risk_pen = 1.0 + 2.0 * scale if risky else 1.0
    return area_pen * risk_pen


# ---------------------------------------------------------
# Core map helpers
# ---------------------------------------------------------
def get_obstacle_map(game_state: GameState) -> np.ndarray:
    obstacle_map = np.zeros((game_state.board.height, game_state.board.width), dtype=bool)
    for snake in game_state.board.snakes:
        for body_part in snake.body[:-1]:
            if body_part is None:
                continue
            obstacle_map[body_part.y, body_part.x] = True
    return obstacle_map


def get_vision_mask(width: int, height: int, center: Point, radius: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    return (abs(x - center.x) + abs(y - center.y)) <= radius


def get_safe_moves(game_state: GameState, obstacle_map: np.ndarray) -> List[Direction]:
    """
    Hard filter — applied before every other decision:
      1. Within board boundaries  (board.width / board.height)
      2. Not the neck cell        (prevents instant self-reversal)
      3. Not an obstacle cell     (any snake body)
    """
    head = game_state.you.head
    if head is None:
        return list(Direction)

    neck: Optional[Point] = None
    body = game_state.you.body
    if len(body) > 1 and body[1] is not None:
        neck = body[1]

    safe = []
    for d in Direction:
        nx, ny = head.x + d.dx, head.y + d.dy
        if nx < 0 or nx >= game_state.board.width:   continue
        if ny < 0 or ny >= game_state.board.height:  continue
        if neck is not None and nx == neck.x and ny == neck.y: continue
        if obstacle_map[ny, nx]:                      continue
        safe.append(d)
    return safe


def flood_fill_cells(grid: np.ndarray, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """BFS from start; returns full set of reachable (row, col) cells."""
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


def get_head_collision_risks(game_state: GameState) -> Set[Tuple[int, int]]:
    """(x, y) cells an equal-or-larger opponent could step into next turn."""
    my_id     = game_state.you.id
    my_length = game_state.you.length or len(game_state.you.body)
    risky: Set[Tuple[int, int]] = set()
    for snake in game_state.board.snakes:
        if snake.id == my_id or snake.head is None:
            continue
        if (snake.length or len(snake.body)) >= snake_length - 2:
            continue
        for d in Direction:
            nx, ny = snake.head.x + d.dx, snake.head.y + d.dy
            if 0 <= nx < game_state.board.width and 0 <= ny < game_state.board.height:
                risky.add((nx, ny))
    return risky


# ---------------------------------------------------------
# Voronoi territory map
# ---------------------------------------------------------
CONTESTED = "__contested__"   # sentinel for cells equidistant between snakes

def compute_voronoi(
    game_state: GameState,
    obstacle_map: np.ndarray,
) -> Tuple[Dict[str, int], Dict[str, Set[Tuple[int, int]]]]:
    """
    Multi-source BFS from every visible snake head simultaneously.
    Each open cell is assigned to whichever snake reaches it first.
    Cells equidistant between two snakes are marked as contested (neither owns them).

    Returns:
      territory_counts  — {snake_id: number_of_cells_owned}
      territory_cells   — {snake_id: set_of_(row,col)_cells_owned}

    This gives a precise per-turn picture of board control — far more accurate
    than a simple length comparison, because position and obstacles matter.
    """
    h, w = obstacle_map.shape
    cell_owner: Dict[Tuple[int, int], str] = {}   # (r,c) → snake_id or CONTESTED
    cell_dist:  Dict[Tuple[int, int], int] = {}

    q: deque[Tuple[int, int]] = deque()

    for snake in game_state.board.snakes:
        if snake.head is None:
            continue
        r, c = snake.head.y, snake.head.x
        if obstacle_map[r, c]:
            continue
        pos = (r, c)
        if pos not in cell_dist:
            cell_dist[pos]  = 0
            cell_owner[pos] = snake.id
            q.append(pos)
        else:
            # Two snake heads on the same cell → contested
            if cell_owner[pos] != snake.id:
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
            elif cell_dist[pos] == new_dist:
                # Equidistant from a different snake → contested
                if cell_owner[pos] != cell_owner[(r, c)]:
                    cell_owner[pos] = CONTESTED

    # Aggregate into counts and cell sets
    territory_counts: Dict[str, int] = {}
    territory_cells:  Dict[str, Set[Tuple[int, int]]] = {}

    for pos, owner in cell_owner.items():
        if owner == CONTESTED:
            continue
        territory_counts[owner] = territory_counts.get(owner, 0) + 1
        if owner not in territory_cells:
            territory_cells[owner] = set()
        territory_cells[owner].add(pos)

    return territory_counts, territory_cells


# ---------------------------------------------------------
# Hunting
# ---------------------------------------------------------
def find_hunt_move(
    game_state: GameState,
    obstacle_map: np.ndarray,
    head: Point,
    safe_moves: List[Direction],
    area_scores: Dict[Direction, int],
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

            direction, length = a_star_wrapper(obstacle_map, head, Point(x=tx, y=ty))
            if direction is None or direction not in safe_moves or length > HUNT_MAX_DISTANCE:
                continue
            area = area_scores[direction]
            if area < snake_length * 1.5:
                continue

            
            risky    = (head.x + direction.dx, head.y + direction.dy) in risk_cells
            eff_cost = length * compute_food_penalties(health, area, snake_length, risky, territory_ratio)

            if eff_cost < best_cost:
                best_direction = direction
                best_cost      = eff_cost

    return best_direction


# ---------------------------------------------------------
# Squeeze (late-game boundary compression)
# ---------------------------------------------------------
def find_squeeze_move(
    game_state: GameState,
    obstacle_map: np.ndarray,
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
    Late-game squeeze — active boundary compression against the last opponent.

    Strategy:
      The Voronoi map already tells us exactly which cells each snake owns.
      We collect every cell owned by the opponent, find the one closest to
      our head (Manhattan distance), and A* toward it.

      Routing into the opponent's nearest territory cell does two things:
        1. We physically move toward their space, shrinking the gap.
        2. Next turn, that cell will be even closer to us in the Voronoi BFS,
           so it flips to ours — the boundary advances by one cell per turn.

      Repeated over many turns this creates a "wall" that walks across the
      board toward the opponent, leaving them with an ever-smaller pocket
      until they can no longer survive.

    Only called when:
      • Exactly one opponent is visible (1v1 end-game)
      • We own > SQUEEZE_TERRITORY_THRESHOLD of territory
      • Health is above the starvation threshold
    """
    my_id = game_state.you.id

    # Union of all cells owned by opponents
    opp_cells: Set[Tuple[int, int]] = set()
    for sid, cells in territory_cells.items():
        if sid != my_id:
            opp_cells |= cells

    if not opp_cells:
        return None  # Opponent owns nothing visible — squeeze complete

    # ── Contested cell priority scoring ────────────────────────────────────
    #
    # Naive approach: route to the *nearest* opponent cell.
    # Better approach: route to the cell that yields the most territory per
    # unit of travel — i.e. the cell surrounded by the most other opponent
    # cells, weighted by how close it is to us.
    #
    # For each opponent cell we compute:
    #
    #   adjacency_gain  — number of its 4 orthogonal neighbours that are also
    #                     opponent-owned.  When we reach this cell and the
    #                     Voronoi boundary advances, those neighbours are the
    #                     next to flip to ours on subsequent turns.  A deep
    #                     interior cell scores higher than an isolated edge
    #                     cell, so the squeeze pushes into dense opponent
    #                     territory rather than nibbling at a thin border.
    #
    #   manhattan_dist  — steps from our head.  Shorter paths are preferred
    #                     because every turn we're not there is a turn the
    #                     opponent could use to eat or escape.
    #
    #   priority_score  — adjacency_gain / max(1, manhattan_dist)
    #                     Higher is better: many cells freed, short journey.
    #
    # We score every opponent cell and pick the highest, then A* toward it.
    #
    def priority_score(pos: Tuple[int, int]) -> float:
        r, c = pos
        adj_opp = sum(
            1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if (r + dr, c + dc) in opp_cells
        )
        dist = abs(r - head.y) + abs(c - head.x)
        return (adj_opp + 1) / max(1, dist)  # +1 so isolated cells still compete

    target_rc = max(opp_cells, key=priority_score)
    target = Point(x=target_rc[1], y=target_rc[0])

    direction, _ = a_star_wrapper(obstacle_map, head, target)
    if direction is None or direction not in safe_moves:
        # Best-score target unreachable — fall back to nearest reachable cell
        candidates = sorted(
            opp_cells,
            key=lambda pos: abs(pos[0] - head.y) + abs(pos[1] - head.x),
        )
        for fallback_rc in candidates:
            direction, _ = a_star_wrapper(obstacle_map, head, Point(x=fallback_rc[1], y=fallback_rc[0]))
            if direction is not None and direction in safe_moves:
                break
        else:
            return None

    area    = area_scores[direction]
    risky   = (head.x + direction.dx, head.y + direction.dy) in risk_cells
    penalty = compute_food_penalties(health, area, snake_length, risky, territory_ratio)

    # Reject squeeze paths that are suicidally costly (deep dead-end + danger)
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
    snake_length: int,
    our_territory_cells: Set[Tuple[int, int]],
) -> Optional[Direction]:
    """
    Territory-denial coil strategy, now Voronoi-aware.

    Scoring per candidate landing cell (priority order):
      1. tail_reachable  — can we still reach our own tail from there?
         Prevents self-entrapment while looping.
      2. not_risky       — landing cell is not in any equal/larger opponent's
         next-turn reach.
      3. our_territory   — how many of our Voronoi-controlled cells are
         reachable from the landing position?  This makes The Dominator
         patrol its own territory rather than wandering into contested space.
      4. total_area      — tiebreaker: more open cells is always better.
    """
    tail = game_state.you.body[-1]

    best_direction: Optional[Direction] = None
    best_score: Tuple[int, int, int, int] = (-1, -1, -1, -1)

    for d in safe_moves:
        nx, ny = head.x + d.dx, head.y + d.dy
        reachable  = flood_fill_cells(obstacle_map, (ny, nx))
        total_area = len(reachable)

        # How many of our Voronoi-controlled cells are reachable from here?
        our_reachable = len(reachable & our_territory_cells) if our_territory_cells else total_area

        if tail is not None:
            tail_reachable = 1 if (tail.y, tail.x) in reachable else 0
        else:
            tail_reachable = 1 if total_area >= snake_length else 0

        not_risky = 0 if (nx, ny) in risk_cells else 1
        score     = (tail_reachable, not_risky, our_reachable, total_area)

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

        health       = game_state.you.health
        snake_length = game_state.you.length or len(game_state.you.body)

        # ── 1. Obstacle map ─────────────────────────────────────────────────
        obstacle_map = get_obstacle_map(game_state)

        # ── 2. Safe moves — absolute hard filter ────────────────────────────
        safe_moves = get_safe_moves(game_state, obstacle_map)
        if not safe_moves:
            return MoveAction(move=Direction.UP)

        # ── 3. Flood-fill area score for each safe move ──────────────────────
        area_scores = score_moves_by_area(safe_moves, obstacle_map, head)

        # ── 4. Head-collision risk cells ─────────────────────────────────────
        risk_cells = get_head_collision_risks(game_state)

        def is_risky(d: Direction) -> bool:
            return (head.x + d.dx, head.y + d.dy) in risk_cells

        ranked_safe_moves = sorted(
            safe_moves,
            key=lambda d: (0 if is_risky(d) else 1, area_scores[d]),
            reverse=True,
        )

        # ── 5. Voronoi territory map ──────────────────────────────────────────
        #
        # Multi-source BFS from every visible snake head — each open cell is
        # claimed by whichever head reaches it first.  This gives:
        #
        #   territory_ratio   — share of non-contested cells we own  [0, 1]
        #   our_cells         — the exact set we control this turn
        #
        # territory_ratio replaces the old length-only dominance check and also
        # feeds into the food-urgency penalty scale (low ratio → more urgent).
        territory_counts, territory_cells = compute_voronoi(game_state, obstacle_map)
        our_cells      = territory_cells.get(game_state.you.id, set())
        our_count      = territory_counts.get(game_state.you.id, 0)
        total_claimed  = sum(territory_counts.values())
        territory_ratio = our_count / max(1, total_claimed)

        # ── 6. Food memory (Blackout vision support) ─────────────────────────
        view_radius = game_state.game.ruleset.settings.viewRadius
        if view_radius is not None:
            vision_mask = get_vision_mask(
                width=game_state.board.width, height=game_state.board.height,
                center=head, radius=view_radius,
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

        # ── 7. HUNT: intercept a visible smaller opponent ────────────────────
        if health > HUNT_HEALTH_THRESHOLD:
            hunt_dir = find_hunt_move(
                game_state=game_state, obstacle_map=obstacle_map, head=head,
                safe_moves=safe_moves, area_scores=area_scores,
                snake_length=snake_length, risk_cells=risk_cells,
                health=health, territory_ratio=territory_ratio,
            )
            if hunt_dir is not None:
                return MoveAction(move=hunt_dir)

        # ── 8. SQUEEZE: late-game boundary compression ───────────────────────
        #
        # When exactly one opponent remains visible and we own the majority of
        # territory, we stop coiling passively and instead advance our Voronoi
        # boundary directly toward their space.
        #
        # Each turn we A* toward the nearest cell they currently own.  Moving
        # there shifts the BFS distances in our favour next turn, so the
        # boundary advances by ~1 cell per move.  Over time this creates a
        # contracting wall that leaves the opponent with an ever-shrinking
        # pocket — they eventually run out of room and die.
        #
        # Squeeze sits between Hunt and Coil:
        #   • Hunt still fires first — a direct head-on kill is always better.
        #   • Squeeze beats Coil in 1v1 — passive patrolling is sub-optimal
        #     when we can actively compress the opponent's remaining space.
        visible_opponents = [
            s for s in game_state.board.snakes
            if s.id != game_state.you.id and s.head is not None
        ]
        should_squeeze = (
            len(visible_opponents) == 1
            and health > SQUEEZE_HEALTH_THRESHOLD
            and territory_ratio > SQUEEZE_TERRITORY_THRESHOLD
        )
        if should_squeeze:
            # ── Health-gated food interrupt ──────────────────────────────────
            #
            # Health thresholds inside squeeze mode:
            #   health > 55  → pure squeeze: advance boundary turn by turn
            #   40 < health ≤ 55  → food interrupt: eat nearest food first,
            #                        then resume squeezing next turn
            #   health ≤ 40  → should_squeeze is False; falls through to the
            #                   normal food-seeking step below
            #
            # This prevents The Dominator from starving while ahead — it
            # briefly breaks off to eat, restores health to 100, then picks up
            # exactly where it left off without losing meaningful territory.
            if health <= SQUEEZE_FOOD_INTERRUPT_HEALTH:
                interrupt_dir: Optional[Direction] = None
                interrupt_cost = float('inf')
                for food in agent_state.possible_food:
                    d, length = a_star_wrapper(obstacle_map, head, food)
                    if d is None or d not in safe_moves:
                        continue
                    penalty  = compute_food_penalties(
                        health=health, area=area_scores[d],
                        snake_length=snake_length, risky=is_risky(d),
                        territory_ratio=territory_ratio,
                    )
                    eff_cost = length * penalty
                    if eff_cost < interrupt_cost:
                        interrupt_dir  = d
                        interrupt_cost = eff_cost
                if interrupt_dir is not None:
                    return MoveAction(move=interrupt_dir)
            else:
                squeeze_dir = find_squeeze_move(
                    game_state=game_state, obstacle_map=obstacle_map, head=head,
                    safe_moves=safe_moves, area_scores=area_scores,
                    risk_cells=risk_cells, territory_cells=territory_cells,
                    snake_length=snake_length, health=health,
                    territory_ratio=territory_ratio,
                )
                if squeeze_dir is not None:
                    return MoveAction(move=squeeze_dir)

        # ── 9. COIL: Voronoi-driven territory patrol ──────────────────────────
        #
        # Replaces the old length-only is_dominant() check.  We coil when:
        #   • We own ≥ COIL_TERRITORY_THRESHOLD (50%) of non-contested cells
        #   • At least one opponent is visible (no point coiling alone)
        #   • Health is above the starvation threshold
        #
        # Inside find_coil_move the score now maximises our_territory cells
        # reachable from the landing position — so The Dominator actively
        # patrols its own controlled space rather than wandering into the
        # contested middle.
        should_coil = (
            health > COIL_HEALTH_THRESHOLD
            and territory_ratio >= COIL_TERRITORY_THRESHOLD
            and len(visible_opponents) > 0
        )
        if should_coil:
            coil_dir = find_coil_move(
                game_state=game_state, obstacle_map=obstacle_map, head=head,
                safe_moves=safe_moves, risk_cells=risk_cells,
                snake_length=snake_length, our_territory_cells=our_cells,
            )
            if coil_dir is not None:
                return MoveAction(move=coil_dir)

        # ── 9. Seek food — health + territory aware penalties ────────────────
        #
        # territory_ratio < TERRITORY_PRESSURE_LOW adds extra urgency on top
        # of the health curve, so a losing snake eats more aggressively to
        # grow and reclaim space.
        result_direction: Optional[Direction] = None
        best_cost = float('inf')

        for food in agent_state.possible_food:
            direction, length = a_star_wrapper(obstacle_map, head, food)
            if direction is None or direction not in safe_moves:
                continue
            penalty  = compute_food_penalties(
                health=health, area=area_scores[direction],
                snake_length=snake_length, risky=is_risky(direction),
                territory_ratio=territory_ratio,
            )
            eff_cost = length * penalty
            if eff_cost < best_cost:
                result_direction = direction
                best_cost        = eff_cost

        if result_direction is not None:
            return MoveAction(move=result_direction)

        # ── 10. Fallback: follow own tail ─────────────────────────────────────
        tail = game_state.you.body[-1]
        if tail is not None:
            direction, _ = a_star_wrapper(obstacle_map, head, tail)
            if direction is not None and direction in safe_moves:
                if not is_risky(direction):
                    return MoveAction(move=direction)
                if is_risky(ranked_safe_moves[0]):
                    return MoveAction(move=direction)
                return MoveAction(move=ranked_safe_moves[0])

        # ── 11. Final fallback: safest, most open direction ───────────────────
        return MoveAction(move=ranked_safe_moves[0])

    def end(self, game_state: GameState):
        if game_state.game.id in self.agent_states:
            del self.agent_states[game_state.game.id]


# ---------------------------------------------------------
# A* Algorithm
# ---------------------------------------------------------
def a_star_wrapper(grid: np.ndarray, start: Point, goal: Point) -> Tuple[Optional[Direction], int]:
    if start.x == goal.x and start.y == goal.y:
        return None, 0
    path = a_star(grid, (start.y, start.x), (goal.y, goal.x))
    if path is None or len(path) < 2:
        return None, 9_999_999
    next_pos = path[1]
    return Direction.from_board_delta((next_pos[1] - start.x, next_pos[0] - start.y)), len(path)


def a_star(
    grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    h, w = grid.shape
    open_set: list[tuple[int, tuple[int, int]]] = [(0, start)]
    g_score: dict[tuple[int, int], int] = {start: 0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

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
            if 0 <= nr < h and 0 <= nc < w and not grid[nr, nc]:
                neighbor: tuple[int, int] = (nr, nc)
                new_g = g_score[current] + 1
                if new_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor]   = new_g
                    heapq.heappush(open_set, (new_g + abs(nr - goal[0]) + abs(nc - goal[1]), neighbor))

    return None
