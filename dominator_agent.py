from dataclasses import dataclass, field
import heapq
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point

# ── Tunable constants ────────────────────────────────────────────────────────
HUNT_HEALTH_THRESHOLD = 40   # Hunt only when health > this
HUNT_MAX_DISTANCE     = 10   # Max A* steps to bother chasing a target
COIL_HEALTH_THRESHOLD = 40   # Coil only when health > this
URGENCY_FULL          = 70   # ≥ this health → full food-path penalties
URGENCY_NONE          = 20   # ≤ this health → zero food-path penalties (desperation)


# ---------------------------------------------------------
# Health-aware food penalty helpers
# ---------------------------------------------------------
def food_penalty_scale(health: int) -> float:
    """
    Linear scale [0.0, 1.0] governing how much area/risk penalties apply to
    food paths.  Full penalties at high health; none at desperation level.

      health ≥ 70 → 1.00  (area_pen 2.0, risk_pen 3.0)
      health = 45 → 0.50  (area_pen 1.5, risk_pen 2.0)
      health ≤ 20 → 0.00  (area_pen 1.0, risk_pen 1.0 — take any food)
    """
    if health >= URGENCY_FULL:
        return 1.0
    if health <= URGENCY_NONE:
        return 0.0
    return (health - URGENCY_NONE) / (URGENCY_FULL - URGENCY_NONE)


def compute_food_penalties(health: int, area: int, snake_length: int, risky: bool) -> float:
    """Combined area × risk multiplier, scaled by current health urgency."""
    scale    = food_penalty_scale(health)
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
    """
    BFS from start; returns the full set of reachable open (row, col) cells.
    Used for tail-reachability checks in coil mode.
    """
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


def flood_fill_area(grid: np.ndarray, start: Tuple[int, int]) -> int:
    """Convenience wrapper — returns just the count of reachable cells."""
    return len(flood_fill_cells(grid, start))


def score_moves_by_area(
    safe_moves: List[Direction],
    obstacle_map: np.ndarray,
    head: Point,
) -> Dict[Direction, int]:
    return {d: flood_fill_area(obstacle_map, (head.y + d.dy, head.x + d.dx)) for d in safe_moves}


def get_head_collision_risks(game_state: GameState) -> Set[Tuple[int, int]]:
    """
    (x, y) cells that an equal-or-larger opponent could step into next turn.
    Moving there = head-on collision we lose or tie (both fatal).
    """
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
) -> Optional[Direction]:
    """
    Route toward a cell adjacent to a smaller opponent's head.
    Same health-aware penalty formula as food seeking.
    """
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

            area     = area_scores[direction]
            risky    = (head.x + direction.dx, head.y + direction.dy) in risk_cells
            eff_cost = length * compute_food_penalties(health, area, snake_length, risky)

            if eff_cost < best_cost:
                best_direction = direction
                best_cost      = eff_cost

    return best_direction


# ---------------------------------------------------------
# Coil (territory-denial)
# ---------------------------------------------------------
def is_dominant(game_state: GameState) -> bool:
    """
    True when we are strictly longer than every visible opponent.
    Requires at least one opponent in view — no point coiling in an empty board.
    In Blackout rules this is automatically limited to visible snakes.
    """
    my_id     = game_state.you.id
    my_length = game_state.you.length or len(game_state.you.body)
    opponents = [s for s in game_state.board.snakes if s.id != my_id and s.head is not None]
    if not opponents:
        return False
    return all((s.length or len(s.body)) < my_length for s in opponents)


def find_coil_move(
    game_state: GameState,
    obstacle_map: np.ndarray,
    head: Point,
    safe_moves: List[Direction],
    risk_cells: Set[Tuple[int, int]],
    snake_length: int,
) -> Optional[Direction]:
    """
    Territory-denial coil strategy — activated when we are the dominant snake.

    For each safe landing cell we run a full flood-fill and score the move on
    three criteria (in priority order):

      1. tail_reachable — can we still reach our own tail from there?
         Keeping the tail reachable maintains the coil loop and prevents
         self-entrapment.  If the tail is outside vision (None), we proxy
         using area ≥ snake_length as a safe-enough heuristic.

      2. not_risky — the landing cell is not in any equal/larger opponent's
         next-turn reach (avoids accidental head-on while coiling).

      3. area — more open space is always preferred; maximising area denies
         territory to trapped opponents and keeps the loop wide.

    No A* calls needed — flood_fill_cells gives us both the area count and
    tail reachability in a single BFS pass per candidate move.
    """
    tail = game_state.you.body[-1]

    best_direction: Optional[Direction] = None
    best_score: Tuple[int, int, int] = (-1, -1, -1)

    for d in safe_moves:
        nx, ny = head.x + d.dx, head.y + d.dy
        reachable = flood_fill_cells(obstacle_map, (ny, nx))
        area      = len(reachable)

        if tail is not None:
            tail_reachable = 1 if (tail.y, tail.x) in reachable else 0
        else:
            # Tail outside vision — use area as proxy
            tail_reachable = 1 if area >= snake_length else 0

        not_risky = 0 if (nx, ny) in risk_cells else 1
        score     = (tail_reachable, not_risky, area)

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
        #    Walls and reversals kill instantly. Nothing may override this.
        safe_moves = get_safe_moves(game_state, obstacle_map)
        if not safe_moves:
            return MoveAction(move=Direction.UP)

        # ── 3. Flood-fill area score for each safe move ──────────────────────
        area_scores = score_moves_by_area(safe_moves, obstacle_map, head)

        # ── 4. Head-collision risk cells ─────────────────────────────────────
        risk_cells = get_head_collision_risks(game_state)

        def is_risky(d: Direction) -> bool:
            return (head.x + d.dx, head.y + d.dy) in risk_cells

        # Rank: non-risky first, then most open — governs all fallbacks
        ranked_safe_moves = sorted(
            safe_moves,
            key=lambda d: (0 if is_risky(d) else 1, area_scores[d]),
            reverse=True,
        )

        # ── 5. Food memory (Blackout vision support) ─────────────────────────
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

        # ── 6. HUNT: intercept a visible smaller opponent ────────────────────
        #    Only when healthy.  Hunt is checked before coil because an
        #    adjacent kill opportunity is more time-sensitive than patrolling.
        if health > HUNT_HEALTH_THRESHOLD:
            hunt_dir = find_hunt_move(
                game_state=game_state, obstacle_map=obstacle_map, head=head,
                safe_moves=safe_moves, area_scores=area_scores,
                snake_length=snake_length, risk_cells=risk_cells, health=health,
            )
            if hunt_dir is not None:
                return MoveAction(move=hunt_dir)

        # ── 7. COIL: territory-denial when dominant ──────────────────────────
        #
        # When we are the longest visible snake AND healthy, stop chasing food
        # and instead patrol the board in a space-filling loop.  This:
        #   • denies territory to cornered opponents, hastening their death
        #   • keeps our tail reachable, preventing self-entrapment
        #   • maximises the open area we control each turn
        #
        # The snake resumes food-seeking the moment health drops to or below
        # COIL_HEALTH_THRESHOLD, so starvation is never a risk.
        if health > COIL_HEALTH_THRESHOLD and is_dominant(game_state):
            coil_dir = find_coil_move(
                game_state=game_state, obstacle_map=obstacle_map, head=head,
                safe_moves=safe_moves, risk_cells=risk_cells,
                snake_length=snake_length,
            )
            if coil_dir is not None:
                return MoveAction(move=coil_dir)

        # ── 8. Seek food — health-aware penalties ────────────────────────────
        #
        #   effective_cost = path_length × compute_food_penalties(health, ...)
        #
        #   health ≥ 70: area_pen 2.0, risk_pen 3.0  (very selective)
        #   health = 45: area_pen 1.5, risk_pen 2.0  (moderately urgent)
        #   health ≤ 20: area_pen 1.0, risk_pen 1.0  (desperate — take anything)
        result_direction: Optional[Direction] = None
        best_cost = float('inf')

        for food in agent_state.possible_food:
            direction, length = a_star_wrapper(obstacle_map, head, food)
            if direction is None or direction not in safe_moves:
                continue
            penalty  = compute_food_penalties(
                health=health, area=area_scores[direction],
                snake_length=snake_length, risky=is_risky(direction),
            )
            eff_cost = length * penalty
            if eff_cost < best_cost:
                result_direction = direction
                best_cost        = eff_cost

        if result_direction is not None:
            return MoveAction(move=result_direction)

        # ── 9. Fallback: follow own tail ─────────────────────────────────────
        tail = game_state.you.body[-1]
        if tail is not None:
            direction, _ = a_star_wrapper(obstacle_map, head, tail)
            if direction is not None and direction in safe_moves:
                if not is_risky(direction):
                    return MoveAction(move=direction)
                if is_risky(ranked_safe_moves[0]):
                    return MoveAction(move=direction)
                return MoveAction(move=ranked_safe_moves[0])

        # ── 10. Final fallback: safest, most open direction ───────────────────
        return MoveAction(move=ranked_safe_moves[0])

    def end(self, game_state: GameState):
        if game_state.game.id in self.agent_states:
            del self.agent_states[game_state.game.id]


# ---------------------------------------------------------
# A* Algorithm
# ---------------------------------------------------------
def a_star_wrapper(grid: np.ndarray, start: Point, goal: Point) -> tuple[Optional[Direction], int]:
    if start.x == goal.x and start.y == goal.y:
        return None, 0
    path = a_star(grid, (start.y, start.x), (goal.y, goal.x))
    if path is None or len(path) < 2:
        return None, 9_999_999
    next_pos = path[1]
    return Direction.from_board_delta((next_pos[1] - start.x, next_pos[0] - start.y)), len(path)


def a_star(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
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
