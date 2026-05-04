from dataclasses import dataclass, field
import heapq
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point

# ── Tunable constants ────────────────────────────────────────────────────────
# Hunting (chasing smaller opponents) is only active above this health level.
HUNT_HEALTH_THRESHOLD = 40

# Maximum A* steps we will chase a hunt target.
HUNT_MAX_DISTANCE = 10

# Health levels used to shape the food-urgency curve.
# Between URGENCY_FULL (full penalties) and URGENCY_NONE (zero penalties)
# all area/risk penalties on food paths scale linearly with health.
URGENCY_FULL = 70   # ≥ this health → full corridor + danger penalties on food
URGENCY_NONE = 20   # ≤ this health → no penalties at all (desperation mode)


# ---------------------------------------------------------
# Helper: health-aware food penalty
# ---------------------------------------------------------
def food_penalty_scale(health: int) -> float:
    """
    Returns a scale factor in [0.0, 1.0] that governs how much the
    area-corridor and head-risk penalties are applied to food paths.

      health ≥ URGENCY_FULL → 1.0  (full penalties — be selective)
      health ≤ URGENCY_NONE → 0.0  (no penalties — take any food you can reach)
      in between           → linear interpolation

    Examples (URGENCY_FULL=70, URGENCY_NONE=20):
      health 70 → scale 1.00  area_pen 2.0  risk_pen 3.0
      health 55 → scale 0.70  area_pen 1.7  risk_pen 2.4
      health 40 → scale 0.40  area_pen 1.4  risk_pen 1.8
      health 25 → scale 0.10  area_pen 1.1  risk_pen 1.2
      health 20 → scale 0.00  area_pen 1.0  risk_pen 1.0  (desperation)
    """
    if health >= URGENCY_FULL:
        return 1.0
    if health <= URGENCY_NONE:
        return 0.0
    return (health - URGENCY_NONE) / (URGENCY_FULL - URGENCY_NONE)


def compute_food_penalties(
    health: int,
    area: int,
    snake_length: int,
    risky: bool,
) -> float:
    """
    Returns the combined area × risk penalty multiplier for a food path,
    scaled by current health urgency.

    At full health, corridor and danger paths are strongly penalised.
    As health falls the snake accepts riskier/tighter routes automatically.
    """
    scale      = food_penalty_scale(health)
    area_pen   = 1.0 + 1.0 * scale if area < snake_length else 1.0
    risk_pen   = 1.0 + 2.0 * scale if risky else 1.0
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
    Hard filter applied before every other decision:
      1. Within board boundaries  (board.width / board.height)
      2. Not the neck cell        (prevents instant self-reversal)
      3. Not an obstacle cell     (snake bodies)
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
        if nx < 0 or nx >= game_state.board.width:
            continue
        if ny < 0 or ny >= game_state.board.height:
            continue
        if neck is not None and nx == neck.x and ny == neck.y:
            continue
        if obstacle_map[ny, nx]:
            continue
        safe.append(d)
    return safe


def flood_fill_area(grid: np.ndarray, start: Tuple[int, int]) -> int:
    """BFS from start; returns count of reachable open cells."""
    h, w = grid.shape
    r0, c0 = start
    if not (0 <= r0 < h and 0 <= c0 < w) or grid[r0, c0]:
        return 0
    visited: Set[Tuple[int, int]] = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not grid[nr, nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return len(visited)


def score_moves_by_area(
    safe_moves: List[Direction],
    obstacle_map: np.ndarray,
    head: Point,
) -> Dict[Direction, int]:
    return {d: flood_fill_area(obstacle_map, (head.y + d.dy, head.x + d.dx)) for d in safe_moves}


def get_head_collision_risks(game_state: GameState) -> Set[Tuple[int, int]]:
    """
    Returns (x, y) cells reachable in one step by any equal-or-larger opponent.
    Stepping into one of these is a head-on collision we lose (or tie → die).
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
    If we arrive there on the same turn they step in, we win the collision.
    Uses the same health-aware penalties as food seeking.
    """
    my_id = game_state.you.id
    best_direction: Optional[Direction] = None
    best_cost = float('inf')

    for snake in game_state.board.snakes:
        if snake.id == my_id or snake.head is None:
            continue
        if (snake.length or len(snake.body)) >= snake_length:
            continue  # Only hunt strictly-smaller snakes

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
            penalty  = compute_food_penalties(health, area, snake_length, risky)
            eff_cost = length * penalty

            if eff_cost < best_cost:
                best_direction = direction
                best_cost      = eff_cost

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

        # Rank by safety first, then open area — used by all fallbacks
        ranked_safe_moves = sorted(
            safe_moves,
            key=lambda d: (0 if is_risky(d) else 1, area_scores[d]),
            reverse=True,
        )

        # ── 5. Food memory (Blackout vision support) ─────────────────────────
        view_radius = game_state.game.ruleset.settings.viewRadius
        if view_radius is not None:
            vision_mask = get_vision_mask(
                width=game_state.board.width,
                height=game_state.board.height,
                center=head,
                radius=view_radius,
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
        #
        # Only when healthy enough. As health nears HUNT_HEALTH_THRESHOLD the
        # hunt-path cost rises (shared penalty function), so food naturally
        # wins the competition before the hard cutoff is even reached.
        if health > HUNT_HEALTH_THRESHOLD:
            hunt_dir = find_hunt_move(
                game_state=game_state,
                obstacle_map=obstacle_map,
                head=head,
                safe_moves=safe_moves,
                area_scores=area_scores,
                snake_length=snake_length,
                risk_cells=risk_cells,
                health=health,
            )
            if hunt_dir is not None:
                return MoveAction(move=hunt_dir)

        # ── 7. Seek food — health-aware penalties ────────────────────────────
        #
        # effective_cost = path_length × compute_food_penalties(health, ...)
        #
        # penalty_scale drives both multipliers:
        #   health ≥ 70 → scale 1.0  → area_pen 2.0, risk_pen 3.0  (selective)
        #   health = 45 → scale 0.50 → area_pen 1.5, risk_pen 2.0
        #   health = 20 → scale 0.0  → area_pen 1.0, risk_pen 1.0  (desperate)
        #
        # The snake never needs a code change to "eat faster when low on health"
        # — the penalty curve does it automatically.
        result_direction: Optional[Direction] = None
        best_cost = float('inf')

        for food in agent_state.possible_food:
            direction, length = a_star_wrapper(obstacle_map, head, food)
            if direction is None or direction not in safe_moves:
                continue

            penalty  = compute_food_penalties(
                health=health,
                area=area_scores[direction],
                snake_length=snake_length,
                risky=is_risky(direction),
            )
            eff_cost = length * penalty

            if eff_cost < best_cost:
                result_direction = direction
                best_cost        = eff_cost

        if result_direction is not None:
            return MoveAction(move=result_direction)

        # ── 8. Fallback: follow own tail ─────────────────────────────────────
        tail = game_state.you.body[-1]
        if tail is not None:
            direction, _ = a_star_wrapper(obstacle_map, head, tail)
            if direction is not None and direction in safe_moves:
                if not is_risky(direction):
                    return MoveAction(move=direction)
                if is_risky(ranked_safe_moves[0]):
                    return MoveAction(move=direction)
                return MoveAction(move=ranked_safe_moves[0])

        # ── 9. Final fallback: safest, most open direction ────────────────────
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
