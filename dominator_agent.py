from dataclasses import dataclass, field
import heapq
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point

# Hunting is only active when health exceeds this level.
# Below it the snake prioritises eating over killing.
HUNT_HEALTH_THRESHOLD = 40

# Maximum A* distance (steps) we will bother chasing a target.
# Beyond this the intercept is unlikely to succeed before the opponent escapes.
HUNT_MAX_DISTANCE = 10

# ---------------------------------------------------------
# Helper Functions
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
    dist_sq = abs(x - center.x) + abs(y - center.y)
    return dist_sq <= radius


def get_safe_moves(game_state: GameState, obstacle_map: np.ndarray) -> List[Direction]:
    """
    Returns only moves that:
      1. Stay within board boundaries (board.width / board.height)
      2. Do not reverse into the neck (instant self-collision)
      3. Are not occupied by any snake body segment
    Evaluated before every other logic — absolute priority.
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
        nx = head.x + d.dx
        ny = head.y + d.dy

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
    """BFS from start, counting reachable open cells. Returns 0 if blocked."""
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
    """Flood-fill area reachable from each safe landing cell."""
    return {d: flood_fill_area(obstacle_map, (head.y + d.dy, head.x + d.dx)) for d in safe_moves}


def get_head_collision_risks(game_state: GameState) -> Set[Tuple[int, int]]:
    """
    Cells that an equal-or-larger opponent could step into next turn.
    Moving there means a head-on we lose or tie — always fatal.
    """
    my_id = game_state.you.id
    my_length = game_state.you.length or len(game_state.you.body)
    risky: Set[Tuple[int, int]] = set()
    for snake in game_state.board.snakes:
        if snake.id == my_id or snake.head is None:
            continue
        opp_length = snake.length or len(snake.body)
        if opp_length < my_length:
            continue
        for d in Direction:
            nx = snake.head.x + d.dx
            ny = snake.head.y + d.dy
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
) -> Optional[Direction]:
    """
    Aggressive mode: find a direction that intercepts a smaller opponent.

    Strategy:
      - For every visible opponent shorter than us, enumerate the four cells
        adjacent to their head (their possible next positions).
      - Run A* from our head to each of those intercept cells.
      - Pick the intercept move with the lowest effective cost (distance ×
        corridor penalty × risk penalty), capped at HUNT_MAX_DISTANCE steps.

    Because we route to a cell *adjacent* to the opponent's head rather than
    the head itself (which is an obstacle), A* works without modifying the map.
    When we arrive at that cell on the same turn the opponent steps there, our
    head meets theirs and we win — we are longer.
    """
    my_id = game_state.you.id
    best_direction: Optional[Direction] = None
    best_cost = float('inf')

    for snake in game_state.board.snakes:
        if snake.id == my_id or snake.head is None:
            continue

        opp_length = snake.length or len(snake.body)
        if opp_length >= snake_length:
            continue  # Only hunt snakes strictly smaller than us

        opp_head = snake.head

        # Each cell the opponent could step into is a valid intercept target
        for d in Direction:
            tx = opp_head.x + d.dx
            ty = opp_head.y + d.dy

            if not (0 <= tx < game_state.board.width and 0 <= ty < game_state.board.height):
                continue
            if obstacle_map[ty, tx]:
                continue  # Already blocked — can't stand there

            target = Point(x=tx, y=ty)
            direction, length = a_star_wrapper(obstacle_map, head, target)

            if direction is None or direction not in safe_moves:
                continue
            if length > HUNT_MAX_DISTANCE:
                continue

            area      = area_scores[direction]
            area_pen  = 2.0 if area < snake_length else 1.0
            risk_pen  = 3.0 if (head.x + direction.dx, head.y + direction.dy) in risk_cells else 1.0
            eff_cost  = length * area_pen * risk_pen

            if eff_cost < best_cost:
                best_direction = direction
                best_cost      = eff_cost

    return best_direction


# ---------------------------------------------------------
# Battlesnake Agent Implementation
# ---------------------------------------------------------
@dataclass
class AgentState:
    possible_food: list[Food] = field(default_factory=list)


class DominatorAgent(BaseAgent):
    def __init__(self):
        self.agent_states: dict[str, AgentState] = {}

    def get_name(self):    return "The Dominator"
    def get_color(self):   return "#FF0000"
    def get_author(self):  return "The Dominator"
    def get_head(self):    return "villain"
    def get_tail(self):    return "sharp"

    def start(self, game_state: GameState):
        self.agent_states[game_state.game.id] = AgentState()

    def move(self, game_state: GameState) -> MoveAction:
        head = game_state.you.head
        if head is None:
            return MoveAction(move=Direction.UP)

        if game_state.game.id not in self.agent_states:
            self.agent_states[game_state.game.id] = AgentState()
        agent_state = self.agent_states[game_state.game.id]

        # ── 1. Obstacle map ─────────────────────────────────────────────────
        obstacle_map = get_obstacle_map(game_state)

        # ── 2. Safe moves — absolute hard filter ────────────────────────────
        #    Walls / reversal kill instantly. Nothing may override this list.
        safe_moves = get_safe_moves(game_state, obstacle_map)
        if not safe_moves:
            return MoveAction(move=Direction.UP)

        # ── 3. Flood-fill area score for each safe move ──────────────────────
        area_scores = score_moves_by_area(safe_moves, obstacle_map, head)

        # ── 4. Head-collision risk cells (equal/larger opponents) ────────────
        risk_cells = get_head_collision_risks(game_state)

        def is_risky(d: Direction) -> bool:
            return (head.x + d.dx, head.y + d.dy) in risk_cells

        # Rank moves: non-risky first, then by open area.
        # Used for all fallbacks so we always pick the safest open direction.
        ranked_safe_moves = sorted(
            safe_moves,
            key=lambda d: (0 if is_risky(d) else 1, area_scores[d]),
            reverse=True,
        )

        snake_length = game_state.you.length or len(game_state.you.body)

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

        # ── 6. HUNT: intercept visible smaller opponents ──────────────────────
        #
        # When we are healthy and longer than an opponent, route to a cell
        # adjacent to their head. If we arrive there on the same turn they
        # step in, our head meets theirs and we win.
        #
        # Hunting is skipped when:
        #   • health ≤ HUNT_HEALTH_THRESHOLD  (we need food more urgently)
        #   • no smaller opponent is visible and reachable within HUNT_MAX_DISTANCE
        if game_state.you.health > HUNT_HEALTH_THRESHOLD:
            hunt_dir = find_hunt_move(
                game_state=game_state,
                obstacle_map=obstacle_map,
                head=head,
                safe_moves=safe_moves,
                area_scores=area_scores,
                snake_length=snake_length,
                risk_cells=risk_cells,
            )
            if hunt_dir is not None:
                return MoveAction(move=hunt_dir)

        # ── 7. A* toward food — penalised for corridors & danger ─────────────
        #
        # effective_cost = path_length × area_penalty × risk_penalty
        #   area_penalty = 2× if first-step area < snake length (corridor trap)
        #   risk_penalty = 3× if first step is a head-collision danger cell
        result_direction: Optional[Direction] = None
        best_effective_cost = float('inf')

        for food in agent_state.possible_food:
            direction, length = a_star_wrapper(obstacle_map, head, food)
            if direction is None or direction not in safe_moves:
                continue

            area      = area_scores[direction]
            area_pen  = 2.0 if area < snake_length else 1.0
            risk_pen  = 3.0 if is_risky(direction) else 1.0
            eff_cost  = length * area_pen * risk_pen

            if eff_cost < best_effective_cost:
                result_direction    = direction
                best_effective_cost = eff_cost

        if result_direction is not None:
            return MoveAction(move=result_direction)

        # ── 8. Fallback: follow own tail (avoids self-enclosure) ─────────────
        tail = game_state.you.body[-1]
        if tail is not None:
            direction, _ = a_star_wrapper(obstacle_map, head, tail)
            if direction is not None and direction in safe_moves:
                if not is_risky(direction):
                    return MoveAction(move=direction)
                # Risky tail-chase: only accept if every ranked move is also risky
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
