from dataclasses import dataclass, field
import heapq
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point

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
    Returns moves that are:
      1. Within board boundaries (uses actual board width/height)
      2. Not the neck position (prevents reversing direction)
      3. Not occupied by an obstacle (snake bodies)
    This is evaluated before any food/pathfinding logic — absolute priority.
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

        # 1. Wall check using actual board dimensions
        if nx < 0 or nx >= game_state.board.width:
            continue
        if ny < 0 or ny >= game_state.board.height:
            continue

        # 2. Neck / reverse-direction check
        if neck is not None and nx == neck.x and ny == neck.y:
            continue

        # 3. Obstacle check
        if obstacle_map[ny, nx]:
            continue

        safe.append(d)

    return safe


def flood_fill_area(grid: np.ndarray, start: Tuple[int, int]) -> int:
    """
    BFS from `start`, counting all reachable open cells.
    Returns 0 if start is blocked or out of bounds.
    """
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
    """
    For each safe move, flood-fill from the landing cell to count open space.
    Higher score = more room to manoeuvre.
    """
    scores: Dict[Direction, int] = {}
    for d in safe_moves:
        nx, ny = head.x + d.dx, head.y + d.dy
        scores[d] = flood_fill_area(obstacle_map, (ny, nx))
    return scores


def get_head_collision_risks(game_state: GameState) -> Set[Tuple[int, int]]:
    """
    Returns (x, y) cells that an equal-or-larger opponent snake could move
    into on the next turn.  Stepping into one of these means our head meets
    theirs — we always lose ties or lose outright when they're bigger.

    In Blackout rules only visible opponents (within view radius) appear on
    the board, so this naturally respects the limited-vision constraint.
    """
    my_id = game_state.you.id
    my_length = game_state.you.length or len(game_state.you.body)

    risky: Set[Tuple[int, int]] = set()

    for snake in game_state.board.snakes:
        if snake.id == my_id or snake.head is None:
            continue

        opp_length = snake.length or len(snake.body)
        if opp_length < my_length:
            # Smaller snake — head-on collision favours us; skip
            continue

        # Every cell the opponent could legally step into next turn
        for d in Direction:
            nx = snake.head.x + d.dx
            ny = snake.head.y + d.dy
            if 0 <= nx < game_state.board.width and 0 <= ny < game_state.board.height:
                risky.add((nx, ny))

    return risky


# ---------------------------------------------------------
# Battlesnake Agent Implementation
# ---------------------------------------------------------
@dataclass
class AgentState:
    possible_food: list[Food] = field(default_factory=list)


class DominatorAgent(BaseAgent):
    def __init__(self):
        self.agent_states: dict[str, AgentState] = {}

    def get_name(self):
        return "The Dominator"

    def get_color(self):
        return "#FF0000"

    def get_author(self):
        return "The Dominator"

    def get_head(self):
        return "villain"

    def get_tail(self):
        return "sharp"

    def start(self, game_state: GameState):
        self.agent_states[game_state.game.id] = AgentState()

    def move(self, game_state: GameState) -> MoveAction:
        head = game_state.you.head
        if head is None:
            return MoveAction(move=Direction.UP)

        # Guard: initialise state if start() was never called
        if game_state.game.id not in self.agent_states:
            self.agent_states[game_state.game.id] = AgentState()
        agent_state = self.agent_states[game_state.game.id]

        # ── Step 1: Build obstacle map ──────────────────────────────────────
        obstacle_map = get_obstacle_map(game_state)

        # ── Step 2: Compute safe moves — ABSOLUTE TOP PRIORITY ─────────────
        # Walls and direction reversals kill immediately — nothing overrides this.
        safe_moves = get_safe_moves(game_state, obstacle_map)

        if not safe_moves:
            return MoveAction(move=Direction.UP)

        # ── Step 3: Score safe moves by flood-fill area ─────────────────────
        area_scores = score_moves_by_area(safe_moves, obstacle_map, head)

        # ── Step 4: Identify head-collision risk cells ──────────────────────
        # Any cell an equal-or-larger opponent could move to next turn.
        risk_cells = get_head_collision_risks(game_state)

        def is_risky(d: Direction) -> bool:
            return (head.x + d.dx, head.y + d.dy) in risk_cells

        # Rank safe moves: non-risky first, then by open area (both descending).
        # This means the final fallback always picks the safest, most open move.
        ranked_safe_moves = sorted(
            safe_moves,
            key=lambda d: (0 if is_risky(d) else 1, area_scores[d]),
            reverse=True,
        )

        snake_length = game_state.you.length or len(game_state.you.body)

        # ── Step 5: Update food memory (Blackout vision support) ────────────
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

        # ── Step 6: A* toward food, penalised for corridors and danger ───────
        #
        # effective_cost = path_length × area_penalty × risk_penalty
        #
        #   area_penalty = 2.0  if first-step area < snake length (dead-end risk)
        #   risk_penalty = 3.0  if first step lands in a head-collision danger cell
        #
        # Examples (snake length 5):
        #   Food 3 steps, safe open path          →  3 × 1 × 1  =  3  ✓ best
        #   Food 3 steps, dead-end corridor        →  3 × 2 × 1  =  6
        #   Food 3 steps, head-collision danger    →  3 × 1 × 3  =  9
        #   Food 3 steps, dead-end + danger        →  3 × 2 × 3  = 18  ✗ avoid

        result_direction: Optional[Direction] = None
        best_effective_cost = float('inf')

        for food in agent_state.possible_food:
            direction, length = a_star_wrapper(obstacle_map, head, food)
            if direction is None or direction not in safe_moves:
                continue

            area        = area_scores[direction]
            area_pen    = 2.0 if area < snake_length else 1.0
            risk_pen    = 3.0 if is_risky(direction) else 1.0
            eff_cost    = length * area_pen * risk_pen

            if eff_cost < best_effective_cost:
                result_direction    = direction
                best_effective_cost = eff_cost

        if result_direction is not None:
            return MoveAction(move=result_direction)

        # ── Step 7: Fallback — follow own tail (avoids self-enclosure) ───────
        tail = game_state.you.body[-1]
        if tail is not None:
            direction, _ = a_star_wrapper(obstacle_map, head, tail)
            if direction is not None and direction in safe_moves:
                # Still check: prefer non-risky tail-chase direction
                if not is_risky(direction):
                    return MoveAction(move=direction)
                # Risky tail-chase — only use it if no better ranked move exists
                best_ranked = ranked_safe_moves[0]
                if is_risky(best_ranked):
                    return MoveAction(move=direction)
                return MoveAction(move=best_ranked)

        # ── Step 8: Final fallback — safest, most open direction ─────────────
        return MoveAction(move=ranked_safe_moves[0])

    def end(self, game_state: GameState):
        if game_state.game.id in self.agent_states:
            del self.agent_states[game_state.game.id]


# ---------------------------------------------------------
# A* Algorithm
# ---------------------------------------------------------
def a_star_wrapper(grid: np.ndarray, start: Point, goal: Point) -> tuple[Optional[Direction], int]:
    """Converts from battlesnake x-y coords to (row, col) index tuples used by a_star()."""
    if start.x == goal.x and start.y == goal.y:
        return None, 0

    path = a_star(grid, (start.y, start.x), (goal.y, goal.x))
    if path is None or len(path) < 2:
        return None, 9_999_999

    next_pos = path[1]
    result_direction = Direction.from_board_delta((next_pos[1] - start.x, next_pos[0] - start.y))
    return result_direction, len(path)


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
                    g_score[neighbor] = new_g
                    f_score = new_g + abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(open_set, (f_score, neighbor))

    return None
