from dataclasses import dataclass, field
import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple

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
    This must be evaluated before any food/pathfinding logic.
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
    BFS from `start`, counting all reachable open cells (not blocked by grid).
    Used to score how much space a move opens up vs traps the snake.
    Returns 0 if start is itself blocked or out of bounds.
    """
    h, w = grid.shape
    r0, c0 = start
    if not (0 <= r0 < h and 0 <= c0 < w) or grid[r0, c0]:
        return 0

    visited: set[Tuple[int, int]] = {start}
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
    For each safe move, compute the number of open cells reachable
    from that position via flood fill. Higher = more open space.
    """
    scores: Dict[Direction, int] = {}
    for d in safe_moves:
        nx, ny = head.x + d.dx, head.y + d.dy
        scores[d] = flood_fill_area(obstacle_map, (ny, nx))
    return scores


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
        # Wall hits and direction reversals kill immediately — nothing overrides this.
        safe_moves = get_safe_moves(game_state, obstacle_map)

        if not safe_moves:
            return MoveAction(move=Direction.UP)

        # ── Step 3: Score safe moves by flood-fill area ─────────────────────
        # How much open space is reachable if we take each safe step?
        # This prevents us from walking into dead-end corridors.
        area_scores = score_moves_by_area(safe_moves, obstacle_map, head)

        # Moves ranked best→worst by open area — used for all fallbacks
        ranked_safe_moves = sorted(safe_moves, key=lambda d: area_scores[d], reverse=True)

        # Snake body length: used to judge whether an area is "too small"
        snake_length = game_state.you.length or len(game_state.you.body)

        # ── Step 4: Update food memory (Blackout vision support) ────────────
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

        # ── Step 5: A* toward food, weighted by corridor danger ─────────────
        # Effective cost = path_length * penalty, where penalty is 2× if the
        # first step leads into an area smaller than our own body. This means
        # food 5 steps away through open space beats food 3 steps away through
        # a dead-end corridor (5×1 = 5 < 3×2 = 6).
        result_direction: Optional[Direction] = None
        best_effective_cost = float('inf')

        for food in agent_state.possible_food:
            direction, length = a_star_wrapper(obstacle_map, head, food)
            if direction is None or direction not in safe_moves:
                continue

            area = area_scores[direction]
            # Penalise moves leading into areas smaller than the snake itself
            penalty = 2.0 if area < snake_length else 1.0
            effective_cost = length * penalty

            if effective_cost < best_effective_cost:
                result_direction = direction
                best_effective_cost = effective_cost

        if result_direction is not None:
            return MoveAction(move=result_direction)

        # ── Step 6: Fallback — follow own tail (avoids self-enclosure) ───────
        tail = game_state.you.body[-1]
        if tail is not None:
            direction, _ = a_star_wrapper(obstacle_map, head, tail)
            if direction is not None and direction in safe_moves:
                return MoveAction(move=direction)

        # ── Step 7: Final fallback — most open safe direction ────────────────
        # ranked_safe_moves[0] has the largest flood-fill area, minimising the
        # chance of entering a dead-end corridor.
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
