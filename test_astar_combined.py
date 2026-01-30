# tests/test_astar.py

"""
These are our unit tests for the A* algorithm.

They're made to test correct path gen for simple grids, obstacle avoidance,
correct type of failure when no path exists, and re planning behavior when environments change
"""

import copy
from typing import List, Tuple

import pytest
from astar import astar

Point = Tuple[int, int]


def is_adjacent(a: Point, b: Point) -> bool:
    # Returns true if the two points are directly adjacent
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def assert_valid_path(grid: List[List[int]], start: Point, goal: Point, path: List[Point]):
    # Helper function to validate a returned path. Makes sure path starts at start and ends at goal,
    # all points are in bounds, no blocked cells are used, and all moves are adjacent
    assert path[0] == start
    assert path[-1] == goal

    h = len(grid)
    w = len(grid[0])

    for i, p in enumerate(path):
        x, y = p
        assert 0 <= x < w and 0 <= y < h
        assert grid[y][x] == 0
        if i > 0:
            assert is_adjacent(path[i - 1], p)


def test_astar_straight_line_no_obstacles():
    # Makes sure that A* returns the shortest path on an empty grid with no actual obstacles
    grid = [[0] * 5 for _ in range(5)]
    start, goal = (0, 0), (4, 0)

    path = astar(grid, start, goal)
    assert path is not None
    assert_valid_path(grid, start, goal, path)

    # 4 moves → 5 points total
    assert len(path) == 5


def test_astar_around_wall_expected_length():
    # Makes sure A* correctly routes around obstacles and still finds the shortest detour
    grid = [[0] * 5 for _ in range(5)]
    grid[0][1] = 1
    grid[0][2] = 1
    grid[0][3] = 1

    start, goal = (0, 0), (4, 0)
    path = astar(grid, start, goal)

    assert path is not None
    assert_valid_path(grid, start, goal, path)

    # One shortest detour path has 7 total points
    assert len(path) == 7


def test_astar_no_path_returns_none():
    # Makes sure A* returns None when the goal is completely unreachable
    grid = [[0] * 3 for _ in range(3)]
    grid[2][1] = 1
    grid[1][2] = 1

    start, goal = (0, 0), (2, 2)
    path = astar(grid, start, goal)

    assert path is None


def test_astar_dynamic_change_replan_avoids_new_obstacle():
    """
    Tests dynamic re-planning:
    After an initial path is computed, a new obstacle
    is introduced and A* must find a new valid path.
    """
    # Tests dynamic replanning where after an initial path is computed, a new obstacle
    # is introduced and A* has to find a new valid path
    grid = [[0] * 5 for _ in range(5)]
    start, goal = (0, 2), (4, 2)

    # Initial plan
    path1 = astar(grid, start, goal)
    assert path1 is not None
    assert_valid_path(grid, start, goal, path1)

    # Introduce a new obstacle mid path
    grid2 = copy.deepcopy(grid)
    grid2[2][2] = 1

    # Replan
    path2 = astar(grid2, start, goal)
    assert path2 is not None
    assert_valid_path(grid2, start, goal, path2)

    # New path must avoid the blocked cell
    assert (2, 2) not in path2

    # Detour should increase path length
    assert len(path2) > len(path1)