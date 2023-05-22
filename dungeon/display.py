"""Utilities to build graphical representation of mazes."""

import torch
import matplotlib.pyplot as plt
from itertools import product


def plot_maze(maze, ratio=20, show=True):
    """Display the 2d map associated to the inputed maze."""
    maze_map = build_matrix_maze(maze, ratio)
    plt.figure()
    plt.imshow(maze_map, cmap='gray_r')
    plt.axis('off')
    if show:
        plt.show()


def build_matrix_maze(maze, ratio=20):
    """Build a 2d array dedicated to graphical display."""
    total_units = maze.size * (ratio + 1) + 1
    maze_map = torch.zeros((total_units, total_units))
    # border
    maze_map[0] = 1
    maze_map[-1] = 1
    maze_map[:, 0] = 1
    maze_map[:, -1] = 1
    # vertical edges
    vertical_coord = list(product(
            torch.arange(1, total_units - 1,  ratio + 1),
            torch.arange(ratio + 1, total_units - 1, ratio + 1)
        ))
    for (ind_0, ind_1), egde_val in zip(vertical_coord, maze.edge_values[:maze.n_edges // 2]):
        if egde_val == 1:
            continue
        maze_map[ind_0-1:ind_0+ratio+1, ind_1] = 1
    # horizontal edges
    horizontal_coord = list(product(
            torch.arange(ratio + 1, total_units - 1,  ratio + 1),
            torch.arange(1, total_units - 1, ratio + 1)
        ))
    for (ind_0, ind_1), egde_val in zip(horizontal_coord, maze.edge_values[maze.n_edges // 2:]):
        if egde_val == 1:
            continue
        maze_map[ind_0, ind_1-1:ind_1+ratio+1] = 1
    # Draw letters on the key point positions
    for letter, index in zip(
        ['S', 'E', 'T'], [maze.start_index, maze.exit_index, maze.treasure_index]
    ):
        ind_0 = (index // maze.size) * (ratio + 1) + 1
        ind_1 = (index % maze.size) * (ratio + 1) + 1
        maze_map[ind_0:ind_0+ratio, ind_1:ind_1+ratio] = draw_letter(ratio, letter)
    return maze_map


def draw_letter(ratio, letter):
    """Hard coded letter patterns."""
    assert letter in ('T', 'S', 'E'), (
        "This function only draws the letters 'T', 'S' and 'E'."
        f" Please input one of these letters. Got a '{letter}'!"
    )
    letter_map = torch.zeros((ratio, ratio))
    ind_0 = ratio // 2
    ind_1 = ratio // 2
    if letter == 'T':
        letter_map[ind_0 - (ratio // 6) + 1:ind_0 + 1 + (ratio // 6), ind_1] = 1
        letter_map[ind_0 - (ratio // 4) + 2, ind_1 - (ratio // 6):ind_1 + (ratio // 6) + 1] = 1
    else:
        letter_map[ind_0 - (ratio // 6), ind_1 - (ratio // 8):ind_1 + (ratio // 8) + 1] = 1
        letter_map[ind_0, ind_1 - (ratio // 8):ind_1 + (ratio // 8) + 1] = 1
        letter_map[ind_0 + (ratio // 6), ind_1 - (ratio // 8):ind_1 + (ratio // 8) + 1] = 1
        if letter == 'S':
            letter_map[ind_0 - (ratio // 6):ind_0, ind_1 - (ratio // 8)] = 1
            letter_map[ind_0:ind_0 + (ratio // 6), ind_1 + (ratio // 8)] = 1
        else:
            letter_map[ind_0 - (ratio // 6):ind_0 + (ratio // 6), ind_1 - (ratio // 8)] = 1
    return letter_map
