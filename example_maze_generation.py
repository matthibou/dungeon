"""Script to load a maze generator and visualize a result."""

import argparse

import mlflow
import matplotlib.pyplot as plt

from dungeon.maze import Maze
from dungeon.display import plot_maze


def get_maze_generator(run_id): 
    logged_model = f'runs:/{run_id}/model'

    # Load generator as a PyFuncModel.
    return mlflow.pyfunc.load_model(logged_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id')
    args = parser.parse_args()

    # get maze from MLflow artifacts
    maze_generator = get_maze_generator(args.run_id)

    intial_maze = Maze.init_for_dqn_training(
        n_edge_withdrawn=3,
        distance_between_key_points=4
    )
    valid_maze = Maze.from_dict(
        maze_generator.predict(
            intial_maze.to_dict()
        )
    )
    plot_maze(intial_maze, show=False)
    plt.title('Initial maze (randomly generated)!')
    plot_maze(valid_maze, show=False)
    plt.title('Corresponding valid maze (using policy from trained DQN)!')
    plt.show()




