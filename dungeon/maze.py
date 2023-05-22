"""4-by-4 grid maze."""

from itertools import product

import torch
import numpy as np


class Maze:
    """4-by-4 grid Maze.

    Represents a 4-by-4 grid maze with three key points
    ("start", "exit" and "treasure").

    A Maze is regarded as valid if there is path between the key points
    "start" and "exit", and the second path between "start"
    and "treasure".

    We use the manhattan distance (l1-norm) to compute
    distance between maze points.

    N.B. We use row-major order for indexing 2d array and
    converting flat index to coordinates.

    Attributes:
      edge_values:
        An `torch.Tensor` filled of 0s and 1s indicating
        the edge status. The presence of the edge is indicated
        by a 0 and the absence by a 1.
        There are 24 edges for a maze (2 * 4 * (4 - 1)) so
        `edge_values` is a tensor of size 24.
        The first 12 values represent the "vertical" edges
        (4 rows of 3 edges). And the last 12 represent the
        "horizontal" edges (4 rows of 4 edges).
      start_index:
        A positive integer indicating the "start" point index.
      exit_index:
        A positive integer indicating the "exit" point index.
      treasure_index:
        A positive integer indicating the "treasure" point index.
    """

    _size = 4
    _n_key_points = 3
    _dtype = torch.int32

    def __init__(
        self,
        edge_values=None,
        key_positions=None
    ):
        """Initialize an instance of Maze.

        Args:
          edge_values:
            24-long `torch.Tensor` with 0 and 1 indicating
            the maze edge states.
            None for a random edge initialization.
            'full' for a full maze with all edges.
          key_positions:
            Tuple of 3 integers representing the key point indices
            ("start", "exit", "treasure").
        """
        if edge_values is None:
            self.edge_values = torch.randint(0, 2, (self.n_edges, ), dtype=self._dtype)
        elif edge_values == 'full':
            self.edge_values = torch.zeros(self.n_edges, dtype=self._dtype)
        else:
            if (
                len(edge_values) != self.n_edges
                or
                0 > torch.any(edge_values)
                or
                torch.any(edge_values) > 1
            ):
                raise ValueError(
                    'Please check `edge_values` tensor.'
                    ' It should be of length 24 and all values be either 0 or 1.'
                    )
            self.edge_values = edge_values

        self.start_index, self.exit_index, self.treasure_index = self.init_position(key_positions)

    def init_position(self, key_positions):
        """Key point indices initialization.

        Check user input for key point indices
        or in case of None input, initialize them randomly.

        Args:
          key_positions:
            tuple of 3 integers or None.

        Returns:
          tuple of 3 integers.
        """
        if key_positions is None:
            return torch.randperm(self.n_squares)[:self.n_key_points].tolist()

        key_pos = torch.tensor(key_positions)
        if torch.any(key_pos > self.n_squares):
            raise ValueError(
                'Key point indices should all be less'
                f' than the number of squares (i.e. {self.n_squares})'
                )
        if torch.any(key_pos < 0):
            raise ValueError('Key point indices should all be greater than 0')

        if len(torch.unique(key_pos)) != self.n_key_points:
            raise ValueError(
                'All key point index should be different. Please check, some of them are equal!'
                f' Got {key_positions}.'
            )
        return key_positions

    @classmethod
    def init_for_dqn_training(
        cls,
        n_edge_withdrawn=3,
        distance_between_key_points=7
    ):
        """Build a Maze with specific characterics.

        Args:
          n_edge_withdrawn:
            Number of edge randomly withdraw from the Maze.
          distance_between_key_points:
            The mini

        Returns:
          tuple of 3 integers.
        """
        maze = cls()
        edge_values = torch.zeros(maze.n_edges, dtype=maze._dtype)
        edge_values[
            torch.randperm(maze.n_squares)[:n_edge_withdrawn]
        ] = 1
        maze.edge_values = edge_values
        while (
            maze.start_exit_path_length() + maze.start_treasure_path_length()
        ) < distance_between_key_points:
            maze.start_index, maze.exit_index, maze.treasure_index = \
                torch.randperm(maze.n_squares)[:maze.n_key_points].tolist()
        return maze

    @classmethod
    def from_dict(cls, dict_):
        """Build a maze from dictionary containing the maze information."""
        return cls(
            edge_values=torch.from_numpy(dict_['edge_values']),
            key_positions=dict_['key_positions'].tolist()
        )

    def to_dict(self):
        """Return a dictionary containing the maze information."""
        return {
            'edge_values': self.edge_values.numpy(),
            'key_positions': np.array(
                [self.start_index, self.exit_index, self.treasure_index]
            )
        }

    def edge_indices(self):
        """Return indices of the edges in the maze."""
        return torch.nonzero(torch.logical_not(self.edge_values)).squeeze(-1)

    def connection_matrix(self):
        """Compute the connection matrix associated to the maze.

        Matrix is exclusively formed by 0s and 1s. A 1-valued square
        with the coordinates (i, j) means that there is no edge
        between the squares the flat indices i and j.
        The matrix is symetric i.e. M[i, j] = M [j, i].

        Returns:
          Squared matrix as a `torch.Tensor` with the sparse layout
          (`torch.sparse_coo`)
        """
        # compute the coordinates for the vertical edges
        indices_left = (
            torch.arange(self.size - 1).reshape(1, -1, 1) +
            torch.arange(2).reshape(1, 1, -1) +
            (torch.arange(self.size) * self.size).reshape(-1, 1, 1)
        ).reshape(-1, 2).T
        # compute the coordinates for the horizontal edges
        indices_up = (
            torch.arange(self.size * (self.size - 1)).reshape(1, -1) +
            (torch.arange(2) * self.size).reshape(-1, 1)
        )
        # put everything together in a sparse matrix
        indices_ = torch.cat((indices_left, indices_up), dim=1)
        conn_matrix = torch.sparse_coo_tensor(
            indices_,
            self.edge_values,
            (self.n_squares, self.n_squares)
        )
        # symetric
        conn_matrix += conn_matrix.T
        # add the diagonal
        conn_matrix += torch.eye(self.n_squares, dtype=self._dtype).to_sparse_coo()
        return conn_matrix

    def propagate(self, position_vector):
        """Propagate a position vector through the maze.

        Compute all the reachable positions in the maze
        from starting positions.

        Args:
          position_vector:
            Vector which should be filled with 0s and 1s
            representing the starting positions.

        Returns:
          A vector where reachable positions have a 1
          otherwise a 0.
        """
        conn_matrix = self.connection_matrix()
        vector_ = torch.zeros_like(position_vector)
        while not torch.all(vector_ == position_vector):
            vector_ = position_vector
            position_vector = torch.matmul(conn_matrix, vector_)
            position_vector[position_vector != 0] = 1
        return position_vector

    def reached_indices(self, vector_):
        """Return only reachacle indices."""
        return torch.nonzero(self.propagate(vector_)).squeeze(-1)

    def reached_indices_from(self, location='start'):
        """Return reachacle indices from key points."""
        if location == 'start':
            return self.reached_indices(self.start_vector)
        elif location == 'exit':
            return self.reached_indices(self.exit_vector)
        elif location == 'treasure':
            return self.reached_indices(self.treasure_vector)
        else:
            raise ValueError(
                "Expect string in ('start', 'exit' or 'treasure')"
                f" as input.  Got '{location}'!"
            )

    def ind_to_coord(self, arr):
        """Compute indices from coordinates."""
        return torch.stack((arr // self.size, arr % self.size), dim=0)

    def coord_to_ind(self, arr):
        """Compute coordinates from indices."""
        return arr[0] * self.size + arr[1]

    def can_reach_treasure(self):
        """Can the "treasure" key point be reached from "start"."""
        return torch.all(
            torch.isin(
                self.treasure_index,
                self.reached_indices_from('start')
            )
        ).item()

    def can_reach_exit(self):
        """Can the "exit" key point be reached from "start"."""
        return torch.all(
            torch.isin(
                self.exit_index,
                self.reached_indices_from('start')
            )
        ).item()

    def is_valid(self):
        """Can the "exit" and "treasure" points be reached from "start"."""
        return torch.all(
            torch.isin(
                torch.tensor([self.treasure_index, self.exit_index]),
                self.reached_indices_from('start')
            )
        ).item()

    def distance_point_group(self, grp_1_indices, grp_2_indices):
        """Compute the minimum distance between two group points."""
        grp_1_coords = self.ind_to_coord(grp_1_indices)
        grp_2_coords = self.ind_to_coord(grp_2_indices)
        distance_matrix = torch.sum(
            torch.abs(
                grp_1_coords.T.reshape(-1, 1, 2) - grp_2_coords.T.reshape(1, -1, 2)
            ), dim=-1
        )
        return torch.min(distance_matrix).item()

    def start_exit_path_length(self):
        """Compute the distance between "start" and "exit"."""
        return self.distance_point_group(
            torch.tensor([self.start_index]),
            torch.tensor([self.exit_index])
        )

    def start_treasure_path_length(self):
        """Compute the distance between "start" and "treasure"."""
        return self.distance_point_group(
            torch.tensor([self.start_index]),
            torch.tensor([self.treasure_index])
        )

    def clone(self):
        """Return a clone maze."""
        return Maze(
            self.edge_values.clone(),
            (
                self.start_index,
                self.exit_index,
                self.treasure_index
            )
        )

    def withdraw_edges(self, edge_to_withdraw):
        """Update edge pattern.

        Args:
          edge_to_withdraw:
            A `torch.Tensor` indicating with edge the withdraw.
            A 1 at a specific index indicates that the edge
            should be withdrawn.
        """
        new_edge_values = self.edge_values | edge_to_withdraw
        if torch.all(new_edge_values == self.edge_values).item():
            print('Warning: Edge pattern has not been changed!!')
        self.edge_values = new_edge_values

    @property
    def start_vector(self):
        """Position vector for the "start" point."""
        vector_ = torch.zeros(self.n_squares, dtype=self._dtype)
        vector_[self.start_index] = 1
        return vector_

    @property
    def exit_vector(self):
        """Position vector for the "exit" point."""
        vector_ = torch.zeros(self.n_squares, dtype=self._dtype)
        vector_[self.exit_index] = 1
        return vector_

    @property
    def treasure_vector(self):
        """Position vector for the "treasure" point."""
        vector_ = torch.zeros(self.n_squares, dtype=self._dtype)
        vector_[self.treasure_index] = 1
        return vector_

    @property
    def size(self):
        """Size of the maze (4 by default)."""
        return self._size

    @property
    def n_squares(self):
        """Number of squares."""
        return self.size ** 2

    @property
    def n_edges(self):
        """Number of edges."""
        return 2 * self.size * (self.size - 1)

    @property
    def n_key_points(self):
        """Number of key points (3 by default)."""
        return self._n_key_points

    def reward(self):
        """Conmputes reward.

        This value is used in a context of reinforcement learning.

        Returns:
          The negative sum of two distances.
          First, the distance between the groups of indices
          reached from "start" and "exit".
          Second, the distance between the groups of indices
          reached from "start" and "treasure".
        """
        reached_coord_start = self.reached_indices_from('start')
        distance_start_exit = self.distance_point_group(
            reached_coord_start,
            self.reached_indices_from('exit')
        )
        distance_start_treasure = self.distance_point_group(
            reached_coord_start,
            self.reached_indices_from('treasure')
        )
        return - (distance_start_exit + distance_start_treasure)

    def edge_coordinates(self):
        """Compute the edge coordinates.

        Square coordinates are regarded as integer starting from 0.
        Edge coordinates starts at 0.5 and increase by 1.

        Returns:
          A `torch.Tensor` with the edge coordinates.
        """
        vertical_edge_coords = torch.tensor(
            list(product(
                torch.arange(self.size).tolist(),
                torch.arange(.5, self.size - 1).tolist()
            ))
        )
        horizontal_edge_coords = torch.tensor(
            list(product(
                torch.arange(.5, self.size - 1).tolist(),
                torch.arange(self.size).tolist()
            ))
        )
        return torch.cat(
            (vertical_edge_coords, horizontal_edge_coords),
            dim=0
        )


class MazeGenerator:
    """Tool for maze generation.

    Tool to generate maze through iterations based on the policy
    of the Deep Q-Network.
    """

    def __init__(self, policy_net, maze_initializer=Maze.init_for_dqn_training):
        """Create an instance of a maze generator.

        Args:
          policy_net:
            Pytorch Module which aims at predicting the next actions.
          maze_initializer:
            Function that initializes a maze.
        """
        self.policy_net = policy_net
        self.maze_initializer = maze_initializer
        self.device = next(policy_net.parameters()).device

    def generate_a_maze(self):
        """Initialize a maze and get the corresponding valid maze."""
        maze = self.maze_initializer()
        return self.corresponding_valid_maze(maze)

    def _next_maze(self, maze, random_policy=False):
        """Return the next maze and the corresponding atribute."""
        if random_policy:
            edge_indices = maze.edge_indices()
            action_index = edge_indices[torch.randint(0, len(edge_indices), (1,)).item()]
            q_values = None
        else:
            with torch.no_grad():
                q_values = self.policy_net.estimate_q_values(maze)
            action_index = torch.argmax(q_values).item()

        action = torch.zeros(maze.n_edges, dtype=maze._dtype)
        action[action_index] = 1

        next_maze = maze.clone()
        next_maze.withdraw_edges(action)
        return next_maze, q_values, action

    def corresponding_valid_maze(self, maze, record_steps=False, random_policy=False):
        """Loop until the maze is valid and return it."""
        steps = []
        while not maze.is_valid():
            next_maze, q_values, action = self._next_maze(maze, random_policy)
            if record_steps:
                steps.append(
                    (maze, q_values, action)
                )
            maze = next_maze
        if record_steps:
            return steps, next_maze
        else:
            return next_maze

    def efficiency(self, n_mazes=30, n_random_experiments=100, seed=0):
        """Estimate efficiency of the generator.

        Estimation of the ratio between the number of steps needed
        to get a valid maze with the policy and with randomly chosen
        steps.

        Args:
          n_mazes:
            Integer indicating the number of mazes used for
            the estimation.
          n_random_experiments:
            Integer indicating the number of runs
            to estimate the mean number of steps to get a valid
            maze using a random policy.
        """
        torch.manual_seed(seed)
        step_ratios = []
        for _ in range(n_mazes):
            maze = self.maze_initializer()
            steps, _ = self.corresponding_valid_maze(
                maze, record_steps=True
            )
            n_steps_policy = len(steps)
            n_steps_random = []
            for _ in range(n_random_experiments):
                steps, _ = self.corresponding_valid_maze(
                    maze, record_steps=True, random_policy=True
                )
                n_steps_random.append(len(steps))
            step_ratios.append(
                n_steps_policy / torch.tensor(n_steps_random).to(dtype=torch.float32).mean()
            )
            # return geometric mean of the ratios
        return len(step_ratios) / torch.sum(1 / torch.tensor(step_ratios))
