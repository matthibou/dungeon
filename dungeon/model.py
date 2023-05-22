"""Deep learning models for Q-learning."""

import torch
import mlflow

from .maze import Maze, MazeGenerator


class DQNEdgeWiseFC(torch.nn.Module):
    """Deep Q Network based on fully connected layers.

    Computes an estimation of the Q function (corresponding to the Bellman equation).

    The estimation is done for each edge independently (edge-wise features).
    """

    n_estimations = 1

    def __init__(
        self,
        n_features=3,
        feature_kind='key_point_distances',
        layer_sizes=[10, 20, 10],
    ):
        """Initialize a network.

        Network architecture is based on 1d convolution
        with a kernel size of 1.
        """
        super(DQNEdgeWiseFC, self).__init__()
        self.feature_kind = feature_kind
        layers = []
        for in_, out_ in zip(
            [n_features, *layer_sizes],
            [*layer_sizes, self.n_estimations]
        ):
            layers.extend(
                [torch.nn.Conv1d(in_, out_, kernel_size=1), torch.nn.ReLU()]
            )
        self.net = torch.nn.Sequential(*layers[:-1])

    def forward(self, x):
        """Network predictions."""
        return self.net(x)

    def features(self, maze):
        """Compute distances between key groups for each edge.

        Using edge coordinates, this function computes the distances
        between the three key groups. By key group, we mean the group
        of reached indices from a key point. Here there is three key
        groups ("start", "exit", "treasure").

        Returns:
          `Torch.Tensor` with the minimum distance between edges and
          the key groups.

        """
        if self.feature_kind == 'key_point_distances':
            # edge wise features
            key_distances = torch.zeros(maze.n_key_points, maze.n_edges)
            for ii, key_point in enumerate(['start', 'exit', 'treasure']):
                key_distances[ii] = (
                    maze.ind_to_coord(  # coordinates of the reachable points from `key_point`
                        maze.reached_indices_from(key_point)
                    ).reshape(-1, 1, 2) -
                    maze.edge_coordinates(  # edge coordinates
                    ).reshape(1, maze.n_edges, 2)
                ).abs().sum(2).min(0).values  # manhattan distances

            # maze global features
            reached_coord_start = maze.reached_indices_from('start')
            distance_start_exit = maze.distance_point_group(
                reached_coord_start,
                maze.reached_indices_from('exit')
            )
            distance_start_treasure = maze.distance_point_group(
                reached_coord_start,
                maze.reached_indices_from('treasure')
            )
            return torch.cat((
                key_distances,
                torch.ones(1, maze.n_edges) * distance_start_exit,
                torch.ones(1, maze.n_edges) * distance_start_treasure
            ), dim=0)
        else:
            raise ValueError("Unknown kind of features!")

    def estimate_q_values(self, maze):
        """."""
        features = self.features(maze).unsqueeze(0).to(
            dtype=torch.float32, device=self.device
        )
        q_values = self.forward(features).squeeze()
        # fill with -infinity the action indices where there is no edge
        q_values[maze.edge_values.to(torch.bool)] = -torch.inf
        return q_values

    @property
    def device(self):
        """On which device the model is."""
        return next(self.parameters()).device


class MazeGeneratorWrapper(mlflow.pyfunc.PythonModel):
    """Utility class to serve a MazeGenerator as Python function."""

    def load_context(self, context):
        """Load parameters and weigths from artifacts."""
        state_dict = mlflow.pytorch.load_state_dict(
            context.artifacts["policy_net_state_dict"]
        )
        policy_net = DQNEdgeWiseFC(
            **state_dict.pop('model_params')
        )
        policy_net.load_state_dict(
            state_dict
        )
        self.maze_generator = MazeGenerator(
            policy_net
        )

    def predict(self, context, model_input):
        """Genrate a maze from inputs.

        Args:
          model_input:
            Dictionary with the fields 'edge_values' and 'key_positions'.
            The two should be `numpy.array` of type respectively
            np.int32 and np.int64.

        Returns:
            Dictionary with the same fields as the inputs
            associated to the valid maze.
        """
        if model_input['edge_values'].size > 0:
            m = Maze.from_dict(model_input)
            m_valid = self.maze_generator.corresponding_valid_maze(m)
            return m_valid.to_dict()
        else:
            return self.maze_generator.generate_a_maze().to_dict()
