"""Training procedure for a Deep Q-learning.

Train a policy network to match the Bellman equation.
From an initial maze the policy network dertermine which edge
to withdraw to achieve the goal of maximizing the sum of the rewards.
The processus stops when there exits a path between the "start"
point and the "exit" point and another path between "start" point
and "treasure" point.


References:
* https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
* https://huggingface.co/learn/deep-rl-course/unit3/deep-q-network?fw=pt
"""
import torch
import random
import math

from tqdm import tqdm
from collections import deque
from itertools import count

import mlflow

from .model import DQNEdgeWiseFC
from .maze import Maze, MazeGenerator
from .utils import flaten_dict


class DQNTrainer:
    """Implement a basic version of the Deep Q-learning procedure.

    Inspired from the Deep Q-learning procedure described
    in the following link:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(
        self,
        params
    ):
        """Create an training procedure instance with the user input variable params."""
        self.params = params
        self.device = params['device']
        # initialization
        torch.manual_seed(params['seed'])
        self.replay_buffer = deque(
            [], maxlen=params['replay_buffer_size']
        )
        self.target_net = DQNEdgeWiseFC(**self.params['model']).to(device=self.device)
        self.policy_net = DQNEdgeWiseFC(**self.params['model']).to(device=self.device)
        self.policy_net.load_state_dict(
            self.target_net.state_dict()
        )
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=params['lr'],
            amsgrad=params['amsgrad']
        )
        self.loss = torch.nn.SmoothL1Loss()
        self.step = 0

        # logging
        self.mlflow_run = mlflow.active_run()
        mlflow.log_params(
            flaten_dict(params)
        )

    def random_action(self):
        """Dertimine if the next action will be random."""
        eps_start, eps_end, eps_decay = self.params['epsilon']
        return random.random() < eps_end + (eps_start - eps_end) * math.exp(- self.step / eps_decay)

    def next_action(self, maze):
        """Determine next action."""
        if self.random_action():
            edge_indices = maze.edge_indices()
            action_index = edge_indices[
                torch.randint(0, len(edge_indices), (1,)).item()
            ]
        else:
            with torch.no_grad():
                q_values = self.policy_net.estimate_q_values(maze)
            action_index = torch.argmax(q_values).item()

        action = torch.zeros(maze.n_edges, dtype=maze._dtype)
        action[action_index] = 1

        next_maze = maze.clone()
        next_maze.withdraw_edges(action)
        return action_index, next_maze

    def optimize_policy_net(self):
        """Do a grad step on the policy network."""
        if len(self.replay_buffer) < self.params['n_batch']:
            return

        # Select a batch from the memory buffer
        batch = random.sample(self.replay_buffer, self.params['n_batch'])
        batch = list(zip(*batch))

        # Unfold stored information
        features = torch.stack(batch[0])
        actions = torch.stack(batch[1])
        next_features = torch.stack(batch[2])
        next_edges = torch.stack(batch[3])
        rewards = torch.stack(batch[4])

        # Compute Bellman equation terms
        Q_values = self.policy_net(features).squeeze(1).gather(1, actions)
        with torch.no_grad():
            next_Q_values = self.target_net(next_features).squeeze(1)
        # fill with -infinity the action indices where there is no edge
        next_Q_values[next_edges] = -torch.inf
        max_next_Q_values = next_Q_values.max(1).values.unsqueeze(1)
        expected_Q_values = max_next_Q_values * self.params['gamma'] + rewards

        # compute loss on non final state (no paths)
        non_final_next_states = rewards != 0
        batch_loss = self.loss(
            Q_values[non_final_next_states],
            expected_Q_values[non_final_next_states]
        )

        # Step of optimization to minimize the two different
        # terms of Bellman equation
        self.optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_value_(
            self.policy_net.parameters(),
            self.params['clip_grad_value']
        )
        self.optimizer.step()
        self.step += 1

        # log
        mlflow.log_metric('loss', batch_loss.item(), step=self.step)

    def update_target_net(self):
        """Avoid learning instability.

        Update the target network using the optimized policy network.
        """
        tau = self.params['tau']
        target_net_state_dict = self.target_net.state_dict()
        for key, policy_net_weights in self.policy_net.state_dict().items():
            target_net_state_dict[key] = \
                policy_net_weights * tau + target_net_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def run(self, seed=2023):
        """Training procedure loop."""
        for i_episode in tqdm(
            range(self.params['n_episodes'])
        ):
            # init a new maze
            maze = Maze.init_for_dqn_training(**self.params['init_args'])
            # loop while the maze is not valid
            for t in count():
                action, next_maze = self.next_action(maze)
                # store features, actions and rewards for further training steps
                self.replay_buffer.append(
                    (
                        self.policy_net.features(maze).to(  # features from current maze
                            dtype=torch.float32, device=self.device
                        ),
                        torch.tensor([action]).to(  # chosen action
                            dtype=torch.int64, device=self.device
                        ),
                        self.policy_net.features(next_maze).to(  # features from next maze
                            dtype=torch.float32, device=self.device
                        ),
                        next_maze.edge_values.to(  # edges of the next maze
                            dtype=torch.bool
                        ),
                        torch.tensor([next_maze.reward()]).to(   # rewards
                            dtype=torch.float32, device=self.device
                        )
                    )
                )
                maze = next_maze
                self.optimize_policy_net()
                self.update_target_net()
                # log effeciency metric
                if next_maze.is_valid():
                    mlflow.log_metric(
                        'n_steps_to_validity', t, step=self.step
                    )
                    break
