"""Script to build a maze generator through a Q-learning procedure."""

import torch
import numpy as np

import mlflow

from dungeon.maze import Maze, MazeGenerator
from dungeon.train import DQNTrainer
from dungeon.model import MazeGeneratorWrapper


params = {
    'replay_buffer_size': 3000,
    'epsilon': (.9, .05, 2000),
    'n_episodes': 1500,
    'lr': 1e-4,
    'tau': .05,
    'n_batch': 64,
    'gamma': .99,
    'clip_grad_value': 100,
    'amsgrad': True,
    'device': 'cpu',
    'seed': 1515,
    'model': {
        'feature_kind': 'key_point_distances',
        'n_features': 5,
        'layer_sizes': [10, 20, 20, 10]
    },
    'init_args': {
        'n_edge_withdrawn': 2,
        'distance_between_key_points': 3
    }
}


with mlflow.start_run():
    trainer = DQNTrainer(params)
    trainer.run()
    # initialize a MazeGenerator instance
    maze_generator = MazeGenerator(
        trainer.policy_net,
        Maze.init_for_dqn_training
    )
    # compute effiency which is compute a mean ratio 
    # between random policy and trained policy (the smaller the better).
    mlflow.log_metric(
        'efficiency', maze_generator.efficiency(50, 200, seed=params['seed'])
    )
    # save model
    m = Maze()
    state_dict = trainer.policy_net.state_dict()
    state_dict['model_params'] = params['model']
    mlflow.pytorch.log_state_dict(
        state_dict,
        artifact_path="policy_network"
    )
    model_info = mlflow.pyfunc.log_model(
        artifact_path='model',
        artifacts={
            'policy_net_state_dict': mlflow.get_artifact_uri('policy_network')
        },
        python_model=MazeGeneratorWrapper(),
        signature=mlflow.models.infer_signature(
            m.to_dict(),
            m.to_dict()
        )
    )
    print(f'model_uri: {model_info.model_uri}')
