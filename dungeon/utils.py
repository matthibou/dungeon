"""Utility functions for training and serving MazeGEnerator DQN."""


def flaten_dict(dict_):
    """Unfold nested dictionary (only 1st order)."""
    dict__ = {}
    for k, v in dict_.items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                dict__[f'{k}__{k_}'] = v_
        else:
            dict__[k] = v
    return dict__
