import gym
import gym_sokoban
import numpy as np
import torch

def get_mappings(env_name: str):
    """ from observation to state values """
    env = gym.make(env_name)
    last_observation = env.reset(render_mode='tiny_rgb_array')
    while last_observation.shape != (10, 10, 3):
        last_observation = env.reset(render_mode='tiny_rgb_array')
    last_observation = last_observation.mean(axis=2)
    mappings = {i:j for i, j
                in zip(last_observation.reshape(-1),
                        env.room_state.reshape(-1))
                }

    # 0:wall, 1:empty, 2:empty destination
    # 3:destination with box on, 4:box, 5:player
    return mappings

def calc_bellman(rewards,gamma):
    reward = np.array(rewards)
    gamma_column = gamma ** np.arange(len(reward))
    Bellman_value = [np.sum(reward[i:] * gamma_column[:len(reward) - i])
                     for i in range(len(reward))]
    Bellman_torch = torch.from_numpy(Bellman_value)
    return Bellman_torch

def get_env(name, max_steps: int = 1000, seed: int = None):
    env = gym.make(name)
    if seed: env.seed(seed)
    env.set_maxsteps(max_steps)
    return env


def save_game_data(single_game_df):
    pass


def save_train_statistics(train_statistics):
    pass


def print_game_stats(played_df, train_df):
    pass
