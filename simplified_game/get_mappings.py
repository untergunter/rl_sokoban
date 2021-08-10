import numpy as np
import gym
import gym_sokoban

def get_mapings(env_name:str):
    env = gym.make(env_name)
    last_observation = env.reset(render_mode='tiny_rgb_array')
    while last_observation.shape != (10, 10, 3):
        last_observation = env.reset(render_mode='tiny_rgb_array')
    last_observation = last_observation.mean(axis=2)


    mappings = {(i,j) for i,j in zip(env.room_state.reshape(-1),
                                     last_observation.reshape(-1))}

    # 0:wall, 1:empty, 2:empty destination, 3:destination with box on, 4:box, 5:player

    return mappings

if __name__ == '__main__':
    get_mapings("PushAndPull-Sokoban-v0")