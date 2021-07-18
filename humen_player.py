import numpy as np
import gym
import gym_sokoban
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

env = gym.make("Sokoban-v1")
env.seed(0)
last_observation = env.reset(render_mode='rgb_array')
print(env.room_state)
moves = {'a':3,'w':1,'s':2,'d':4}
print(env.get_action_lookup())
while True:
    img = Image.fromarray(last_observation, 'RGB')
    plt.imshow(img)
    plt.show()
    move = input('awsd').lower()
    if not move in moves:
        print('invalid move')
    else:
        last_observation, reward, done, info = env.step(moves[move], observation_mode='rgb_array')