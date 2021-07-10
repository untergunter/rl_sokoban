import numpy as np
import gym
import gym_sokoban
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy_tuple import ArrayTupleConverter

class NaiveNet(nn.Module):

    def __init__(self, input_size, hidden_size, number_of_classes):
        super(NaiveNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, number_of_classes)

    def forward(self, x):
        x = torch.from_numpy(x).float().flatten()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def calc_step(self,x):
        prob_dist = self.forward(x)
        pred = int(torch.argmax(prob_dist))
        return pred

if __name__ == '__main__':

    """ set-up env """
    env = gym.make("Sokoban-v1")

    """ get env data for model """
    x_dim,y_dim = env.dim_room
    input_size = x_dim*y_dim
    hidden_size = x_dim*y_dim*2
    out_size = env.action_space.n

    """ set-up model """
    model = NaiveNet(input_size,hidden_size,out_size)

    converter = ArrayTupleConverter((x_dim,y_dim))

    for attempt in range(1):
        env.reset()
        states = set()
        state = env.room_state
        ob = tuple(state.flatten())
        states.add(ob)
        for action in range(100):
            state = env.room_state
            observation, reward, done, info = env.step(action)
            ob = tuple(state.flatten())
            states.add(ob)
        print(len(states))