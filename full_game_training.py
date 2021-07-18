import numpy as np
from collections import defaultdict
import gym
import gym_sokoban
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import Agent
from numpy_tuple import ArrayTupleConverter
from observation_reshaper import TransformToTiny

test_name = 'long term score'

class DeepNet(nn.Module):

    def __init__(self, input_size, hidden_size,number_of_outputs):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.number_of_outputs = number_of_outputs

    def forward(self, x:torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_all_moves(self,step_less_state):
        """ create a 2d tensor each row = concat(state,step)"""
        all_steps = torch.stack([torch.cat([step_less_state.flatten(),torch.tensor([action])])
                                 for action in range(self.number_of_outputs)])

        """ output is [q_val,q_val...]"""
        expected_q_value = self.forward(all_steps)

        return expected_q_value

if __name__ == '__main__':

    """ set-up env """
    env = gym.make("Sokoban-v1")
    env.seed(0)

    """ get env data for model """
    x_dim,y_dim = env.dim_room
    input_size = x_dim*y_dim + 1
    hidden_size = x_dim*y_dim*2
    action_space = env.action_space.n

    """ set-up model """
    model = DeepNet(input_size,hidden_size,action_space)


    """ set up agents """
    agent = Agent(model,action_space)
    """ set up helpers """

    """ training helpers """
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    """ training """
    for attempt in tqdm(range(200)):
        current_state = env.reset(render_mode='tiny_rgb_array')
        current_state = current_state.mean(axis=2)
        while current_state.shape !=(10,10):
            """ initialization issue """
            current_state = env.reset(render_mode='tiny_rgb_array')
            current_state = current_state.mean(axis=2)
        current_state = torch.from_numpy(current_state).float().flatten()
        env.set_maxsteps(201)
        victory = False
        iterations_history = []

        for iteration in range(200):
            raw_prediction = model.forward_all_moves(current_state)
            chosen_action = int(torch.argmax(raw_prediction))
            new_state, reward, done, info = env.step(chosen_action)
            new_state = torch.from_numpy(new_state).float().flatten()
            action_had_no_use = torch.all(current_state==new_state)
            iterations_history.append([current_state, raw_prediction, chosen_action,
                                       -1 if action_had_no_use else reward, action_had_no_use, done])
            if action_had_no_use:
                """ step has no use! we know it is a mistake"""
                bad_input = torch.cat([current_state,torch.tensor([chosen_action])])
                penalty = torch.tensor([-1])
                raw_prediction = model(bad_input)
                loss = loss_fn(raw_prediction,penalty)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done is True:
                victory = True
                break

            current_state = new_state

        # create fixed reword
        raw_rewards_array = torch.tensor([ ob[3] for ob in iterations_history])
        gammas = torch.tensor([ 0.9**i for i in range(len(iterations_history))])
