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

    def __init__(self, input_size, hidden_size):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x,action):
        x = torch.cat([torch.from_numpy(x).float().flatten(),torch.tensor([action])])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    model = DeepNet(input_size,hidden_size)


    """ set up agents """
    agent = Agent(model,action_space)
    """ set up helpers """

    observation = env.reset()
    state = env.room_state
    """ used to reduce the image to minimum """
    observation_normalizer = TransformToTiny(observation, state)

    """ function validation """
    all_types = np.unique(observation_normalizer(observation))
    unique_observations = set(number for number in all_types)

    """ used for hashing states """
    converter = ArrayTupleConverter((x_dim,y_dim))

    """ training helpers """
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    results_over_iterations = {column:[] for column in
                               ['repeated_states',
                                'sum_reward',
                                'attempt',
                                'victory',
                                'total steps',
                                'actions']
                               }
    """ training """

    for attempt in tqdm(range(200)):
        state = env.reset()
        victory = False
        repeated_states = 0
        sum_reward = 0
        all_obs = defaultdict(set)
        current_state = observation_normalizer(state)
        action_list = []

        for iteration in range(200):
            raw_prediction = model(current_state)
            action = int(torch.argmax(raw_prediction))
            action_list.append(action)
            observation, reward, done, info = env.step(action)
            sum_reward += reward
            new_state = observation_normalizer(observation)

            if np.all(current_state==new_state):

                repeated_states +=1
                ob = tuple(state.flatten())
                all_obs[ob].add(action)
                bad_moves = all_obs[ob]
                if len(bad_moves) == action_space:
                    """ the game started in an unmovable position """
                    break
                better_value = best_known_move_vector(action_space, bad_moves)
                loss = loss_fn(raw_prediction,better_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done is True:
                victory = True
                break

            current_state = new_state

        results_over_iterations['repeated_states'].append(repeated_states)
        results_over_iterations['sum_reward'].append(sum_reward)
        results_over_iterations['attempt'].append(attempt)
        results_over_iterations['victory'].append(victory)
        results_over_iterations['total steps'].append(iteration)
        results_over_iterations['actions'].append(action_list)

    results_df = pd.DataFrame(results_over_iterations)
    results_df.to_csv(f'results/{test_name}.csv', index=False)

