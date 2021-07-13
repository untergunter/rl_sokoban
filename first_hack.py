import numpy as np
from collections import defaultdict
import gym
import gym_sokoban
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy_tuple import ArrayTupleConverter
from observation_reshaper import TransformToTiny

test_name = 'no effect penalty, got box reward'

def best_known_move_vector(vector_length,bad_moves_indexes:set):
    uniform_value = 1 / (vector_length-len(bad_moves_indexes))
    best_known_moves = torch.tensor([0 if idx in bad_moves_indexes else uniform_value for idx in range(vector_length)])
    return best_known_moves


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
    env.seed(0)

    """ get env data for model """
    x_dim,y_dim = env.dim_room
    input_size = x_dim*y_dim
    hidden_size = x_dim*y_dim*2
    out_size = env.action_space.n

    """ set-up model """
    model = NaiveNet(input_size,hidden_size,out_size)

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
        last_observation = env.reset(render_mode='tiny_rgb_array')
        last_observation = last_observation.mean(axis=2)
        while last_observation.shape !=(10,10):
            last_observation = env.reset(render_mode='tiny_rgb_array')
            last_observation = last_observation.mean(axis=2)
        env.set_maxsteps(201)
        victory = False
        repeated_states = 0
        sum_reward = 0
        all_obs = defaultdict(set)
        action_list = []

        for iteration in range(200):
            raw_prediction = model(last_observation)
            action = int(torch.argmax(raw_prediction))
            action_list.append(action)
            observation, reward, done, info = env.step(action,observation_mode='tiny_rgb_array')
            observation = observation.mean(axis=2)
            sum_reward += reward

            if np.all(observation==last_observation):

                repeated_states +=1
                ob = tuple(last_observation.flatten())
                all_obs[ob].add(action)
                bad_moves = all_obs[ob]
                if len(bad_moves) == out_size:
                    """ the game started in an unmovable position """
                    print(last_observation)
                    break
                better_value = best_known_move_vector(out_size,bad_moves)
                loss = loss_fn(raw_prediction,better_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            """ teach model to do what it did to get a better reward """
            if reward > 0:
                better_value = torch.zeros(size=raw_prediction.shape)
                better_value[action] = 1
                loss = loss_fn(raw_prediction,better_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done is True:
                victory = True
                break

            last_observation = observation

        results_over_iterations['repeated_states'].append(repeated_states)
        results_over_iterations['sum_reward'].append(sum_reward)
        results_over_iterations['attempt'].append(attempt)
        results_over_iterations['victory'].append(victory)
        results_over_iterations['total steps'].append(iteration)
        results_over_iterations['actions'].append(action_list)

    results_df = pd.DataFrame(results_over_iterations)
    results_df.to_csv(f'results/{test_name}.csv', index=False)
