import numpy as np
import pandas as pd
import gym
import gym_sokoban
from tqdm import tqdm
import torch
import glob
from spinner import read_and_fix_state
from random import shuffle

from models import DeepQNet

def add_Bellman_equation(df,gamma):
    reward = df['reward'].values
    gamma_column = gamma**np.arange(len(reward))
    Bellman_value = [np.sum(reward[i:]*gamma_column[:len(reward) - i]) for i in range(len(reward))]
    df['Bellman_value'] = Bellman_value
    return df

def prep_transform_state(path):
    file = read_and_fix_state(path)
    state = file['state'].tolist()[-2] # -2 got all types of tiles for sure
    values_in_state = np.unique(state)
    normalized = [i/len(values_in_state) for i in range(len(values_in_state))]
    mapper = {original:normal_value for original, normal_value in zip(values_in_state,normalized)}

    def transform_state(state):
        for original,new in mapper.items():
            state[state == original] = new
        return state

    return transform_state


def train_step_to_max_score(model,criterion,optimizer):
    training_set = glob.glob('/home/ido/data/idc/reinforcement learning/course/final_project/repo/training_set/*')
    state_normalizer = prep_transform_state(training_set[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    env = gym.make("Sokoban-v1")
    env.seed(1024)
    env.set_maxsteps(201)
    gamma = 0.9

    for all_files_iterations in range(100):
        shuffle(training_set)
        for single_file in tqdm(training_set):
            raw_file = read_and_fix_state(single_file)
            file = add_Bellman_equation(raw_file,gamma)
            boards = [state_normalizer(state)[1:9,1:9].reshape(-1) for state in file['state']]
            actions = file['action']
            rewards = file['Bellman_value']

            for state,action,reward in zip(boards,actions,rewards):
                state_and_action = torch.tensor(np.concatenate([state,np.array([action])])).float().to(device)
                reward = torch.tensor([reward])  # Bellman value reward

                optimizer.zero_grad()
                predicted_q = model(state_and_action)

                y = reward.to(device)
                loss = criterion(predicted_q, y)

                # backwards
                loss.backward()
                optimizer.step()

                # make sure not to take the wrong step
                for wrong_action in model.actions:
                    bad_reward = torch.tensor([-1])
                    y = bad_reward.to(device)
                    if wrong_action == action: continue # dont want to bring it down
                    state_and_action = torch.tensor(np.concatenate(
                        [state, np.array([wrong_action])])).float().to(device)
                    predicted_q = model(state_and_action)
                    optimizer.zero_grad()
                    loss = criterion(predicted_q, y)

                    # backwards
                    loss.backward()
                    optimizer.step()


        current_state = env.reset(render_mode='tiny_rgb_array')
        current_state = current_state.mean(axis=2)
        while current_state.shape != (10, 10):
            """ initialization issue """
            current_state = env.reset(render_mode='tiny_rgb_array')
            current_state = current_state.mean(axis=2)
        rewards = []
        done = False
        # test
        for turn in range(200):
            if done:break
            current_state = state_normalizer(current_state)[1:9, 1:9].reshape(-1)
            action = model.calculate_next_step(current_state,device=device)
            current_state, reward, done, info = env.step(action, observation_mode='tiny_rgb_array')
            rewards.append(reward)

        mean_reward = np.mean(np.array(rewards))
        print(f'epoc {all_files_iterations} mean env score = {mean_reward} {"won" if done else "lost"}')

if __name__ == '__main__':
    model = DeepQNet(8*8 + 1 , 8*8+1,(1,2,3,4))
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_step_to_max_score(model,criterion,optimizer)