import numpy as np
import pandas as pd
import random
import torch
from utils import get_mappings,get_env,calc_bellman
from simulated_environment import SimulateV1
from torch.utils.data import TensorDataset,DataLoader,RandomSampler
from datetime import datetime

class Agent():

    def __init__(self,env_name,model,search_rate:int=0.9):
        self.model = model
        self.mappings = get_mappings(env_name)
        self.simulated_environment = SimulateV1()
        self.search_rate = search_rate

    def train_model(self,states_actions,rewards,
                    gamma:float = 0.8,batch_size:int=16):

        states_actions = torch.tensor([torch.from_numpy(i)
                                      for i in states_actions])
        Bellman_rewards = calc_bellman(rewards,gamma)
        rewards = torch.tensor([torch.from_numpy(i) for i in rewards])

        dataset = TensorDataset(states_actions,rewards,Bellman_rewards)

        loader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=batch_size
        )

        loss_fn = torch.nn.MSELoss()
        learning_rate = 1e-3
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        total_loss = 0
        for time in range(2):
            for batch in loader:
                state,reward,bellman = batch

                optimizer.zero_grad()
                outputs = self.model(state)
                loss_1 = loss_fn(outputs, reward)
                loss_2 = loss_fn(outputs, bellman)
                total_loss += loss_1.item()
                total_loss += loss_2.item()
                loss_1.backward()
                loss_2.backward()
                optimizer.step()
        print(f"mean batch loss:{total_loss/len(dataset)}")

    def play_game(self, env):
        """ play a single game, returns the training output"""
        last_state = np.array([0])
        while last_state.shape != (10, 10,3):  # bad restart
            last_state = env.reset(render_mode='tiny_rgb_array')
        done = False
        states_actions, rewards = [], []
        while not done:
            print(f'finding action {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            action,state_action = self.__select_next_step(last_state)
            states_actions.append(state_action)
            print(f'taking actions {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            last_state, reward, done, env = self.__get_to_and_act(env, last_state, action)
            rewards.append(reward)
            print(f'took a step {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        return states_actions,rewards

    def __normalize(self,state):
        normalized_state = np.copy(state)
        normalized_state = normalized_state.mean(axis=2)
        for image_value, state_value in self.mappings.items():
            normalized_state[normalized_state == image_value] = state_value
        normalized_state = normalized_state.astype(int)
        return normalized_state


    def __select_next_step(self, state):
        normalized_state = self.__normalize(state)
        possible_actions = self.simulated_environment.\
            get_possible_actions(normalized_state) # [(y,x,action),...(y,x,action)]
        if len(possible_actions) == 1:
            action = possible_actions[0]
        else:
            chance_to_random_action = random.random()
            if chance_to_random_action < self.search_rate:
                action = random.choice(possible_actions)
            else:
                action = self.__select_highest_valued_action(state,possible_actions)
        return action,np.concatenate([state[1:-1,1:-1].flatten(),np.array(action)])

    def __select_highest_valued_action(self,state,possible_actions):
        state = torch.from_numpy(state[1:-1,1:-1].flatten())
        all_steps = torch.stack([torch.cat(
                                [state,torch.tensor([action])])
                                for action in possible_actions])
        expected_return = self.model(all_steps)
        selected_index = int(torch.argmax(expected_return))
        selected_action = possible_actions[selected_index]
        return selected_action

    def __get_to_and_act(self, env, state, location_action):
        state = self.__normalize(state)
        get_to_y,get_to_x,action = location_action
        steps_to_location = self.simulated_environment.\
            steps_to_get_to(state,get_to_y,get_to_x)
        done = False
        for walk in steps_to_location:
            _, reward, done, _ = env.step(walk)
            print(f'walked {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            if done:
                last_state = self.__normalize(env.render(mode='tiny_rgb_array'))
                break

        if not done:
            _, reward, done, _ = env.step(action)
            last_state = self.__normalize(env.render(mode='tiny_rgb_array'))
        return last_state, reward, done, env






if __name__ == '__main__':
    agent = Agent("PushAndPull-Sokoban-v0",None)
    env = get_env("PushAndPull-Sokoban-v0")
    agent.play_game(env)