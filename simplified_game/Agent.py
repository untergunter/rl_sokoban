import numpy as np
import pandas as pd
import random
import torch
from utils import get_mappings,get_env
from simulated_environment import SimulateV1

class Agent():

    def __init__(self,env_name,model,search_rate:int=0.5):
        self.model = model
        self.mappings = get_mappings(env_name)
        self.simulated_environment = SimulateV1()
        self.search_rate = search_rate

    def train_model(self,game_df,gamma:float = 0.8):
        pass

    def play_game(self, env):
        """ play a single game, returns the training output"""
        last_state = np.array([0])
        while last_state.shape != (10, 10,3):  # bad restart
            last_state = env.reset(render_mode='tiny_rgb_array')
        done = False
        states_actions, rewards = [], []
        while not done:
            selected_state_action = self.__select_next_step(last_state)
            states_actions.append(selected_state_action)
            last_state, reward, done, env = self.__get_to_and_act(env, last_state, selected_state_action)
            rewards.append(reward)
        game_results = pd.DataFrame({'ipnuts': states_actions, 'reward': rewards})
        return game_results

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
        return action

    def __select_highest_valued_action(self,state,possible_actions):
        state = torch.from_numpy(state.flatten())
        all_steps = torch.stack([torch.cat(
                                [state,torch.tensor([action])])
                                for action in possible_actions ])
        expected_return = self.model(all_steps)
        selected_index = int(torch.argmax(expected_return))
        selected_action = possible_actions[selected_index]
        return selected_action

    def __get_to_and_act(self, env, state, location_action):
        state = self.__normalize(state)
        get_to_y,get_to_x,action = location_action
        steps_to_location = self.simulated_environment.\
            steps_to_get_to(state,get_to_y,get_to_x)

        for walk in steps_to_location:
            _, _, _, env = env.step(walk)

        last_state, reward, done, env = env.step(action)
        return last_state, reward, done, env






if __name__ == '__main__':
    agent = Agent("PushAndPull-Sokoban-v0",None)
    env = get_env("PushAndPull-Sokoban-v0")
    agent.play_game(env)