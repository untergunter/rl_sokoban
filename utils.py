import numpy as np
import pandas as pd
from spinner import read_and_fix_state

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

def agent_area(raw_state):
    agent_y,agent_x = np.argwhere(raw_state==142.66666667)[0]
    above = raw_state[agent_y-1,agent_x]
    under = raw_state[agent_y+1,agent_x]
    left = raw_state[agent_y,agent_x-1]
    right = raw_state[agent_y, agent_x + 1]
    result = np.array([above,under,left,right])
    return result

