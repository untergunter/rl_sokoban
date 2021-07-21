import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

def turn_solution_left(solution):
    rotate_dict = {1:3,3:2,2:4,4:1}
    rotated = [rotate_dict[direction] for direction in solution]
    return rotated

def turn_states_left(states):
    rotated = [np.rot90(state) for state in states]
    return rotated

def read_state(raw_string_array):
    state_as_np = np.array([float(i) for i in
                 raw_string_array.replace('[', ' ').replace(']', ' ').replace('\n', ' ').split(' ')
                 if i != '']).reshape((10,10))
    return state_as_np

def read_clean(raw_states_list):
    as_np = [read_state(i) for i in raw_states_list]
    return as_np

def turn_states_90(column_of_np_states):
    rotated = [np.rot90(i) for i in column_of_np_states]
    return rotated

def read_and_fix_state(path):
    df = pd.read_csv(path)
    df['state'] = read_clean(df['state'])
    return df

def spin_90(df):
    df['state'] = turn_states_90(df['state'])
    df['action'] = turn_solution_left(df['action'])
    return df

def create_spin_offs():
    all_raw_files = glob.glob('training_set/*.csv')
    for raw_file in tqdm(all_raw_files):
        file_name = raw_file.split('/')[-1].split('.')[0]
        if file_name.count('_')==1: #do not duplicate duplications
            file = read_and_fix_state(raw_file)
            for spin in range (1,4):
                file = spin_90(file)
                outpath = f'training_set/{file_name}_{spin}.csv'
                file.to_csv(outpath,index=False)

if __name__ == '__main__':
    create_spin_offs()