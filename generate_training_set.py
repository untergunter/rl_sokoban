from brute_solve import BruteSolve
import gym
import gym_sokoban
import pandas as pd
import glob


all_files = glob.glob('training_set/*.csv')
if len(all_files) ==0:
    next_number = 0
else:
    files_num = [int(f.split('/')[-1].split('.')[0].split('_')[-1]) for f in all_files]
    next_number = max(files_num) + 1

env = gym.make("Sokoban-v1")
env.seed(next_number)
result_dict = {'state':[], 'steps':[]}
for i in range(100):
    current_state = env.reset(render_mode='tiny_rgb_array')
    current_state = current_state.mean(axis=2)
    while current_state.shape != (10, 10):
        """ initialization issue """
        current_state = env.reset(render_mode='tiny_rgb_array')
        current_state = current_state.mean(axis=2)
    solution = BruteSolve(env.room_state).solution
    result_dict['state'].append(env.room_state)
    result_dict['steps'].append(solution)
results_df = pd.DataFrame(result_dict)
results_df.to_csv(f'training_set/seed_{next_number}.csv',index=False)

