from brute_solve import BruteSolve
import gym
import gym_sokoban
import pandas as pd
import glob
from tqdm import tqdm
from dask.distributed import Client
from dask import delayed,compute

# all_files = glob.glob('training_set/*.csv')
# if len(all_files) ==0:
#     next_number = 0
# else:
#     files_num = [int(f.split('/')[-1].split('.')[0].split('_')[-1]) for f in all_files]
#     next_number = max(files_num) + 1
#
# env = gym.make("Sokoban-v1")
# env.seed(next_number)
#
# for i in tqdm(range(100)):
#     result_dict = {i: [] for i in ('state', 'reward', 'done', 'info','action')}
#     current_state = env.reset(render_mode='tiny_rgb_array')
#     current_state = current_state.mean(axis=2)
#     while current_state.shape != (10, 10):
#         """ initialization issue """
#         current_state = env.reset(render_mode='tiny_rgb_array')
#         current_state = current_state.mean(axis=2)
#     solution = BruteSolve(env.room_state).solution
#     for action in solution:
#         observation, reward, done, info = env.step(action, observation_mode='tiny_rgb_array')
#         result_dict['state'].append(current_state)
#         result_dict['action'].append(action)
#         result_dict['reward'].append(reward)
#         result_dict['done'].append(done)
#         result_dict['info'].append(info)
#     results_df = pd.DataFrame(result_dict)
#     results_df.to_csv(f'training_set/seed_{next_number}.csv',index=False)
#     next_number+=1

def generate_solution(next_number:int):
    env = gym.make("Sokoban-v1")
    env.seed(next_number)
    result_dict = {i: [] for i in ('state', 'reward', 'done', 'info', 'action')}
    current_state = env.reset(render_mode='tiny_rgb_array')
    current_state = current_state.mean(axis=2)
    while current_state.shape != (10, 10):
        """ initialization issue """
        current_state = env.reset(render_mode='tiny_rgb_array')
        current_state = current_state.mean(axis=2)
    solution = BruteSolve(env.room_state).solution
    for action in solution:
        observation, reward, done, info = env.step(action, observation_mode='tiny_rgb_array')
        result_dict['state'].append(current_state)
        result_dict['action'].append(action)
        result_dict['reward'].append(reward)
        result_dict['done'].append(done)
        result_dict['info'].append(info)
    results_df = pd.DataFrame(result_dict)
    results_df.to_csv(f'training_set/seed_{next_number}.csv', index=False)

if __name__=="__main__":
    all_files = glob.glob('training_set/*.csv')
    if len(all_files) == 0:
        next_number = 0
    else:
        files_num = [int(f.split('/')[-1].split('.')[0].split('_')[-1]) for f in all_files]
        next_number = max(files_num) + 1

    with Client(n_workers=5,threads_per_worker=1,memory_limit='32GB') as client:
        tasks = [delayed(generate_solution)(i) for i in range(next_number+1,next_number+100)]
        compute(*tasks)