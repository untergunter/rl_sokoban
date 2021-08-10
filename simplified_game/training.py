from utils import get_env,print_game_stats
from Agent import Agent
import pandas as pd

def main():
    name = "Sokoban-v1"
    n_steps = 1_000
    agent = Agent()
    env = get_env(name,n_steps)
    for time in range(1_000):
        single_game_df = agent.play_game(env)
        train_statistics = agent.train_model(single_game_df)
        # save_game_data(single_game_df)
        # save_train_statistics(train_statistics)
        print_game_stats(single_game_df,train_statistics)

if __name__ =='__main__':
    main()