from utils import get_env,print_game_stats
from Agent import Agent
from models import FCnet
import pandas as pd

def main():
    name = "Sokoban-v1"
    n_steps = 1_000
    model = FCnet(8*8+3,100)
    agent = Agent(name,model)
    env = get_env(name,n_steps)
    for time in range(1_000):
        states_actions,rewards = agent.play_game(env)
        train_statistics = agent.train_model(states_actions,rewards)
        # save_game_data(single_game_df)
        # save_train_statistics(train_statistics)
        # print_game_stats(rewards)
        agent.search_rate -= 0.005

if __name__ =='__main__':
    main()