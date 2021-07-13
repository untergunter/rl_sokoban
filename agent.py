import numpy as np


class Agent:

    def __init__(self, model, n_states: int):
        self.model = model
        self.n_states = n_states


    def pred(self,state,action):
        prediction = self.model(state, action)
        return prediction

    def select_action(self, current_state):

        best_action = None
        best_expected_score = -np.inf
        for action in range(1,self.n_states):
            expected_score = self.pred(current_state,action)
            if expected_score > best_expected_score:
                best_expected_score = expected_score
                best_action = action

        return best_action
