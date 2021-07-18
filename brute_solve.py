import numpy as np


# moves: up->1, down->2,left->3,right->4
# board: 0->wall, 1->empty, 2->destination
# , 3->destination with box, 4->box, 5->player


class BruteSolve:
    def __init__(self, basic_state):

        # int -> tuple(np.array)
        self.state_by_id = {0:basic_state}

        # tuple(np.array) -> state_index, previous_state_index,move_from_previous_to_me
        self.state_data = {tuple(basic_state.reshape(-1)):(0,None,None)}

        self.current_states = [basic_state]
        self.next_states = list()
        self.destinations = {tuple(i) for i in np.argwhere(basic_state == 2)}
        self.move_functions = {1:self.move_up,
                               2:self.move_down,
                               3:self.move_left,
                               4:self.move_right}
        self.solution = self.solve()

    def solve(self):
        while True:
            self.next_states = list()
            for state in self.current_states:
                for move_number in self.move_functions:
                    next_state = self.apply_move(move_number,state)
                    next_state_as_tuple = tuple(next_state.reshape(-1))
                    if next_state_as_tuple not in self.state_data:
                        self.add_state(next_state,next_state_as_tuple,state,move_number)

                    if self.game_over_victory(next_state):
                        solution = self.rewind(next_state)
                        return solution
            self.current_states = self.next_states

    def add_state(self,state,state_tuple, previous_state,move):

        self.next_states.append(state)

        state_id = len(self.state_by_id)
        self.state_by_id[state_id] = state

        previous_state_id = self.state_data[tuple(previous_state.reshape(-1))][0]
        self.state_data[state_tuple] = (state_id,previous_state_id,move)

    def rewind(self,winning_state):
        steps_to_victory = []
        current = winning_state
        while current is not None:
            state_tuple = tuple(current.reshape(-1))
            _,previous_state_id,move = self.state_data[state_tuple]
            if move is None: break # state 0
            steps_to_victory.append(move)
            current = self.state_by_id[previous_state_id]

        steps_to_victory.reverse()
        return steps_to_victory


    def move_down(self,state):
        state = np.rot90(state,2)
        state = self.move_up(state)
        state = np.rot90(state,2)
        return state


    def move_right(self,state):
        state = np.rot90(state)
        state = self.move_up(state)
        state = np.rot90(state,-1)
        return state

    def move_left(self,state):
        state = np.rot90(state,-1)
        state = self.move_up(state)
        state = np.rot90(state)
        return state

    def move_up(self, state):

        player_y, player_x = np.argwhere(state == 5)[0]
        above = state[player_y - 1, player_x]

        if above == 0:  # wall
            return state

        elif above in (1, 2):  # empty/destination
            state[player_y - 1, player_x] = 5
            state[player_y, player_x] = 1
            return state

        elif above in (3, 4):  # box on destination / box
            above_2 = state[player_y - 2, player_x]
            if above_2 in (1, 2):  # empty/destination
                state[player_y - 2, player_x] = 4 if above_2 == 1 else 3
                state[player_y - 1, player_x] = 5
                state[player_y , player_x] = 1
            return state

        error_message = f'dont know how to deal with above = {above}'
        raise Exception(error_message)

    def apply_move(self, move, state):
        state = state.copy()
        movment = self.move_functions[move]
        state = movment(state)
        state = self.fix_destinations(state)
        return state

    def fix_destinations(self,state):
        for y,x in self.destinations:
            if state[y,x] == 1:
                state[y, x] = 2
        return state

    def game_over_victory(self,state):
        for y,x in self.destinations:
            if state[y,x] != 3:
                return False
        return True

if __name__ == '__main__':
    state = np.array([[0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,1,1,1,1,1,0,0],
                      [0,0,0,0,2,4,4,2,1,0],
                      [0,0,0,0,0,0,2,1,2,0],
                      [0,0,0,0,0,0,1,1,1,0],
                      [0,0,0,0,0,0,0,4,1,0],
                      [0,0,0,0,0,0,0,1,4,0],
                      [0,0,0,0,0,0,0,0,5,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]])

    bs = BruteSolve(state)
    solution = bs.solution

