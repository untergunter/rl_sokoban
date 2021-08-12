import numpy as np
from scipy.ndimage import label
from itertools import chain

class SimulateV1:

    def __init__(self):
        self.move_functions = {1:self.__move_up,
                               2:self.__move_down,
                               3:self.__move_left,
                               4:self.__move_right}

        self.xy_dif = {(-1,0):2, (1,0):1,
                       (0,1):4 ,(0,-1):3 }
        self.can_walk_on = {1,2} # empty,empty box destination
        self.wall=0
        self.empty=1
        self.empty_destination=2
        self.destination_with_box_on = 3
        self.box = 4
        self.player = 5

    def __boxes_neighbors(self,state):
        near_box = self.__neighbors(state,self.box)
        return near_box

    def __player_neighbors(self,state):
        near_player = self.__neighbors(state, self.player)
        return near_player

    def __neighbors(self,state,value):
        current_location = state==value
        right = np.roll(current_location, 1)
        left = np.roll(current_location, -1)
        down = np.roll(current_location, 1, axis=0)
        up = np.roll(current_location, -1, axis=0)
        near = right + left + up + down
        return near

    def get_possible_actions(self, state):
        next_to_boxes = self.__boxes_neighbors(state)
        reachable_locations = self.__reachable_locations(state)
        reachable_next_to_box = self.__get_intersected_indexes(
            next_to_boxes,reachable_locations)
        valid_actions = self.__find_locations_actions_moved_box(state, reachable_next_to_box)
        return valid_actions

    def __get_player_x_y(self, state):
        player_y, player_x = np.argwhere(state == self.player)[0]
        return player_y, player_x

    def steps_to_get_to(self,state,y_destination,x_destination):
        distance_to_destination = self.__calc_distance(state,y_destination,
                                                       x_destination)
        player_y, player_x = self.__get_player_x_y(state)
        moves_to_take = self.__find_path(distance_to_destination,player_y, player_x)
        return moves_to_take

    def __calc_distance(self,state,y_destination,x_destination):
        distances = np.full(state.shape,np.inf)
        distances[(distances==self.empty)|
                  (self.empty_destination)|
                  (self.player)] = -np.inf

        distances[y_destination,x_destination] = 0
        player_y, player_x = self.__get_player_x_y(state)
        reached_player = distances[player_y, player_x] != -np.inf
        current_distance = 0
        can_step_on = distances != np.inf
        while not reached_player:
            next_got_to = self.__neighbors(distances,current_distance)
            current_distance += 1
            distances [next_got_to & can_step_on] = current_distance
        return distances

    def __find_next_coordinates(self,distances,y,x):
        best_score = distances[y,x]
        if distances[y,x-1] < best_score:
            best_y = y
            best_x = x-1
        if distances[y,x+1] < best_score:
            best_y = y
            best_x = x+1
        if distances[y-1,x] < best_score:
            best_y = y-1
            best_x = x
        if distances[y+1,x] < best_score:
            best_y = y+1
            best_x = x
        return best_y,best_x

    def __find_path(self,distances,current_y,current_x):
        steps = []
        destination_y,destination_x = np.argwhere(distances == 0)[0]
        got_there = (current_y==destination_y) & (current_x==destination_x)
        while not got_there:
            next_y,next_x = self.__find_next_coordinates(distances,current_y,current_x)
            step = self.xy_dif((current_y-next_y,current_x-next_x))
            steps.append(step)
            current_y, current_x = next_y,next_x
            got_there = (current_y == destination_y) & (current_x == destination_x)
        return steps

    def __reachable_locations(self,state):
        state = np.copy(state)
        open_floor_and_player = ((state==self.player)|
                                 (state==self.empty)|
                                 (state==self.empty_destination)
                                 ).astype(int)
        connected_open_areas= label(open_floor_and_player)[0]
        player_y,player_x = self.__get_player_x_y(state)
        reachable = connected_open_areas==connected_open_areas[player_y,player_x]
        return reachable

    def __get_intersected_indexes(self,array_1,array_2):
        both_true = array_1 & array_2
        relevant_indexes = np.argwhere(both_true == True).T
        relevant_indexes = list(zip(relevant_indexes[0], relevant_indexes[1]))
        return relevant_indexes

    def __actions_moved_a_box(self,state,location):
        new_state = np.copy(state)
        # remove player from current location
        new_state[new_state == self.player] = self.empty
        # put player at location
        new_state[location] = self.player
        actions_moved_a_box =[(location[0],location[1],action_number)
                              for action_number in self.move_functions
                              if self.__moved_box(new_state,action_number)]
        return actions_moved_a_box

    def __moved_box(self,state,move):
        state = np.copy(state)
        new_state = self.__apply_move(move,state)
        # original_box_locations = self.__set_of_indexes_where(state,self.box)
        # moved_box_locations = self.__set_of_indexes_where(new_state,self.box)
        box_moved = ~np.all(np.equal(state==self.box,new_state==self.box))
        return box_moved

    def __set_of_indexes_where(self,array,match_value):
        indexes = np.argwhere(array==match_value)
        indexes_set = {tuple(location_index) for location_index in indexes}
        return indexes_set

    def __find_locations_actions_moved_box(self, state, locations):
        relevant_locations_lists = [self.__actions_moved_a_box(state,location)
                                    for location in locations]
        relevant_locations_actions = list(chain.from_iterable(relevant_locations_lists))
        return relevant_locations_actions

    def __apply_move(self, move, state):
        """ un safe - deleting destination if boxes moved from over them """
        state = state.copy()
        movement = self.move_functions[move]
        state = movement(state)
        return state


    def __move_down(self, state):
        state = np.rot90(state,2)
        state = self.__move_up(state)
        state = np.rot90(state,2)
        return state


    def __move_right(self, state):
        state = np.rot90(state)
        state = self.__move_up(state)
        state = np.rot90(state,-1)
        return state


    def __move_left(self, state):
        state = np.rot90(state,-1)
        state = self.__move_up(state)
        state = np.rot90(state)
        return state


    def __move_up(self, state):

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