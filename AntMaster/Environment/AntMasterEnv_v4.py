import sys
sys.path.append("D:\\yves\\source\\repos\\RL\\AntMasters")
import copy
from os import listdir
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO

from Model.constants import Constants

class AntMasterEnv_v4(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    WATER = "water"
    FOOD = "food"
    OTHER_ANTS = "other_ants"
    OTHER_HILLS = "other_hills"
    OWN_ANTS = "own_ants"
    OWN_HILLS = "own_hills"
    AGENT_ANT = "agent_ant"
    VISIBILITY = "visibility"
    OWN_FOOD_RESERVE = "own_food_reserve"
    OTHER_FOOD_RESERVE = "other_food_reserve"
    REWARD_DIED = -0
    REWARD_KILLED = 0
    REWARD_MOVES_ILLEGALLY = -0.1
    REWARD_MOVES_TO_OWN_HILL = -0.1
    REWARD_MOVES_TO_OTHER_HILL = 0
    REWARD_FOUND_FOOD = 0.5
    REWARD_CLOSE_TO_OWN_ANTS = 0 
    REWARD_NEW_TERRITORY = 0.01 
    REWARD_FURTHER_FROM_FOOD = - 0.05
    REWARD_CLOSER_TO_FOOD = 0.05

    def __init__(self, heigth=36, width=36, max_steps=150, render_mode=None, food_factor=1, predict_env=False):

        #env
        self.height = heigth
        self.width = width
        self.food_factor = food_factor
        self.food_radius = 1
        self.combat_radius = 3
        self.see_radius = 5
        self.max_steps = max_steps
        self.current_step = 0
        # 1e coordinaat is row, 2e is column
        self.high_observation = 255
        self.observation_space = spaces.Box(low=0, high=self.high_observation, shape=(self.height, self.width, 7), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.observation = None
        self.info = None

        #render
        self.window_size = 512
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.size = 5
        self.canvas = None

        #predict
        self.path_model = Constants.path_model
        self.predict_env = predict_env
        self.model_predict = self.get_model_predict()


    def _get_info(self, key=None):
        if key is None:
            return self.info
        else:
            return self.info[key]

    def _set_info(self, key, array):
        self.info[key] = array

    def _initialize_info(self):
        self.info = {}
        self.info[self.WATER] = np.zeros((self.height, self.width), dtype=np.uint8)
        self.info[self.FOOD] = np.zeros((self.height, self.width), dtype=np.uint8)
        self.info[self.OTHER_ANTS] = np.zeros((self.height, self.width), dtype=np.uint8)
        self.info[self.OTHER_HILLS] = np.zeros((self.height, self.width), dtype=np.uint8)
        self.info[self.OWN_ANTS] = np.zeros((self.height, self.width), dtype=np.uint8)
        self.info[self.OWN_HILLS] = np.zeros((self.height, self.width), dtype=np.uint8)
        self.info[self.VISIBILITY] = np.zeros((self.height, self.width), dtype=np.uint8)
        self.info[self.OWN_FOOD_RESERVE] = np.zeros((1), dtype=np.uint8)
        self.info[self.OTHER_FOOD_RESERVE] = np.zeros((1), dtype=np.uint8)
        self.info[self.AGENT_ANT] = np.zeros((self.height, self.width), dtype=np.uint8)

    def get_observation(self, key = None):
        

        visibility = self._get_info(self.VISIBILITY)
        image_constant = np.array([self.high_observation], dtype=np.uint8)

        water = np.multiply(self._get_info(self.WATER), visibility) * image_constant
        food = np.multiply(self._get_info(self.FOOD), visibility) * image_constant
        other_ants = np.multiply(self._get_info(self.OTHER_ANTS), visibility) * image_constant
        other_hills = np.multiply(self._get_info(self.OTHER_HILLS), visibility) * image_constant
        own_hills = np.multiply(self._get_info(self.OWN_HILLS), visibility) * image_constant

        own_ants = self._get_info(self.OWN_ANTS) * image_constant
        agent_ant = self._get_info(self.AGENT_ANT) * image_constant

        observation = np.dstack( (water, food, other_hills, own_hills, other_ants, own_ants, agent_ant) )

        if key is None:
            return observation
        else:
            if key == self.WATER:
                return observation[:,:,0]
            if key == self.FOOD:
                return observation[:,:,1]
            if key == self.OTHER_HILLS:
                return observation[:,:,2]
            if key == self.OWN_HILLS:
                return observation[:,:,3]
            if key == self.OTHER_ANTS:
                return observation[:,:,4]
            if key == self.OWN_ANTS:
                return observation[:,:,5]
            elif key == self.AGENT_ANT:
                return observation[:,:,6]


    def reset(self, seed=None, options=None):

        self.current_step = 0
        self._initialize_info()

        self._set_info(self.OWN_ANTS, self.initialize_own_ants())
        self._set_info(self.AGENT_ANT, self.initialize_agent_ant())
        self.update_visibility()
        self._set_info(self.WATER, self.initialize_water())
        self._set_info(self.FOOD, self.initialize_food())
        self._set_info(self.OTHER_ANTS, self.initialize_other_ants())
        self._set_info(self.OTHER_HILLS, self.initialize_other_hills())
        self._set_info(self.OWN_HILLS, self.initialize_own_hills())

        return self.get_observation(), self._get_info()

    def initialize_water(self):

        water = self._get_info(self.WATER)
        water[0][:] = 1
        water[self.height-1][:] = 1
        water[:, 0] = 1
        water[:, self.width-1] = 1
        # water[12,1:9] = 1
        # water[14,20:29] = 1
        return water

    def initialize_food(self):

        food = self._get_info(self.FOOD)

        # food[3][1] = 1
        # food[2][5] = 1
        # food[3][24] = 1
        # food[5][28] = 1
        # food[20][10] = 1
        # food[22][23] = 1
        # food[24][18] = 1

        for i in range(0, 7):

            rando_row = np.random.randint(1, self.height - 2)
            rando_col = np.random.randint(1, self.width - 2)
            
            food[rando_row][rando_col] = 1
        return food

    def initialize_other_ants(self):

        other_ants = self._get_info(self.OTHER_ANTS)

        rando_row = np.random.randint(1, self.height-2)
        rando_col = np.random.randint(1, self.width-2)
        
        other_ants[rando_row][rando_col] = 1

        # other_ants[3][4] = 1

        return other_ants

    def initialize_other_hills(self):

        other_hills = self._get_info(self.OTHER_HILLS)
        rando_row = np.random.randint(1, self.height-2)
        rando_col = np.random.randint(1, self.width-2)
        
        other_hills[rando_row][rando_col] = 1

        # other_hills[3][5] = 1

        return other_hills

    def initialize_own_ants(self):

        own_ants = self._get_info(self.OWN_ANTS)
        return own_ants

    def initialize_own_hills(self):

        own_hills = self._get_info(self.OWN_HILLS)
        rando_row = np.random.randint(1, self.height - 2)
        rando_col = np.random.randint(1, self.width - 2)
        
        own_hills[rando_row][rando_col] = 1

        # own_hills[26][27] = 1

        return own_hills

    def initialize_agent_ant(self):

        agent_ant = self._get_info(self.AGENT_ANT)
        
        random_row = np.random.randint(1,self.height - 2)
        random_column = np.random.randint(1,self.width - 2)
        agent_ant[random_row][random_column] = 1

        # agent_ant[26][26] = 1

        return agent_ant

    def get_model_predict(self):
        return None
        file_list = listdir(self.path_model)

        files_cnt = len(file_list)

        if files_cnt == 0 or self.predict_env:
            return None

        else:
            latest_file = self.path_model + "\\" + file_list[files_cnt - 1]
            env = AntMasterEnv_v4(36, 36, 1500, predict_env=True)
            model = PPO.load(latest_file, env=env)
            return model

    def update_visibility(self):

        # agent ant visibility
        agent_ant_location = self.get_agent_ant_location()
        agent_ant_boundaries = self.get_row_and_column_boundaries(agent_ant_location[0], agent_ant_location[1], self.see_radius)
        agent_ant_visibility = np.zeros((self.height, self.width), dtype=np.uint8)
        agent_ant_visibility[agent_ant_boundaries[0]:agent_ant_boundaries[1],agent_ant_boundaries[2]:agent_ant_boundaries[3]] = 1

        # own ants visibility
        own_ants_visibility = np.zeros((self.height, self.width), dtype=np.uint8)
        for own_ant_location in self.get_own_ants_locations():
            own_ant_boundaries = self.get_row_and_column_boundaries(own_ant_location[0], own_ant_location[1], self.see_radius)
            own_ants_visibility[own_ant_boundaries[0]:own_ant_boundaries[1],own_ant_boundaries[2]:own_ant_boundaries[3]] = 1

        # add to visibility
        visibility = self._get_info(self.VISIBILITY)
        new_vis = np.maximum.reduce([visibility, agent_ant_visibility, own_ants_visibility])

        self._set_info(self.VISIBILITY, new_vis)

    def get_row_and_column_boundaries(self, row, colum, range):
        return (max(row-range, 0),
                min(row+range+1, self.height),
                max(colum-range, 0),
                min(colum+range+1, self.width))

    def step(self, action):

        self.current_step = self.current_step + 1

        # Execute bots + ApplyMovements
        reward_move = self.apply_movements(action)

        # UpdateVisibility
        self.update_visibility()

        # CheckFoodInRange()
        self.check_food_in_range()

        # SpawnNewAnts()
        self.spawn_ants()

        # SpawnFood()
        self.spawn_food()

        # DoCombat()
        reward_combat = self.do_combat()

        # DestroyHills
        self.destroy_hills()

        reward = reward_move + reward_combat
        terminated = self.is_terminated()

        return self.get_observation(), reward, terminated, False, self._get_info()

    def do_combat(self):

        reward = 0
        dead_list = []
        matrix = self.get_all_ants()
        agent_ant_location = self.get_agent_ant_location()
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):

                friendly = matrix[row, column]

                if friendly == 0:
                    continue

                if friendly == 1:
                    enemy = 2
                else:
                    enemy = 1
          
                friendly_boundaries = self.get_row_and_column_boundaries(row, column, self.combat_radius)

                ants_in_radius_friendly = matrix[friendly_boundaries[0]:friendly_boundaries[1], friendly_boundaries[2]:friendly_boundaries[3]]
                friendly_ants_in_radius_friendly = np.argwhere(ants_in_radius_friendly == friendly)
                enemy_ants_in_radius_friendly = np.argwhere(ants_in_radius_friendly == enemy)

                weakness = len(friendly_ants_in_radius_friendly) - len(enemy_ants_in_radius_friendly)

                if len(enemy_ants_in_radius_friendly) == 0:
                    continue


                is_agent_ant_fighting = False
                minWeaknessOfEnemy = +999999
                for enemy_ant in enemy_ants_in_radius_friendly:
                    enemy_row_in_matrix = enemy_ant[0] + friendly_boundaries[0]
                    enemy_column_in_matrix = enemy_ant[1] + friendly_boundaries[2]

                    enemy_boundaries = self.get_row_and_column_boundaries(enemy_row_in_matrix, enemy_column_in_matrix, self.combat_radius)

                    ants_in_radius_enemy = matrix[enemy_boundaries[0]:enemy_boundaries[1],enemy_boundaries[2]:enemy_boundaries[3]]

                    cnt_enemy_ants_in_radius_enemy = len(np.argwhere(ants_in_radius_enemy == enemy))
                    cnt_friendly_ants_in_radius_friendly = len(np.argwhere(ants_in_radius_enemy == friendly))

                    diff = cnt_enemy_ants_in_radius_enemy - cnt_friendly_ants_in_radius_friendly
                    if diff < minWeaknessOfEnemy:
                        minWeaknessOfEnemy = diff

                    if enemy_boundaries[0] <= agent_ant_location[0] <= enemy_boundaries[0] and enemy_boundaries[2] <= agent_ant_location[1] <= enemy_boundaries[3]:
                        is_agent_ant_fighting = True

                # if the weakest enemy is stronger dan tile, then destroy tile
                if minWeaknessOfEnemy >= weakness:
                    dead_list.append((row, column))

                    if agent_ant_location[0] == row and agent_ant_location[1] == column:
                        reward += self.REWARD_DIED

                    elif is_agent_ant_fighting:
                        reward += self.REWARD_KILLED

        if len(dead_list) > 0:

            other_ants = self._get_info(self.OTHER_ANTS)
            own_ants = self._get_info(self.OWN_ANTS)
            agent_ant = self._get_info(self.AGENT_ANT)

            for ant in dead_list:
                ant_row = ant[0]
                ant_column = ant[1]

                if other_ants[ant_row, ant_column] == 1:
                    other_ants[ant_row, ant_column] = 0

                elif own_ants[ant_row, ant_column] == 1:
                    own_ants[ant_row, ant_column] = 0

                elif agent_ant[ant_row, ant_column] == 1:
                    agent_ant[ant_row, ant_column] = 0

            self._set_info(self.OTHER_ANTS, other_ants)
            self._set_info(self.OWN_ANTS, own_ants)
            self._set_info(self.AGENT_ANT, agent_ant)

        return reward

    def destroy_hills(self):

        other_ants = self._get_info(self.OTHER_ANTS)
        own_ants = self._get_info(self.OWN_ANTS)
        agent_ant = self._get_info(self.AGENT_ANT)

        other_hills_locations = self.get_other_hills_locations()
        other_hills = self._get_info(self.OTHER_HILLS)

        self._set_info(self.OTHER_HILLS, other_hills)

        for other_hill in other_hills_locations:
            if own_ants[other_hill[0], other_hill[1]] == 1 or agent_ant[other_hill[0], other_hill[1]] == 1:
                other_hills[other_hill[0], other_hill[1]] = 0

        own_hills_locations = self.get_own_hills_locations()
        own_hills = self._get_info(self.OWN_HILLS)

        for own_hill_loc in own_hills_locations:
            if other_ants[own_hill_loc[0], own_hill_loc[1]] == 1:
                own_hills[own_hill_loc[0], own_hill_loc[1]] = 0

        
        
        self._set_info(self.OWN_HILLS, own_hills)

    def spawn_food(self):

        while self.get_total_food() < self.height * self.width * 0.00125 * self.food_factor and self.is_unoccupied_tile():

            unoccupied_tiles = self.get_unoccupied_tiles()
            random = np.random.randint(0, len(unoccupied_tiles))
            random_unoccoupied_tile = unoccupied_tiles[random]

            food = self._get_info(self.FOOD)
            food[random_unoccoupied_tile[0], random_unoccoupied_tile[1]] = 1

            self._set_info(self.FOOD, food)

    def check_food_in_range(self):

        food_locations = self.get_food_locations()
        eaten_food_locations = []

        for food_location in food_locations:

            count_own_ants = self.get_number_of_ants_in_food_radius(
                self.OWN_ANTS, food_location)
            count_agent_ants = self.get_number_of_ants_in_food_radius(
                self.AGENT_ANT, food_location)

            count_other_ants = self.get_number_of_ants_in_food_radius(
                self.OTHER_ANTS, food_location)

            if count_own_ants + count_agent_ants > count_other_ants:
                own_food = self._get_info(self.OWN_FOOD_RESERVE)
                self._set_info(self.OWN_FOOD_RESERVE, np.add(
                    own_food, np.ones((1), dtype=np.uint8)))
                eaten_food_locations.append(food_location)

            elif count_own_ants + count_agent_ants < count_other_ants:
                other_food = self._get_info(self.OTHER_FOOD_RESERVE)
                self._set_info(self.OTHER_FOOD_RESERVE, np.add(
                    other_food, np.ones((1), dtype=np.uint8)))
                eaten_food_locations.append(food_location)

        food = self._get_info(self.FOOD)

        for eaten_food_location in eaten_food_locations:
            food[eaten_food_location[0], eaten_food_location[1]] = 0

        self._set_info(self.FOOD, food)

    def apply_movements(self, action):

        old_observation = self.get_observation()

        reward = 0

        moves = {}

        # agent_ant
        old_agent_ant_location = self.get_agent_ant_location()
        direction = self._action_to_direction(action)
        new_agent_ant_location = np.add(old_agent_ant_location, direction)

        moves[(old_agent_ant_location[1], old_agent_ant_location[0])] = [self.AGENT_ANT, old_agent_ant_location, new_agent_ant_location]

        if self.moves_outside_of_map(new_agent_ant_location) or self._get_info(self.WATER)[new_agent_ant_location[0], new_agent_ant_location[1]] == 1:
            reward += self.REWARD_MOVES_ILLEGALLY

        movement = self.movement_towards_food(old_observation, old_agent_ant_location, new_agent_ant_location)

        if movement > 0:
            reward += self.REWARD_FURTHER_FROM_FOOD

        if movement < 0:
            reward += self.REWARD_CLOSER_TO_FOOD

        if self.agent_ant_found_food(new_agent_ant_location):
            reward += self.REWARD_FOUND_FOOD

        if self.agent_explored_new_territory(new_agent_ant_location):
            reward += self.REWARD_NEW_TERRITORY

        if self.agent_ant_moves_to_hill(self.OWN_HILLS, new_agent_ant_location):
            reward += self.REWARD_MOVES_TO_OWN_HILL

        if self.agent_ant_moves_to_hill(self.OTHER_HILLS, new_agent_ant_location):
            reward += self.REWARD_MOVES_TO_OTHER_HILL

            # own_ants
        own_ants_location = self.get_own_ants_locations()
        for own_ant_location in own_ants_location:
            direction = self._action_to_direction(self.get_own_ant_action((own_ant_location[0], own_ant_location[1])))
            new_own_ant_location = np.add(own_ant_location, direction)
            moves[(own_ant_location[1], own_ant_location[0])] = [self.OWN_ANTS, own_ant_location, new_own_ant_location]

        if self.agent_ant_close_to_own_ants():
            reward += self.REWARD_CLOSE_TO_OWN_ANTS

            # other_ants
        other_ants_location = self.get_other_ants_location()
        for other_ant_location in other_ants_location:
            direction = self._action_to_direction(np.random.randint(5))
            new_other_ant_location = np.add(other_ant_location, direction)

            moves[(other_ant_location[1], other_ant_location[0])] = [self.OTHER_ANTS, other_ant_location, new_other_ant_location]

        sorted_dict = sorted(moves.items())

        # ApplyMovements

        for key, value in sorted_dict:
            if (self.can_move_to(value[2])):
                self.move(value[0], value[1], value[2])

        return reward

    def movement_towards_food(self, old_observation, old_agent_ant_location, new_agent_ant_location):
        food = old_observation[:,:,1]
        food_locations = np.argwhere(food == self.high_observation)
        if len(food_locations) == 0:
            return 0
        else:
            closest_food_old = 9999
            closest_food_new = 9999
            for food_location in food_locations:
                distance_to_old_food = abs(food_location[0] - old_agent_ant_location[0]) + abs(food_location[1] - old_agent_ant_location[1])
                closest_food_old = min(closest_food_old, distance_to_old_food)

            for food_location in food_locations:
                distance_to_new_food = abs(food_location[0] - new_agent_ant_location[0]) + abs(food_location[1] - new_agent_ant_location[1])
                closest_food_new = min(closest_food_new, distance_to_new_food)
        
            return closest_food_new - closest_food_old

            

    def get_own_ant_action(self, own_ant_location_tuple):

        if self.model_predict is None:
            return np.random.randint(5)

        else:
            # get observation for that ant
            obs = copy.deepcopy(self.get_observation())

            agent_ant = obs[:, :, 6]
            own_ants = obs[:, :, 5]

            agent_ant_location = np.argwhere(agent_ant == self.high_observation)[0]

            agent_ant[agent_ant_location[0], agent_ant_location[1]] = 0
            own_ants[agent_ant_location[0], agent_ant_location[1]] = self.high_observation

            agent_ant[own_ant_location_tuple[0], own_ant_location_tuple[1]] = self.high_observation
            own_ants[own_ant_location_tuple[0], own_ant_location_tuple[1]] = 0

            obs[:, :, 6] = agent_ant
            obs[:, :, 5] = own_ants

            # predict based on that observation
            action = self.model_predict.predict(obs, deterministic=True)[0][()]

            return action

    def agent_explored_new_territory(self, new_agent_ant_location):
        
        agent_ant_boundaries = self.get_row_and_column_boundaries(new_agent_ant_location[0], new_agent_ant_location[1], self.see_radius)

        visibility = self._get_info(self.VISIBILITY)[
            agent_ant_boundaries[0]:agent_ant_boundaries[1], agent_ant_boundaries[2]:agent_ant_boundaries[3]]
        amount_of_new_locations = np.argwhere(visibility == 0)

        return len(amount_of_new_locations) > 0

    def agent_ant_close_to_own_ants(self):

        agent_ant_location = self.get_agent_ant_location()

        agent_ant_boundaries = self.get_row_and_column_boundaries(agent_ant_location[0],agent_ant_location[1], self.combat_radius)

        own_ants = self._get_info(self.OWN_ANTS)

        own_ants_close_to_agent_ant = own_ants[agent_ant_boundaries[0]:agent_ant_boundaries[1], agent_ant_boundaries[2]:agent_ant_boundaries[3]]

        return len(np.argwhere(own_ants_close_to_agent_ant == 1)) > 0

    def agent_ant_moves_to_hill(self, key, new_agent_ant_location):

        return self._get_info(key)[new_agent_ant_location[0], new_agent_ant_location[1]] == 1

    def agent_ant_found_food(self, new_agent_ant_location):

        agent_ant_boundaries = self.get_row_and_column_boundaries(new_agent_ant_location[0],new_agent_ant_location[1], self.food_radius)

        food_in_radius = self._get_info(self.FOOD)[agent_ant_boundaries[0]:agent_ant_boundaries[1],agent_ant_boundaries[2]:agent_ant_boundaries[3]]

        return len(np.argwhere(food_in_radius == 1)) > 0

    def is_terminated(self):
        has_reached_max_steps = self.current_step == self.max_steps
        is_agent_ant_dead = len(np.argwhere(self._get_info(self.AGENT_ANT) == 1)) == 0
        has_own_hills = len(np.argwhere(self._get_info(self.OWN_HILLS) == 1)) > 0
        has_other_hills = len(np.argwhere(self._get_info(self.OTHER_HILLS) == 1)) > 0
        one_player_is_defeated = not (has_own_hills and has_other_hills)

        return has_reached_max_steps or is_agent_ant_dead or one_player_is_defeated

    def is_unoccupied_tile(self):

        return len(self.get_unoccupied_tiles()) > 0

    def get_total_food(self):
        return len(np.argwhere(self._get_info(self.FOOD) == 1))

    def spawn_ants(self):

        own_food_reserve = self._get_info(self.OWN_FOOD_RESERVE)[0]

        own_hills_locations = self.get_own_hills_locations()
        agent_ant = self._get_info(self.AGENT_ANT)
        own_ants = self._get_info(self.OWN_ANTS)

        own_hills_locations_without_ants = {}

        for own_hill_location in own_hills_locations:

            if agent_ant[own_hill_location[0], own_hill_location[1]] == 0 and own_ants[own_hill_location[0], own_hill_location[1]] == 0:
                own_hills_locations_without_ants[(own_hill_location[0], own_hill_location[1])] = True

        while own_food_reserve > 0 and len(own_hills_locations_without_ants) > 0:
            own_food_reserve = own_food_reserve - 1

            # get random hill
            random = np.random.randint(0, len(own_hills_locations_without_ants))
            random_hill = list(own_hills_locations_without_ants.keys())[random]

            own_ants[random_hill[0], random_hill[1]] = 1

            del own_hills_locations_without_ants[random_hill]

        food_array = np.ones((1), dtype=np.uint8)
        food_array[0] = own_food_reserve

        self._set_info(self.OWN_FOOD_RESERVE, food_array)
        self._set_info(self.OWN_ANTS, own_ants)

        # repeat for other player

        other_food_reserve = self._get_info(self.OTHER_FOOD_RESERVE)[0]

        other_hills_locations = self.get_other_hills_locations()
        other_ants = self._get_info(self.OTHER_ANTS)

        other_hills_locations_without_ants = {}

        for other_hill_location in other_hills_locations:

            if other_ants[other_hill_location[0], other_hill_location[1]] == 0:
                other_hills_locations_without_ants[(other_hill_location[0], other_hill_location[1])] = True

        while other_food_reserve > 0 and len(other_hills_locations_without_ants) > 0:
            other_food_reserve = other_food_reserve - 1

            random = np.random.randint(0, len(other_hills_locations_without_ants))

            random_hill = list(other_hills_locations_without_ants.keys())[random]

            other_ants[random_hill[0], random_hill[1]] = 1

            del other_hills_locations_without_ants[random_hill]

        other_food_array = np.ones((1), dtype=np.uint8)
        other_food_array[0] = other_food_reserve

        self._set_info(self.OTHER_FOOD_RESERVE, other_food_array)
        self._set_info(self.OTHER_ANTS, other_ants)

    def get_number_of_ants_in_food_radius(self, key, location):

        ants = self._get_info(key)
        food_location_boundaries = self.get_row_and_column_boundaries(location[0], location[1], self.food_radius)
        food_radius = ants[food_location_boundaries[0]:food_location_boundaries[1],food_location_boundaries[2]:food_location_boundaries[3]]

        cnt_ants_in_food_radius = len(np.argwhere(food_radius == 1))

        return cnt_ants_in_food_radius

    def _action_to_direction(self, action):

        if action == '0' or action == 0:    # right
            return np.array([0, 1])
        elif action == '1' or action == 1:  # down
            return np.array([1, 0])
        elif action == '2' or action == 2:  # left
            return np.array([0, -1])
        else:                               # up
            return np.array([-1, -0])

    def get_agent_ant_location(self):
        return np.argwhere(self._get_info(self.AGENT_ANT) == 1)[0]

    def get_own_ants_locations(self):
        return np.argwhere(self._get_info(self.OWN_ANTS) == 1)

    def get_food_locations(self):
        return np.argwhere(self._get_info(self.FOOD) == 1)

    def get_other_ants_location(self):
        return np.argwhere(self._get_info(self.OTHER_ANTS) == 1)

    def get_all_ants(self):
        other_ants = self._get_info(self.OTHER_ANTS)
        other_ants = other_ants * 2

        own_ants = self._get_info(self.OWN_ANTS)
        agent_ant = self._get_info(self.AGENT_ANT)

        return np.maximum.reduce([other_ants, own_ants, agent_ant])

    def get_own_hills_locations(self):
        return np.argwhere(self._get_info(self.OWN_HILLS) == 1)

    def get_other_hills_locations(self):
        return np.argwhere(self._get_info(self.OTHER_HILLS) == 1)

    def get_unoccupied_tiles(self):
        water = self._get_info(self.WATER)
        own_ants = self._get_info(self.OWN_ANTS)
        agent_ant = self._get_info(self.AGENT_ANT)
        other_ants = self._get_info(self.OTHER_ANTS)

        return np.argwhere(np.maximum.reduce([water, own_ants, agent_ant, other_ants]) == 0)

    def can_move_to(self, new_position):
        outside_of_map = self.moves_outside_of_map(new_position) 
        has_water = self._get_info(self.WATER)[new_position[0], new_position[1]] == 1
        has_own_ant = self._get_info(self.OWN_ANTS)[new_position[0], new_position[1]] == 1
        has_agent_ant = self._get_info(self.AGENT_ANT)[new_position[0], new_position[1]] == 1
        has_other_ant = self._get_info(self.OTHER_ANTS)[new_position[0], new_position[1]] == 1

        return not (outside_of_map or has_water or has_own_ant or has_agent_ant or has_other_ant)

    def moves_outside_of_map(self, new_position):
        return new_position[0] < 0 or new_position[0] > self.height - 1 or new_position[1] < 0 or new_position[1] > self.width - 1

    def move(self, key, old_position, new_position):

        array = self._get_info(key)
        array[old_position[0], old_position[1]] = 0
        array[new_position[0], new_position[1]] = 1
        self._set_info(key, array)

    def render(self, full_visibility = True):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))

        pix_square_size = (
            self.window_size / self.height
        )

        # gridlines
        for x in range(self.height + 1):
            pygame.draw.line(
                self.canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                self.canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        
        if full_visibility:
            value = 1
            visibility = self._get_info()[self.VISIBILITY]
            water = self._get_info()[self.WATER]
            food = self._get_info()[self.FOOD]
            other_ants = self._get_info()[self.OTHER_ANTS]
            other_hills = self._get_info()[self.OTHER_HILLS]
            own_ants = self._get_info()[self.OWN_ANTS]
            own_hills = self._get_info()[self.OWN_HILLS]
            agent_ant = self._get_info()[self.AGENT_ANT]


        else:
            value = 255
            visibility = self._get_info()[self.VISIBILITY] * np.array([self.high_observation], dtype=np.uint8)
            water = self.get_observation(self.WATER)
            food = self.get_observation(self.FOOD)
            other_ants = self.get_observation(self.OTHER_ANTS)
            other_hills = self.get_observation(self.OTHER_HILLS)
            own_ants = self.get_observation(self.OWN_ANTS)
            own_hills = self.get_observation(self.OWN_HILLS)
            agent_ant = self.get_observation(self.AGENT_ANT)

        
        self.draw_tiles(visibility, value, (192, 192, 192))
        self.draw_tiles(water, value, (0, 0, 255))
        self.draw_tiles(food, value, (255, 0, 0))
        self.draw_tiles(other_ants, value, (102, 255, 255))
        self.draw_tiles(other_hills, value, (0, 153, 255))
        self.draw_tiles(own_ants, value, (204, 204, 255))
        self.draw_tiles(own_hills, value, (204, 102, 255))
        self.draw_tiles(agent_ant, value, (0, 0, 0))
        

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

    def draw_tiles(self, array, value, color):
        ones = np.argwhere(array == value)

        pix_square_size = (
            self.window_size / self.height
        )

        for index in ones:

            pygame.draw.rect(
                self.canvas,
                color,
                pygame.Rect(index[1] * pix_square_size, index[0] *
                            pix_square_size, pix_square_size, pix_square_size)
            )
