# This is necessary to find the main code
import sys
import math
sys.path.insert(0, '../../../bomberman')


# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from world import World
import random
import numpy as np
from collections import deque
from game import Game
import torch
import pandas as pd
import csv
from sensed_world import SensedWorld
from events import *

from model import Linear_QNet, QTrainer 
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class TestCharacter(CharacterEntity):
    def __init__(self, name, avatar, x, y, model):
        CharacterEntity.__init__(self, name, avatar, x, y)
        self.model = model
        gamma = 0.95 #discount rate
        self.trainer = QTrainer(self.model, lr = LR, gamma = gamma)
        self.epsilon = 1 # randomness
    n_games = 0
    
    memory = deque(maxlen = MAX_MEMORY) # pop left overloading memory

    old = []

    prev_exit = 0
    # moves to iteration through the list of paths 
    moves = 0
    # our depth range for looking for the monster
    rnge = 2
    # used to determine if the bomb explosion is gone
    bombCycle = 0
    # used to update the placed bomb location
    bomb_location = None
    # used to determine if the bomb has been dropped
    bomb_start = False

    # Just looks for neighbors of 8 as the frontier
    def look_for_empty_cell(self, wrld, current):
        # List of empty cells
        cells = []
        # Go through neighboring cells
        for dx in [-1, 0, 1]:
            # Avoid out-of-bounds access
            if ((current[0] + dx >= 0) and (current[0] + dx < wrld.width())):
                for dy in [-1, 0, 1]:
                    # Avoid out-of-bounds access
                    if ((current[1] + dy >= 0) and (current[1] + dy < wrld.height())):
                        # Is this cell safe?
                        if(wrld.exit_at(current[0] + dx, current[1] + dy) or
                           wrld.empty_at(current[0] + dx, current[1] + dy) or
                           (wrld.wall_at(current[0] + dx, current[1] + dy) and dy != -1)):
                                cells.append((current[0] + dx, current[1] + dy))
        # All done
        return cells

    # Just looks for neighbors of 8 as the frontier
    def look_for_empty_cell_with_explostion(self, wrld, current):
        # List of empty cells
        cells = []
        # Go through neighboring cells
        for dx in [-1, 0, 1]:
            # Avoid out-of-bounds access
            if ((current[0] + dx >= 0) and (current[0] + dx < wrld.width())):
                for dy in [-1, 0, 1]:
                    # Avoid out-of-bounds access
                    if ((current[1] + dy >= 0) and (current[1] + dy < wrld.height())):
                        # Is this cell safe?
                        if (wrld.exit_at(current[0] + dx, current[1] + dy) or 
                            not wrld.wall_at(current[0] + dx, current[1] + dy) or
                            not wrld.characters_at(current[0] + dx, current[1] + dy)):
                                cells.append((current[0] + dx, current[1] + dy))
        # All done
        return cells
    
    
    # Just looks for neighbors of 8 as the frontier
    def look_for_empty_cell_states(self, wrld, current):
        # List of empty cells
        cells = []
        # Go through neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Avoid out-of-bounds access
                if ((current[0] + dx >= 0) and (current[0] + dx < wrld.width())):
                # Avoid out-of-bounds access
                    if ((current[1] + dy >= 0) and (current[1] + dy < wrld.height())):
                    # Is this cell safe?
                        if(wrld.exit_at(current[0] + dx, current[1] + dy) or
                            wrld.empty_at(current[0] + dx, current[1] + dy)):
                            cells.append(1)
                        else:
                            cells.append(0)
                    else:
                        cells.append(0)
                else:
                    cells.append(0)
        # All done
        return cells
    
    # Looks for neighbors of 8 as the frontier with the explosion radius incorporated
    def look_for_empty_cell_bomb(self, wrld, current):
        # List of empty cells
        cells = []
        # Go through neighboring cells
        for dx in [-1, 0, 1]:
            # Avoid out-of-bounds access
            if ((current[0] + dx >= 0) and (current[0] + dx < wrld.width())):
                for dy in [-1, 0, 1]:
                    # Avoid out-of-bounds access
                    if ((current[1] + dy >= 0) and (current[1] + dy < wrld.height())):
                        # Is this cell safe?
                        if ((wrld.exit_at(current[0] + dx, current[1] + dy) or
                              wrld.empty_at(current[0] + dx, current[1] + dy)) and 
                              not self.bomb_blast(current[0] + dx, current[1] + dy)):
                                cells.append((current[0] + dx, current[1] + dy))
        # All done
        return cells

    # Looks for the empty cells around the current position, but does not allow area around monster to be considered
    def look_for_empty_cell_monster(self, wrld, current, monster):
        # List of empty cells
        cells = []
        # Go through neighboring cells
        for dx in [-1, 0, 1]:
            # Avoid out-of-bounds access
            if ((current[0] + dx >= 0) and (current[0] + dx < wrld.width())):
                for dy in [-1, 0, 1]:
                    # Avoid out-of-bounds access
                    if ((current[1] + dy >= 0) and (current[1] + dy < wrld.height())):
                        # Is this cell safe?
                        if(wrld.exit_at(current[0] + dx, current[1] + dy) or
                           wrld.empty_at(current[0] + dx, current[1] + dy)):
                            # Makes the area around the monster to not be considered as a frontier          
                            for mx in [-2, -1, 0, 1, 2]:
                                for my in [-2, -1, 0, 1, 2]:
                                    if not ((current[0] + dx == monster[0] + mx) or
                                        (current[1] + dy == monster[1] + my)):

                                        cells.append((current[0] + dx, current[1] + dy))
        # All done
        return cells
    
    # deteremine every possible monster move with a specified depth
    def monster_moves(self, wrld, monster, depth):
        # List of empty cells
        cells = []
        # Go through neighboring cells
        for dx in range(-depth, depth):
            # Avoid out-of-bounds access
            if ((monster[0] + dx >= 0) and (monster[0] + dx < wrld.width())):
                for dy in range(-depth, depth):
                    # Avoid out-of-bounds access
                    if ((monster[1] + dy >= 0) and (monster[1] + dy < wrld.height())):
                        # Is this cell safe?
                        if not (wrld.explosion_at(monster[0] + dx, monster[1] + dy) or
                            wrld.wall_at(monster[0] + dx, monster[1] + dy) or
                            wrld.monsters_at(monster[0] + dx, monster[1] + dy) or
                            wrld.exit_at(monster[0] + dx, monster[1] + dy)):
                            cells.append((monster[0] + dx, monster[1] + dy))

        # All done
        return cells
    
    # function to determine if the monster is within the range of us 
    def monster_in_range(self, wrld):
        for dx in range(-self.rnge, self.rnge+1):
            # Avoid out-of-bounds access
            if ((self.x + dx >= 0) and (self.x + dx < wrld.width())):
                for dy in range(-self.rnge, self.rnge+1):
                    # Avoid out-of-bounds access
                    if ((self.y + dy >= 0) and (self.y + dy < wrld.height())):
                        # Is a character at this position?
                        if (wrld.monsters_at(self.x + dx, self.y + dy)):
                            return (True, self.x + dx, self.y + dy)
        # Nothing found
        return (False, 0, 0)
    
    # heuristic for the chebyshev distance from one gobal position to the next 
    def heuristic(self, goal, next):
        return max(abs(next[0] - goal[0]), abs(next[1] - goal[1]))
   
    # fucnction to calculated the euclidean distance from one position to the next
    def euclidean_distance(self, goal, next):
        return math.sqrt((goal[0]-next[0])**2 + (goal[1] - next[1])**2)

    # function to calculate the cost from the currrent position to the next with walls 
    def cost(self, wrld, current, next):
        move_cost = abs(current[0] - next[0]) + abs(current[1] - next[1])
        # increases the cost if a wall is there 
        if wrld.wall_at(next[0],next[1]):
            move_cost += 16
            for dx in [-1, 0, 1]:
                if ((current[0] + dx >= 0) or (current[0] + dx < wrld.width())):
                    move_cost -= 3
        return move_cost
    
    # function to determine the appropiate location to move after placing a bomb 
    def direction_after_bomb(self, wrld):
        # determines the location of the location
        found, mx, my = self.monster_in_range(wrld)
        monster = mx, my
        weighted_angles = [] 
        possible_angles = []
        # deteremines all the possible angled moves you can make 
        for dx in [-1,1]:
            if ((self.x + dx >= 0) and (self.x + dx < wrld.width())):
                for dy in [-1,1]:    
                    if ((self.y + dy >= 0) and (self.y + dy < wrld.height())):
                        if (wrld.exit_at(self.x + dx, self.y + dy) or
                           wrld.empty_at(self.x + dx, self.y + dy) ):
                            possible_angles.append((self.x + dx, self.y + dy))
        # go through the possible moves and add a weight based on the euclidean distance from the monster and the possible move 
        for move in possible_angles:
            weight = self.euclidean_distance(monster, move)
            weighted_angles.append((move, weight))
        return self.best_move(weighted_angles)

    # goes through the possible moves and their weights to return the best move
    def best_move(self, weighted_angles):
        best_val = -99
        bestMove = [self.x, self.y]
        # goes through all the moves
        for move in weighted_angles:
            if move[1] > best_val:
                bestMove = move[0]
                best_val = move[1] 
        return bestMove

    def bomb_moves(self, wrld):
        next_move = self.direction_after_bomb(wrld)
        self.move(next_move[0], next_move[1])
    
    def get_state(self, wrld):
        state = []
        found, mx, my = self.monster_in_range(wrld)
        if not found: 
            state.append(0)
        else:
            distance_monster = self.heuristic((mx, my),(self.x, self.y))
            state.append(1/(1+distance_monster))
        
        state.extend(self.look_for_empty_cell_states(wrld,(self.x, self.y)))

        self.path = self.makePath(wrld, self.astar(wrld))
        state.append(1/(1 + len(self.path)))

        bomb = 0
        for dx in range(-4,4):
            if wrld.bomb_at(self.x + dx, self.y) or wrld.bomb_at(self.x, self.y +dx):
                bomb = 1
        state.append(bomb)

        if len(wrld.bombs) == 0 and len(wrld.explosions) == 0:
            state.append(1)
        else:
            state.append(0)

        return state 

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon -= 0.003 * self.n_games
        if self.epsilon < .01:
            self.epsilon = .01
        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 12)
            final_move[move] = 1
        else :
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            print(move)
            final_move[int(move)] = 1
        # self.epsilon =  self.epsilon*0.95
        # final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # if random.randint(0, 1) < self.epsilon:
        #     move = random.randint(0, 9)
        #     final_move[move] = 1
        # else :
        #     state0 = torch.tensor(state, dtype=torch.float)
        #     prediction = self.model(state0)
        #     move = torch.argmax(prediction).item()
        #     final_move[int(move)] = 1
        
        # print(final_move)
        
        
        return final_move
    
    def get_score(self, wrld):
        return wrld.scores["me"]

    def get_games(self):
        # data = pd.read_csv("../games.csv")
        # print(data)
        # self.n_games = data[0]
        with open("../games.csv",'r') as f:
            data = csv.reader(f)  
            for row in data:
                self.n_games = int(row[0])
                break
                

    def set_record(self, record):
        with open("../record.csv", 'w') as csvfile:
            updater = csv.writer(csvfile)
            updater.writerow([record])
            csvfile.close()
        
    def get_record(self):
         with open("../record.csv",'r') as f:
            data = csv.reader(f)  
            for row in data:
                return int(row[0])
                

    def get_move(self, final_move):
        dx, dy, bomb = 0, 0, False
        if(final_move[0] == 1):
            #North
            dx = 0
            dy = -1
            bomb = False
        elif (final_move[1] == 1):
            #NorthEast
            dx = 1
            dy = -1
            bomb = False
        elif (final_move[2] == 1):
            #East
            dx = 1
            dy = 0
            bomb = False
        elif (final_move[3] == 1):
            #SouthEast
            dx = 1
            dy = 1
            bomb = False
        elif (final_move[4] == 1):
            #South
            dx = 0
            dy = 1
            bomb = False
        elif (final_move[5] == 1):
            #SouthWest
            dx = -1
            dy = 1
            bomb = False
        elif (final_move[6] == 1):
            #West
            dx = -1
            dy = 0
            bomb = False
        elif (final_move[7] == 1):
            #NorthWest
            dx = -1
            dy = -1
            bomb = False
        elif (final_move[8] == 1):
            #Don't Move
            dx = 0
            dy = 0
            bomb = False
        elif (final_move[9] == 1):
            #BOMBNE
            dx = 1
            dy = -1
            bomb = True
        
        elif (final_move[10] == 1):
            #BOMBSE
            dx = 1
            dy = 1
            bomb = True
        
        elif (final_move[11] == 1):
            #BOMBSW
            dx = -1
            dy = 1
            bomb = True
        elif (final_move[12] == 1):
            #BOMBSE
            dx = -1
            dy = -1
            bomb = True
        
        
        return dx, dy, bomb
    
    def get_reward(self, wrld, dx, dy, hit_wall, hit_monster, hit_me, reaches_exit, monster_kill, used_bomb):
        reward = 0

        # found, mx, my = self.monster_in_range(wrld)
        # if found == True:
        #     reward -= 2
        # else:
        #     reward += 2
        # if mx == range(self.x-1, self.x+1) or  my == range(self.y-1, self.y+1):
        #     reward -= 10
        # else:
        #     reward += 4

        new_distance = len(self.makePath_reward(wrld, self.astar_reward(wrld, dx, dy), dx, dy))
        # print("NEEEEEWW:   " + str(new_distance))
        if new_distance >= len(self.path):
            reward -= 500/(1+new_distance)
        elif new_distance < len(self.path):
            reward += 100/(1+new_distance)
        else:
            reward += 500/(1+new_distance)

        if dx == 0 and dy == 0:
            reward -= 100

        # if dy > 0 or dx > 0:
        #     reward += 10

        # if hit_wall == True:
        #     reward += 20

        if used_bomb == True:
            reward -= 1000
         
        if hit_monster == True:
            reward += 300
        
        if reaches_exit == True:
            reward += 5000

        if monster_kill == True:
            reward -= 5000

        if hit_me == True:
            reward -= 10000
        
        reward -= self.moves

        return reward
    
    def remeber(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def do(self, wrld):

        found, mx, my = self.monster_in_range(wrld)
        self.path = self.makePath(wrld, self.astar(wrld))
        (next_wrld, events) = SensedWorld.next(wrld) 
        record = self.get_record() 
        score = self.get_score(next_wrld)

        if not found : 
            next = self.path[1]  
            dx = next[0] - self.x
            dy = next[1] - self.y
            self.move(dx,dy)

            done = False
            for e in events:
                if e.tpe == Event.CHARACTER_KILLED_BY_MONSTER or e.tpe == Event.CHARACTER_FOUND_EXIT or e.tpe == Event.BOMB_HIT_CHARACTER:
                    done = True  
        else : 

            # plot_scores = []
            # plot_mean_score = []
            # total_score = 0

            self.get_games()

            #get old state 
            state_old = self.get_state(wrld)

            self.old = state_old
            #get move
            final_move = self.get_action(state_old)

            dx, dy, bomb = self.get_move(final_move)

            # print("bomb:", bomb)
            used_bomb = False
            if bomb == True:
                self.bombing()
                used_bomb = True
                self.move(dx, dy)
            else:
                self.move(dx, dy)
            
            self.moves += 1

            hit_wall = False
            for e in events:
                if e.tpe == Event.BOMB_HIT_WALL:
                    hit_wall = True

            hit_monster = False
            for e in events:
                if e.tpe == Event.BOMB_HIT_MONSTER:
                    hit_monster = True

            hit_me = False
            for e in events:
                if e.tpe == Event.BOMB_HIT_CHARACTER:
                    hit_me = True
            
            reaches_exit = False
            for e in events:
                if e.tpe == Event.CHARACTER_FOUND_EXIT:
                    reaches_exit = True
            
            monster_kill = False
            for e in events:
                if e.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                    monster_kill = True

            reward = self.get_reward(wrld, dx, dy, hit_wall, hit_monster, hit_me, reaches_exit, monster_kill, used_bomb)
            print(reward) 

            done = False

            for e in events:
                if e.tpe == Event.CHARACTER_KILLED_BY_MONSTER or e.tpe == Event.CHARACTER_FOUND_EXIT or e.tpe == Event.BOMB_HIT_CHARACTER:
                    done = True  

            state_new = self.get_state(next_wrld)

            #train short memory
            # self.train_short_memory(state_old, final_move, reward, state_new, done)
            self.remeber(state_old, final_move, reward, state_new, done)
            self.train_long_memory()
            #remember

        
    
        if done:
            self.train_long_memory()

            if score > record:
                    self.set_record(score)
                    self.model.save()
                    
            print('Game', self.n_games, 'Score', score, ' Record:' , record)


        

        # if done:
        #     self.train_long_memory()

        #     if score > record:
        #             self.set_record(score)
        #             self.model.save()
        #     print('Game', self.n_games, 'Score', score, ' Record:' , record)

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / self.n_games
            # plot_mean_score.append(mean_score)
            # plot(plot_scores, plot_mean_score)

        # # detemines if the monster was found
        # found, mx, my = self.monster_in_range(wrld)
        # current_pos = self.x, self.y
        # monster = mx, my
        # # Resets the bomb counter
        # if self.bombCycle == 3 and self.bomb_start:
        #     self.bombCycle = 0
        #     self.bomb_start = False
        #     self.bomb_location = None

        # # FOUND STATE
        # if found:
        #     # initializes values
        #     self.moves = 1
        #     self.path = [current_pos]

        #     # BOMB AND RUN STATE
        #     if self.bombCycle == 0: 
        #         # places the bomb and moves away from monster
        #         bomb_move = self.direction_after_bomb(wrld)
        #         self.path.append(bomb_move)
        #         self.bomb_location = self.bombing()
        #         self.bomb_start = True

        #     # EXPECTIMAX AWAY STATE
        #     else:
        #         # expectimaxes away from the monster
        #         new_move = self.exp_max(wrld,monster)
        #         self.path.append(new_move) 

        # #  A* TO TARGET STATE
        # else:
        #     # creates path with a
        #     self.path = self.makePath(wrld, self.astar(wrld))
        #     self.moves = 1
        # found = False

        # # iterates through the path created 
        # if self.moves < len(self.path):
        #     next = self.path[self.moves]  
        #     dx = next[0] - self.x
        #     dy = next[1] - self.y
        #     self.move(dx,dy)
        #     if self.bomb_start == True:
        #         self.bombCycle += 1
        #     self.moves += 1
        
    # goes through the given A* came_from dictionary to find the path to the target cell
    def makePath(self, wrld, came_from):
        current = wrld.exitcell
        path = []
        # if the target cell is in dictionary do
        if current in came_from:              
            while current != (self.x, self.y):
                path.append(current)
                current = came_from[current]
            path.append((self.x, self.y))
            path.reverse()
            return path
        else:
            # stay still
            path.append((self.x, self.y))
            path.append((self.x, self.y))
            return path
    
    #uses the astar but with the frontier not using area around the monster as a frontier
    def astar_monster(self, wrld, mx, my):
        frontier = PriorityQueue()
        came_from = {}
        cost_so_far = {}
        monster = (mx,my)

        frontier.put((self.x, self.y), 0)
        came_from[(self.x, self.y)] = None
        cost_so_far[(self.x, self.y)] = 0
        while not frontier.empty():
            current = frontier.get()
            goal = wrld.exitcell
            if current == goal:
                break
            for next in self.look_for_empty_cell_monster(wrld, current, monster):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current
        return came_from
    

    def makePath_reward(self, wrld, came_from, dx, dy):
        current = wrld.exitcell
        path = []
        # if the target cell is in dictionary do
        if current in came_from:              
            while current != (self.x + dx, self.y + dy):
                path.append(current)
                current = came_from[current]
            path.append((self.x + dx, self.y + dy))
            path.reverse()
            return path
        else:
            # stay still
            path.append((self.x + dx, self.y + dy))
            path.append((self.x + dx, self.y + dy))
            return path


    def astar_reward(self, wrld, dx, dy):
        # initializes A* variables
        frontier = PriorityQueue()
        came_from = {}
        cost_so_far = {}

        frontier.put((self.x + dx, self.y + dy), 0)
        came_from[(self.x + dx, self.y + dy)] = None
        cost_so_far[(self.x + dx, self.y + dy)] = 0
        
        # continues if there is a frontier
        while not frontier.empty():
            current = frontier.get()
            goal = wrld.exitcell
            if current == goal:
                break
            for next in self.look_for_empty_cell_with_explostion(wrld, current):
                new_cost = cost_so_far[current] + self.cost(wrld, current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current
        return came_from


    # Normal A* star using lecture pseduo code 
    def astar(self, wrld):
        # initializes A* variables
        frontier = PriorityQueue()
        came_from = {}
        cost_so_far = {}

        frontier.put((self.x, self.y), 0)
        came_from[(self.x, self.y)] = None
        cost_so_far[(self.x, self.y)] = 0
        
        # continues if there is a frontier
        while not frontier.empty():
            current = frontier.get()
            goal = wrld.exitcell
            if current == goal:
                break
            for next in self.look_for_empty_cell_bomb(wrld, current):
                new_cost = cost_so_far[current] + self.cost(wrld, current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current
        return came_from

    # function to determine the best possible move using expectimax
    def exp_max(self, wrld, monster):
        # returns the possible moves with their weights 
        possible_moves = self.exp_probabilities(wrld, monster)
        best_val = -99999
        new_move = (0,0)
        # goes through every weighted move and returns the best one
        for move in possible_moves:
            if move[1] > best_val:
                best_val = move[1]
                new_move = move[0] 
        return new_move

    # function to calculate the chance node value using probabilities
    def exp_probabilities(self, wrld, monster):
        possible_moves = []
        current_position = self.x, self.y
        # Returns the possible moves that the character can do
        character_moves = self.look_for_empty_cell_bomb(wrld, current_position)
        # goes through every move
        for character_move in character_moves:
            # given the leaf values based on the monster and character position
            end_values = self.find_values(wrld, character_move, monster)
            chance_value = 0
            # goes through each leaf value
            for values in end_values:
                # determines the chance value depending on the length of leaf values
                chance_value += (values/len(end_values))
            possible_moves.append((character_move, chance_value))

        return possible_moves

    # function to determine the leaf nodes values for each assoicated with each monster location 
    def find_values(self, wrld, possible_move, monster):
        # finds all possible monster moves 
        possible_monster_moves = self.monster_moves(wrld, monster, 3)
        # created the base final values
        end_values = []
        # uses the chebyshev distance to depending on our possible move and the monster's current location
        inital_distance = self.heuristic(possible_move, monster)
        # goes through every monster move
        for monster_move in possible_monster_moves:
            # uses the chebyshev distance to depending on our possible move and the monster's possible move
            next_distance = self.heuristic(possible_move, monster_move) 
            # heavy negative weight if monster kills the character
            if next_distance == 0:
                value = -100
            else :
                # scale factor to determine to greater weigh positive changes
                scale_factor = 2
                # determines the leaf node values by the change in intial distance and next distance
                value = scale_factor*(next_distance - inital_distance)
            end_values.append(value)
        return end_values

    # function to place the bomb and returns the location of the bomb
    def bombing(self):
        if self.bombCycle <= 0:
            self.place_bomb()
        return self.x , self.y

    # returns true is the the cell is in the exploding bomb distance and false if not 
    def bomb_blast(self, dx, dy):
        # only looks is there is a placed bomb
        if self.bomb_location != None:
            # only looks 4 left and 4 right from the bomb location 
            for bx in range(-4, 5, 1):
                if ((dx == (self.bomb_location[0] + bx)) and (dy == (self.bomb_location[1]))):
                        return True
            # only looks 4 up and 4 down from the bomb location
            for by in range(-4, 5, 1):
                if ((dx == (self.bomb_location[0])) and (dy == (self.bomb_location[1] + by))):
                    return True
        return False 