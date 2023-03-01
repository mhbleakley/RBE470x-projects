import sys
import math

sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

import torch
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from world import World
from game import Game

from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter

import random
import numpy as np
from collections import deque

from model import Linear_QNet, QTrainer 
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class TrainAgent:
    def __init__(self, mapfile, monster_name1, monster_position1, monster_name2, monster_position2):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen = MAX_MEMORY) # pop left overloading memory
        self.game = Game.fromfile(mapfile) 

        if(monster_position1[0] != -999 or monster_position1[1] != -999 ):
            if(monster_name1 == "aggressive"):
                self.game.add_monster(SelfPreservingMonster(monster_name1, # name
                                                    "A",      # avatar
                                                    monster_position1[0], monster_position1[1],     # position
                                                    1             # detection range
                ))
            elif(monster_name1 == "selfpreserving"):
                self.game.add_monster(SelfPreservingMonster(monster_name1, # name
                                                    "S",          # avatar
                                                    monster_position1[0], monster_position1[1],        # position
                                                    1            # detection range
                ))
            else : 
                self.game.add_monster(StupidMonster(monster_name1, # name
                                                    "S",          # avatar
                                                    monster_position1[0], monster_position1[1],        # position
                ))
        if(monster_position2[0] != -999 or monster_position2[1] != -999 ):
            if(monster_name2 == "aggressive"):
                self.game.add_monster(SelfPreservingMonster(monster_name2, # name
                                                    "A",      # avatar
                                                    monster_position2[0], monster_position2[1],     # position
                                                    1             # detection range
                ))
            elif(monster_name2 == "selfpreserving"):
                self.game.add_monster(SelfPreservingMonster(monster_name2, # name
                                                    "S",          # avatar
                                                    monster_position2[0], monster_position2[1],        # position
                                                    1            # detection range
                ))
            else : 
                self.game.add_monster(StupidMonster(monster_name2, # name
                                                    "S",          # avatar
                                                    monster_position2[0], monster_position2[1],        # position
                ))


        self.game.add_character(TestCharacter("me", # name
                              "C",  # avatar
                              0, 0  # position
        ))
        self.model = Linear_QNet(11,256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self, World):

        pass

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
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else :
            state0 = torch.tensor(state, dtype=torch.float)
            prediction  = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0 
    agent = TrainAgent()
    ## game = 
    while True:
        self.game.go(0) 
        #get old state 
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game 
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remeber(state_old, final_move, reward, state_new, done)

        self.g.done:


        if done:
            #train long memory, plot the result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()
            print('Game', agent.n_games, 'Score', score, ' Record:' , record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_scores, plot_mean_score)
            

if __name__ == '__main__':
    train()
