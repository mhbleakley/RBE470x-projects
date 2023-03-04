# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
import csv
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter
from model import Linear_QNet, QTrainer 

with open("../games.csv", 'w') as csvfile:
    updater = csv.writer(csvfile)
    updater.writerow([0])
    csvfile.close()

with open("../record.csv", 'w') as csvfile:
    updater = csv.writer(csvfile)
    updater.writerow([-5000])
    csvfile.close()


model = Linear_QNet(13,256,13)
model.load_state_dict(torch.load('../project1/model/model.pth'), strict=False)

# Create the game
for i in range(100):
    random.seed(i) # TODO Change this if you want different random choices
    g = Game.fromfile('map.txt')
    # g.add_monster(StupidMonster("stupid", # name
    #                             "S",      # avatar
    #                             3, 9      # position
    # ))

    # # TODO Add your character

    g.add_monster(StupidMonster("stupid", # name
                                "S",      # avatar
                                3, 5,     # position
    ))
    g.add_monster(SelfPreservingMonster("aggressive", # name
                                        "A",          # avatar
                                        3, 13,        # position
                                        1             # detection range
    ))

    # TODO Add your character
    g.add_character(TestCharacter("me", # name
                                "C",  # avatar
                                0, 0  # position
    ))

    # Run!
    g.go(1)

    with open("../games.csv", 'w') as csvfile:
        updater = csv.writer(csvfile)
        updater.writerow([i + 1])
        csvfile.close()
    