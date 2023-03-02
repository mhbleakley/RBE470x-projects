# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
import csv
from game import Game
from monsters.stupid_monster import StupidMonster

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter

with open("../games.csv", 'w') as csvfile:
    updater = csv.writer(csvfile)
    updater.writerow([0])
    csvfile.close()

# Create the game
for i in range(1000000):
    # Create the game
    random.seed(i) # TODO Change this if you want different random choices
    g = Game.fromfile('map.txt')
    g.add_monster(StupidMonster("stupid", # name
                                "S",      # avatar
                                3, 9      # position
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
    