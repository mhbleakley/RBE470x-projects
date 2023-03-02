# This is necessary to find the main code
import sys
import math
sys.path.insert(0, '../bomberman')
# Import necessary stuff
import numpy as np
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import sigmoid, sigmoid_prime
from losses import loss_func, loss_func_prime
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from world import World

class TestCharacterNN(CharacterEntity):
    # moves to iteration through the list of paths 
    moves = 1
    # our depth range for looking for the monster
    rnge = 2
    # used to determine if the bomb explosion is gone
    bombCycle = 0
    # used to update the placed bomb location
    bomb_location = None
    # used to determine if the bomb has been dropped
    bomb_start = False
    # network
    learning_rate = 0.75
    epochs = 500
    gamma = 0.99
    net = Network()
    net.add(FCLayer(16, 10))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))
    net.add(FCLayer(10, 10))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))
    net.add(FCLayer(10, 10))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))          
    net.use(loss_func, loss_func_prime)
    target_net = Network()
    target_net.add(FCLayer(16, 10))
    target_net.add(ActivationLayer(sigmoid, sigmoid_prime))
    target_net.add(FCLayer(10, 10))
    target_net.add(ActivationLayer(sigmoid, sigmoid_prime))
    target_net.add(FCLayer(10, 10))
    target_net.add(ActivationLayer(sigmoid, sigmoid_prime))          
    target_net.use(loss_func, loss_func_prime)
    tn_update = 10

    def do(self, wrld):
        # detemines if the monster was found
        found, mx, my = self.monster_in_range(wrld)
        current_pos = self.x, self.y

        # FOUND STATE
        if found:
            # initializes values
            self.moves = 1
            self.path = [current_pos]
            x_train = self.get_state(wrld)
            qt = self.target_net.predict(self.get_state(wrld))
            for out in qt:
                qt[out] = qt[out]*self.gamma + self.rewards(wrld, self.x, self.y, qt)
            y_train = qt
            self.net.fit(x_train, y_train, self.epochs, self.learning_rate)
            self.epochs -= 1
            if self.epochs%self.tn_update == 0:
                self.target_net = self.net


        #  A* TO TARGET STATE
        else:
            # creates path with a
            self.path = self.makePath(wrld, self.astar(wrld))
            self.moves = 1
        self.mon_found = False

        # iterates through the path created 
        if self.moves < len(self.path):
            next = self.path[self.moves]  
            dx = next[0] - self.x
            dy = next[1] - self.y
            self.move(dx,dy)
            if self.bomb_start == True:
                self.bombCycle += 1
            self.moves += 1
     
    def get_state(self, wrld):
        state = [self.x, self.y]
        found, mx, my = self.monster_in_range(wrld)
        state.append(mx)
        state.append(my)
        
        state.append(self.look_for_empty_cell_states(wrld,(self.x, self.y)))

        state.append(self.heuristic((self.x, self.y), (mx, my)))

        self.path = self.makePath(wrld, self.astar(wrld))
        state.append(len(self.path))

        bomb = 0
        for dx in range(-4,4):
            if wrld.bomb_at(self.x + dx, self.y) or wrld.bomb_at(self.x, self.y +dx):
                bomb = 1
        state.append(bomb)

        if len(wrld.bomb) == 0 and len(wrld.explosions) == 0:
            state.append(1)
        else:
            state.append(0)    
        
        return state

    def rewards(self, wrld, dx, dy, qt):
        reward = 0
        found, mx, my = self.monster_in_range(wrld)
        q_target = 0
        return q_target

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
  
    # heuristic for the chebyshev distance from one gobal position to the next 
    def heuristic(self, goal, next):
        return max(abs(next[0] - goal[0]), abs(next[1] - goal[1]))

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

