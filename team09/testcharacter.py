# This is necessary to find the main code
import sys
import math
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from world import World

class TestCharacter(CharacterEntity):
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
    

    def do(self, wrld):
        # detemines if the monster was found
        found, mx, my = self.monster_in_range(wrld)
        current_pos = self.x, self.y
        monster = mx, my
        # Resets the bomb counter
        if self.bombCycle == 3 and self.bomb_start:
            self.bombCycle = 0
            self.bomb_start = False
            self.bomb_location = None

        # FOUND STATE
        if found:
            # initializes values
            self.moves = 1
            self.path = [current_pos]

            # BOMB AND RUN STATE
            if self.bombCycle == 0: 
                # places the bomb and moves away from monster
                bomb_move = self.direction_after_bomb(wrld)
                self.path.append(bomb_move)
                self.bomb_location = self.bombing()
                self.bomb_start = True

            # EXPECTIMAX AWAY STATE
            else:
                # expectimaxes away from the monster
                new_move = self.exp_max(wrld,monster)
                self.path.append(new_move) 

        #  A* TO TARGET STATE
        else:
            # creates path with a
            self.path = self.makePath(wrld, self.astar(wrld))
            self.moves = 1
        found = False

        # iterates through the path created 
        if self.moves < len(self.path):
            next = self.path[self.moves]  
            dx = next[0] - self.x
            dy = next[1] - self.y
            self.move(dx,dy)
            if self.bomb_start == True:
                self.bombCycle += 1
            self.moves += 1
        
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
    
    # Approximate Q-learning
    def appQlearning(self, state, action):
        return 
        