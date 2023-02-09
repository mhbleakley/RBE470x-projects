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
    pathComplete = True
    moves = 1
    rnge = 2
    bombCycle = 0
    bombRecent = 100
    bomb_location = None
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
                        # if(self.bomb_location != None and 
                        # not self.bomb_blast(current[0] + dx, current[1] + dy)
                        # and not wrld.wall_at(current[0] + dx, current[1] + dy)
                        # and not wrld.monsters_at(current[0] + dx, current[1] + dy)
                        # and not wrld.bomb_at(current[0] + dx, current[1] + dy)
                        # and not wrld.explosion_at(current[0] + dx, current[1] + dy)):
                        #         # print("Frontier: " + str(current[0] + dx) + " : "+ str(current[1] + dy))
                        #         cells.append((current[0] + dx, current[1] + dy))
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
    
    def heuristic(self, goal, next):
        return max(abs(next[0] - goal[0]), abs(next[1] - goal[1]))
        # return sum(abs(val1-val2) for val1, val2 in zip(goal, next))
   
    def euclidean_distance(self, goal, next):
        return math.sqrt((goal[0]-next[0])**2 + (goal[1] - next[1])**2)

    def cost(self, wrld, current, next):
        move_cost = abs(current[0] - next[0]) + abs(current[1] - next[1])
        if wrld.wall_at(next[0],next[1]):
            move_cost += 16
            for dx in [-1, 0, 1]:
                if ((current[0] + dx >= 0) or (current[0] + dx < wrld.width())):
                    move_cost -= 3
        return move_cost
    
    def direction_after_bomb(self, wrld):
        found, mx, my = self.monster_in_range(wrld)
        monster = mx, my
        weighted_angles : list[tuple(tuple, float)] = []
        possible_angles = []
        for dx in [-1,1]:
            if ((self.x + dx >= 0) and (self.x + dx < wrld.width())):
                for dy in [-1,1]:    
                    if ((self.y + dy >= 0) and (self.y + dy < wrld.height())):
                        if (wrld.exit_at(self.x + dx, self.y + dy) or
                           wrld.empty_at(self.x + dx, self.y + dy) ):
                            possible_angles.append((self.x + dx, self.y + dy))
    
        for move in possible_angles:
            weight = self.euclidean_distance(monster, move)
            weighted_angles.append((move, weight))
        # print(weighted_angles)
        return self.best_move(weighted_angles)

    def best_move(self, weighted_angles):
        best_val = -99
        bestMove = [self.x, self.y]
        for move in weighted_angles:
            if move[1] > best_val:
                bestMove = move[0]
                best_val = move[1] 
        return bestMove

    def bomb_moves(self, wrld):
        # direction = 0
        # if (self.x != 0):
        #     # move left
        #     direction = -1
        # if (self.x != wrld.width() - 1):
        #     # move right
        #     direction = 1
        # if self.bombCycle == 5:
        #     self.move(direction,-1)
        # if self.bombCycle == 2:
        #     self.move(-direction,1)
        # if self.bombCycle == 1:
        #     self.move(direction,1)
        # if (self.bombCycle == 6 or self.bombCycle == 4 or self.bombCycle == 3):
        #     self.move(0,0)
        next_move = self.direction_after_bomb(wrld)
        # print(next_move)
        self.move(next_move[0], next_move[1])
        # print(self.bombCycle)
    
    def do(self, wrld):
        found, mx, my = self.monster_in_range(wrld)
        current_pos = self.x, self.y
        monster = mx, my
        # print("BOMBCylce " + str(self.bombCycle))
        if self.bombCycle == 3 and self.bomb_start:
            self.bombCycle = 0
            self.bomb_start = False
            self.bomb_location = None
        # print("IS FOund? " + str(found))
        if found:
            self.moves = 1
            self.path = [current_pos]
            if self.bombCycle == 0: 
                bomb_move = self.direction_after_bomb(wrld)
                # print("Bomb move: " + str(bomb_move))
                self.path.append(bomb_move)
                self.bomb_location = self.bombing()
                self.bomb_start = True
            else:
                new_move = self.exp_max(wrld,monster)
                # print("Expecti move: " + str(new_move))
                self.path.append(new_move) 
        else:
            self.path = self.makePath(wrld, self.astar(wrld))
            self.moves = 1
        found = False

        # print(self.path)
        if self.moves < len(self.path):
            next = self.path[self.moves]
            # print(self.path)   
            dx = next[0] - self.x
            dy = next[1] - self.y
            self.move(dx,dy)
            if self.bomb_start == True:
                self.bombCycle += 1
            self.moves += 1
            self.bombRecent +=1
        

    def makePath(self, wrld, came_from):
        current = wrld.exitcell
        path = []
        if current in came_from:              
            while current != (self.x, self.y):
                path.append(current)
                current = came_from[current]
            path.append((self.x, self.y))
            path.reverse()
            return path
        else:
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
    
    def astar(self, wrld):
        frontier = PriorityQueue()
        came_from = {}
        cost_so_far = {}

        frontier.put((self.x, self.y), 0)
        came_from[(self.x, self.y)] = None
        cost_so_far[(self.x, self.y)] = 0
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

    def exp_max(self, wrld, monster):
        possible_moves = self.exp_probabilities(wrld, monster)
        best_val = -99999
        new_move = (0,0)
        for move in possible_moves:
            if move[1] > best_val:
                best_val = move[1]
                new_move = move[0] 
        return new_move

    def exp_probabilities(self, wrld, monster):
        possible_moves : list[tuple(tuple, float)] = []
        current_position = self.x, self.y
        # Returns the possible moves that the character can do
        character_moves = self.look_for_empty_cell_bomb(wrld, current_position)
        # print("Currrent Position: " + str(current_position))
        # print("Possible Moves: " + str(character_moves))
        for character_move in character_moves:
            end_values = self.find_values(wrld, character_move, monster)
            chance_value = 0
            for values in end_values:
                chance_value += (values/len(end_values))
            possible_moves.append((character_move, chance_value))

        return possible_moves

    def find_values(self, wrld, possible_move, monster):
        # finds all possible monster moves
        possible_monster_moves = self.monster_moves(wrld, monster, 3)
        # created the base final values
        end_values = []

        inital_distance = self.heuristic(possible_move, monster)

        for monster_move in possible_monster_moves:
            next_distance = self.heuristic(possible_move, monster_move) 

            if next_distance == 0:
                value = -100
            else :
                scale_factor = 2
                value = scale_factor*(next_distance - inital_distance)

            end_values.append(value)
        return end_values

    def bombing(self):
        if self.bombCycle <= 0:
            self.place_bomb()
        return self.x , self.y
    
    def bomb_blast(self, dx, dy):
        if self.bomb_location != None:
            for bx in range(-4, 5, 1):
                if ((dx == (self.bomb_location[0] + bx)) and (dy == (self.bomb_location[1]))):
                        return True
            for by in range(-4, 5, 1):
                    # print("Bomb Locations : " + str(self.bomb_location[0] + bx) + " : " +  str(self.bomb_location[1] + by))
                    # print("Our Location : " + str(dx) + " : " +  str(dy))
                if ((dx == (self.bomb_location[0])) and (dy == (self.bomb_location[1] + by))):
                    return True
        return False 

    
    # def exp_max(self, wlrd, state):
    #     return max(self.exp_val(result(state,a)))
    
    # def exp_val(self, wrld, state):
    #     if terminal_test:
    #         return utility(state)
    #     v = 0
    #     for a in actions(state):
    #         p = probability(a)
    #         v = v + p*self.max_val(result(state,a))
    #     return v

    # def max_val(self, wrld, state):
    #     if terminal_test:
    #         v = -9999
    #         for a in actions(state)
    #             v = max(v, self.exp_val(result(state,a)))
    #         return v
