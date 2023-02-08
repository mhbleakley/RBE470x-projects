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
    rnge = 3

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
                           wrld.empty_at(current[0] + dx, current[1] + dy)):
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
    
    def cost(self, current, next):
        return abs(current[0] - next[0]) + abs(current[1] - next[1])

    def do(self, wrld):
        found, mx, my = self.monster_in_range(wrld)
        current_pos = self.x, self.y
        monster = mx, my
        if found:
            self.moves = 1
            self.path = [current_pos]
            # print("Monster is " + str(monster))
            new_move = self.exp_max(wrld,monster)
 
            self.path.append(new_move)   
            
            # for cell in self.path:
            #     self.set_cell_color(cell[0],cell[1], Fore.BLACK + Back.BLACK)
        else:
            self.path = self.makePath(wrld, self.astar(wrld))
            self.moves = 1
        # for cell in self.path:
        #     self.set_cell_color(cell[0],cell[1], Fore.RED + Back.BLACK)
        # print("This is the current pos " + str(current_pos))
        # print("This is the path " + str(self.path))
        found = False
        
        if self.moves < len(self.path):
            next = self.path[self.moves]
            dx = next[0] - self.x
            dy = next[1] - self.y
            self.move(dx,dy)
            self.moves += 1
        

    def makePath(self, wrld, came_from):
        current = wrld.exitcell
        path = []
        while current != (self.x, self.y):
            path.append(current)
            current = came_from[current]
        path.append((self.x, self.y))
        path.reverse()
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
            for next in self.look_for_empty_cell(wrld, current):
                new_cost = cost_so_far[current] + self.cost(current, next)
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
        # print("All possible moves with weight" + str(possible_moves))
        # print("Next Move should Be" + str(new_move))
        return new_move

    def exp_probabilities(self, wrld, monster):
        possible_moves : list[tuple(tuple, float)] = []
        current_position = self.x, self.y
        # Returns the possible moves that the character can do
        character_moves = self.look_for_empty_cell(wrld, current_position)
        # print(character_moves)
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
            # print("*****")
            # print("OUR initial CHEBEY " + str(inital_distance))
            # print("OUR poissible move " + str(possible_move))
            # print("MONSTER Move " + str(monster_move))
            # print("CHEYBEY Distance " + str(next_distance))
            # print("****")

            if next_distance == 0:
                value = -100
            else :
                scale_factor = 2
                value = scale_factor*(next_distance - inital_distance)

            end_values.append(value)
        return end_values

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
