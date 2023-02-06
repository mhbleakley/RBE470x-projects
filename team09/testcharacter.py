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
    rnge = 4

    def look_for_empty_cell(self, wrld, current, monster):
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
                            for mx in [-2, -1, 0, 1, 2]:
                                for my in [-2, -1, 0, 1, 2]:
                                    if not ((current[0] + dx == monster[0] + mx) or
                                        (current[1] + dy == monster[1] + my)):

                                        cells.append((current[0] + dx, current[1] + dy))
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
                            return (True, dx, dy)
        # Nothing found
        return (False, 0, 0)
    
    def heuristic(self, goal, next):
        return sum(abs(val1-val2) for val1, val2 in zip(goal, next))
    
    def cost(self, current, next):
        return abs(current[0] - next[0]) + abs(current[1] - next[1])

    def do(self, wrld):
        found, mx, my = self.monster_in_range(wrld)
        if found:
            self.pathComplete = True
            self.moves = 1
            for cell in self.path:
                self.set_cell_color(cell[0],cell[1], Fore.BLACK + Back.BLACK)
        if self.pathComplete:
            self.path = self.makePath(wrld, self.astar(wrld, mx, my))
            self.pathComplete = False
        for cell in self.path:
            self.set_cell_color(cell[0],cell[1], Fore.RED + Back.BLACK)
        if self.moves < len(self.path):
            next = self.path[self.moves]
            dx = next[0] - self.x
            dy = next[1] - self.y
            self.move(dx,dy)
            self.moves += 1
        print(self.path)

    def makePath(self, wrld, came_from):
        current = wrld.exitcell
        path = []
        while current != (self.x, self.y):
            path.append(current)
            current = came_from[current]
        path.append((self.x, self.y))
        path.reverse()
        return path
    
    def astar(self, wrld, mx, my):
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
            for next in self.look_for_empty_cell(wrld, current, monster):
                new_cost = cost_so_far[current] + self.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current
        return came_from
    
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
