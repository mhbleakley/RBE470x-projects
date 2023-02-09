# This is necessary to find the main code
import sys
import math
from node import Node
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
    bombCycle = 0

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
    
    def cost(self, wrld, current, next):
        move_cost = abs(current[0] - next[0]) + abs(current[1] - next[1])
        if wrld.wall_at(next[0], next[1]):
            move_cost += 20
        return move_cost
    
    def bomb_moves(self, wrld):
        direction = 0
        if (self.x != 0):
            # move left
            direction = -1
        if (self.x != wrld.width() - 1):
            # move right
            direction = 1
        if self.bombCycle == 5:
            self.move(direction,-1)
        if self.bombCycle == 2:
            self.move(-direction,1)
        if self.bombCycle == 1:
            self.move(direction,1)
        if (self.bombCycle == 6 or self.bombCycle == 4 or self.bombCycle == 3):
            self.move(0,0)
    
    def do(self, wrld):
        found, mx, my = self.monster_in_range(wrld)
        if found:
            self.moves = 1
            root = Node(0)
            self.make_min(wrld, root, 2)
            res = self.expectimax(root, True)
            if res == self.expectimax(root.N, True):
                self.move(0,-1)
            if res == self.expectimax(root.S, True):
                self.move(0,1)
            if res == self.expectimax(root.E, True):
                self.move(1,0)
            if res == self.expectimax(root.W, True):
                self.move(-1,0)
            if res == self.expectimax(root.NW, True):
                self.move(-1,-1)
            if res == self.expectimax(root.NE, True):
                self.move(1,-1)
            if res == self.expectimax(root.SW, True):
                self.move(-1,1)
            if res == self.expectimax(root.SE, True):
                self.move(1,1)
            if res == self.expectimax(root.D, True):
                self.move(0,0)
        else:
            self.path = self.makePath(wrld, self.astar(wrld))
            self.moves = 1
            found = False
        if self.moves < len(self.path):
            next = self.path[self.moves]   
            if (wrld.wall_at(next[0], next[1]) and self.bombCycle == 0):
                self.place_bomb()
                self.bombCycle = 6
                self.bomb_moves(wrld)
            if (self.bombCycle != 0):
                self.bomb_moves(wrld)
                self.bombCycle -= 1       
            if self.bombCycle == 0:
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
                new_cost = cost_so_far[current] + self.cost(wrld, current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current
        return came_from
    
####################################################################################################

    # Initializing Nodes to None
    def newNode(self, v, dx, dy, mx, my, nd):

        temp = Node(v)
        temp.dx = nd.dx + dx
        temp.dy = nd.dy + dy
        temp.mx = nd.mx + mx
        temp.my = nd.my + my
        return temp

    # Getting expectimax
    def expectimax(self, node, is_min):

        # Condition for Terminal node
        if (node.N == None and node.NE == None and node.E == None and node.SE == None and node.S == None and node.SW == None and node.W == None and node.NW == None and node.B == None and node.D == None):
            return node.value
        
        # Minimizer node. Chooses the min from the
        # all sub-trees
        if (is_min):
            return min(self.expectimax(node.N, False), self.expectimax(node.NE, False), self.expectimax(node.E, False), self.expectimax(node.SE, False), self.expectimax(node.S, False), self.expectimax(node.SW, False), self.expectimax(node.W, False), self.expectimax(node.NW, False), self.expectimax(node.B, False), self.expectimax(node.D, False))

        # Chance node. Returns the average of
        # the all sub-trees
        else:
            return (self.expectimax(node.N, True), self.expectimax(node.NE, True), self.expectimax(node.E, True), self.expectimax(node.SE, True), self.expectimax(node.S, True), self.expectimax(node.SW, True), self.expectimax(node.W, True), self.expectimax(node.NW, True), self.expectimax(node.D, True))/9
        
    def look_for_traversable(self, wrld, current):
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
                                cells.append((dx,dy))
        # All done
        return cells

    def make_min(self, wrld, nd, depth):
        current = (self.x,self.y)
        safe = self.look_for_traversable(wrld,current)
        for moves in safe:
            if (0,-1) in moves:
                nd.N = self.newNode(self.make_chance(wrld, nd, depth), 0, -1, 0, 0, nd)
            if (0,1) in moves: 
                nd.S = self.newNode(self.make_chance(wrld, nd, depth), 0, 1, 0, 0, nd)
            if (1,0) in moves: 
                nd.E = self.newNode(self.make_chance(wrld, nd, depth), 1, 0, 0, 0, nd)
            if (-1,0) in moves: 
                nd.W = self.newNode(self.make_chance(wrld, nd, depth), -1, 0, 0, 0, nd)
            if (1,1) in moves:
                nd.SE = self.newNode(self.make_chance(wrld, nd, depth), 1, 1, 0, 0, nd)
            if (-1,1) in moves:
                nd.SW = self.newNode(self.make_chance(wrld, nd, depth), -1, 1, 0, 0, nd)
            if (1,-1) in moves:
                nd.NE = self.newNode(self.make_chance(wrld, nd, depth), 1, -1, 0, 0, nd)
            if (-1,-1) in moves:
                nd.NW = self.newNode(self.make_chance(wrld, nd, depth), -1, -1, 0, 0, nd)
            nd.B = self.newNode(self.make_chance(wrld, nd, depth), 0, 0, 0, 0, nd)

    def make_chance(self, wrld, nd, depth):
        current = (self.x,self.y)
        safe = self.look_for_empty_cell(wrld,current)
        if depth != 0:
            for moves in safe:
                if (0,-1) in moves:
                    nd.N = self.newNode(self.make_min(wrld, nd, depth), 0, -1, 0, 0, nd)
                if (0,1) in moves: 
                    nd.S = self.newNode(self.make_min(wrld, nd, depth), 0, 1, 0, 0, nd)
                if (1,0) in moves: 
                    nd.E = self.newNode(self.make_min(wrld, nd, depth), 1, 0, 0, 0, nd)
                if (-1,0) in moves: 
                    nd.W = self.newNode(self.make_min(wrld, nd, depth), -1, 0, 0, 0, nd)
                if (1,1) in moves:
                    nd.SE = self.newNode(self.make_min(wrld, nd, depth), 1, 1, 0, 0, nd)
                if (-1,1) in moves:
                    nd.SW = self.newNode(self.make_min(wrld, nd, depth), -1, 1, 0, 0, nd)
                if (1,-1) in moves:
                    nd.NE = self.newNode(self.make_min(wrld, nd, depth), 1, -1, 0, 0, nd)
                if (-1,-1) in moves:
                    nd.NW = self.newNode(self.make_min(wrld, nd, depth), -1, -1, 0, 0, nd)
                nd.D = self.newNode(self.make_min(wrld, nd, depth), 0, 0, 0, 0, nd)
        else:
            for moves in safe:
                if (0,1) in moves:
                    nd.N = self.newNode(self.value_final(wrld, nd), 0, -1, 0, 0, nd)
                if (0,-1) in moves: 
                    nd.S = self.newNode(self.value_final(wrld, nd), 0, 1, 0, 0, nd)
                if (1,0) in moves: 
                    nd.E = self.newNode(self.value_final(wrld, nd), 1, 0, 0, 0, nd)
                if (-1,0) in moves: 
                    nd.W = self.newNode(self.value_final(wrld, nd), -1, 0, 0, 0, nd)
                if (1,1) in moves:
                    nd.SE = self.newNode(self.value_final(wrld, nd), 1, 1, 0, 0, nd)
                if (-1,1) in moves:
                    nd.SW = self.newNode(self.value_final(wrld, nd), -1, 1, 0, 0, nd)
                if (1,-1) in moves:
                    nd.NE = self.newNode(self.value_final(wrld, nd), 1, -1, 0, 0, nd)
                if (-1,-1) in moves:
                    nd.NW = self.newNode(self.value_final(wrld, nd), -1, -1, 0, 0, nd)
                nd.D = self.newNode(self.value_final(wrld, nd), 0, 0, 0, 0, nd)
                    
    def value_final(self, wrld, nd):
        dangerValue = 0
        goal = wrld.exit_cell
        next = (nd.dx, nd.dy)
        if (abs(nd.dx - nd.mx) < 3) and (abs(nd.dy - nd.my) < 3):
            dangerValue = 1000
        return dangerValue + self.heuristic(goal, next)