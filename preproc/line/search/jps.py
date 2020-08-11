from .astar import AStar, Node

# import numba as nb
import numpy as np

class JumpPoint(AStar):
    """A Jump Point path-planning algorithm for line segmentation. It is an
    optimization of the A* algorithm for uniform-cost grids.

    Based on:
        https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf

    """
    def __init__(self, grid):
        super().__init__(grid, name="JPS")

    def find_successors(self, node):
        neighbors = self.find_neighbors(node)

        for neighbor in neighbors:
            jump_node = self.jump(neighbor, node)

            if jump_node is None:
                continue

            if jump_node in self.close:
                continue

            new_gscore = node.gscore + self.get_score(node, jump_node, self.start)

            if jump_node.parent is not None or new_gscore < neighbor.gscore:
                neighbor.gscore = new_gscore
                fscore = new_gscore + self.heuristic(jump_node, self.goal)
                jump_node.parent = node
                self.open.put(jump_node, fscore)

    def in_bounds(self, node):
        max_r, max_c = self.grid.shape[:2]
        return node.row in range(0, max_r) and node.col in range(0, max_c)

    def jump(self, node, parent):
        if not node or not self.in_bounds(node):
            return None

        r, c = node.row, node.col
        dr, dc = r - parent.row, c - parent.col
        #  print(node, parent, dr, dc)

        if self.wall([r, c]):
            return None

        if node == self.goal:
            return node

        # Diagonal case
        if dr != 0 and dc != 0:
            if ((not self.wall([r - dr, c - dc]) and self.wall([r, c - dc])) or
                    (not self.wall([r + dr, c + dc]) and self.wall([r + dr, c]))):
                return node
        # Horizontal case
        if dc != 0:
            if ((not self.wall([r - 1, c + dc]) and self.wall([r - 1, c])) or
                    (not self.wall([r + 1, c + dc]) and self.wall([r + 1, c]))):
                return node
        # Vertical case
        if dr != 0:
            if ((not self.wall([r - dr, c - 1]) and self.wall([r, c - 1])) or
                    (not self.wall([r - dr, c + 1]) and self.wall([r, c + 1]))):
                return node
        # Recursive horizontal/vertical search
        if dr != 0 and dc != 0:
            if self.jump(Node(r, c + dc), node):
                return node
            if self.jump(Node(r + dr, c), node):
                return node
        # Recursive diagonal search
        if not self.wall([r, c + dc]) or not self.wall([r + dr, c]):
            return self.jump(Node(r + dr, c + dc), node)

    def find_neighbors(self, node):
        if node.parent is not None:
            # get neighbors based on direction
            neighbors = []
            r, c = node.row, node.col
            dr, dc = self.direction(node)

            # Diagonal direction
            if dr != 0 and dc != 0:
                neighbors.append(Node(r, c + dc))
                neighbors.append(Node(r + dr, c))
                neighbors.append(Node(r + dr, c + dc))
                # forced neighbors
                if self.wall([r, c - dc]):
                    neighbors.append(Node(r + dr, c - dc))
                if self.wall([r - dr, c]):
                    neighbors.append(Node(r - dr, c + dc))
            # Horizontal direction
            elif dc != 0:
                neighbors.append(Node(r, c + dc))
                # forced neighbors
                if self.wall([r + 1, c]):
                    neighbors.append(Node(r + 1, c + dc))
                if self.wall([r - 1, c]):
                    neighbors.append(Node(r - 1, c + dc))
            # Vertical direction
            elif dr != 0:
                neighbors.append(Node(r + dr, c))
                # forced neighbors
                if self.wall([r,  c + dc]):
                    neighbors.append(Node(r + dr, c + dc))
                if self.wall([r, c - dc]):
                    neighbors.append(Node(r + dr, c - dc))

            return neighbors
        else:
            # does not have a parent, return all the neighbors
            return self.get_neighbors(node)

    def direction(self, node):
        dr = (node.row - node.parent.row)/max(abs(node.row - node.parent.row), 1)
        dc = (node.col - node.parent.col)/max(abs(node.col - node.parent.col), 1)

        return dr, dc

    def wall(self, node):
        r, c = node
        try:
            if self.grid[r, c] == 0:
                return True
            else:
                return False
        except IndexError:
            return True
