from queue import PriorityQueue
from numpy import ndarray

# import numba as nb
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Node class
class Node:
    """Represents a Node used for the A* algorithm.
    It implements rich comparison methods for storing Nodes in a Priority Queue
    """
    def __init__(self, row, col, gscore=None):
        self.row = int(row)
        self.col = int(col)
        self.gscore = gscore or float('inf')
        self.parent = None

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __lt__(self, other):
        return self.gscore < other.gscore

    def __hash__(self):
        return hash((self.row, self.col))

    def __str__(self):
        return f"({self.row}, {self.col})"


class AStar:
    """An A* path-planning algorithm for line segmentation.

    Based on:
        https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf

    """
    def __init__(self, grid: ndarray, name: str="A*"):
        self.grid = AStar.minmax_scale(grid)  # ndarray of type float
        self.open = PriorityQueue()
        self.close = set()  # O(1) for searching elements
        self.step = 2

        self.start: Node = None
        self.end: Node = None

        self.name = name
        self.logger = logging.getLogger(self.name)

    @staticmethod
    def minmax_scale(grid: ndarray):
        """Rescales an image grid to range 0..1"""
        mn, mx = grid.min(), grid.max()
        return (grid - mn) / (mx - mn)

    def find(self, start: Node, goal: Node):
        """Performs the search algorithm, given the start and end points"""
        self.start = Node(start[0], start[1], gscore=0)
        self.goal = Node(goal[0], goal[1])

        start_score = AStar.heuristic(self.start, self.goal)
        self.open.put((start_score, self.start))

        self.logger.info(f"Starting the {self.name} algorithm")

        while not self.open.empty():

            _, current = self.open.get()
            self.close.add(current)

            if current == self.goal:
                self.logger.info(f"{self.name} has found a path")
                return AStar.get_path_to(current)
            
            self.find_successors(current)

        self.logger.warning(f"{self.name} couldn't find a path")
        return None

    def find_successors(self, current):
        for neighbor in self.get_neighbors(current):
            if neighbor not in self.close:
                new_gscore = current.gscore + self.get_score(current, neighbor, self.start)

                if neighbor.parent is None or new_gscore < neighbor.gscore:
                    neighbor.gscore = new_gscore
                    neighbor.parent = current

                    fscore = new_gscore + AStar.heuristic(neighbor, self.goal)
                    self.open.put((fscore, neighbor))

    @staticmethod
    def heuristic(current: Node, goal: Node):
        """Computes the heuristic score from the next node to the goal"""
        row_diff = (current.row - goal.row) ** 2
        col_diff = (current.col - goal.col) ** 2
        return 40 * (row_diff + col_diff) ** 0.5

    def get_neighbors(self, current: Node):
        r, c = current.row, current.col
        s = self.step
        max_r, max_c = self.grid.shape[:2]

        neighbors = (
            Node(r - s, c - s), Node(r - s, c), Node(r - s, c + s),
            Node(r, c - s), Node(r, c + s), Node(r + s, c - s),
            Node(r + s, c), Node(r + s, c + s)
        )

        return (n for n in neighbors if (n.row in range(0, max_r)) and (n.col in range(0, max_c)))

    def get_score(self, current: Node, neighbor: Node, start: Node):
        """Computes the overall score"""
        # Computes the accumulated score from the start point to the current node
        neighbor_start_diff = abs(neighbor.row - start.row)
        
        if current == neighbor:
            offset_1 = 10.0
        else:
            offset_1 = 14.0

        val = self.grid[neighbor.row, neighbor.col]
        # self.logger.info(f" ({neighbor.row} ({val}) {neighbor.col}) ")

        if int(val) == 1:
            offset_2 = 0.0
        elif int(val) == 0:
            offset_2 = 1.0

        tmp_1 = self.upward_obstacle(neighbor)
        tmp_2 = self.downward_obstacle(neighbor)

        offset_3 = 1 / (1 + min(tmp_1, tmp_2))
        offset_4 = 1 / (1 + min(tmp_1, tmp_2) ** 2)

        return 3 * neighbor_start_diff + offset_1 + 50 * offset_2 + 150 * offset_3 + 50 * offset_4

    def upward_obstacle(self, node):
        step = 1
        try:
            while(step <= 50):
                if self.grid[node.row - step, node.col] == 0:
                    return float(step)
                else:
                    step += 1
        except IndexError:
            # node.row is within less than 50 steps from the grid limits
            pass

        return float('inf')

    def downward_obstacle(self, node):
        step = 1
        try:    
            while step <= 50:
                if self.grid[node.row + step, node.col] == 0:
                    return float(step)
                else:
                    step += 1
        except IndexError:
            # node.row is within less than 50 steps from the grid limits
            pass

        return float('inf')

    @staticmethod
    def get_path_to(current: Node):
        """Retrieves the path from the starting node to the given node"""
        total_path = [[current.row, current.col]]

        while current.parent is not None:
            current = current.parent
            total_path.append([current.row, current.col])

        return total_path # , self.close
