# task 06
# bfs with queue and node
# Define the Node class
from collections import deque
class Node:
    def __init__(self, value):
        self.value = value
        self.visited = False
        self.neighbors = []

def bfs(start_node):
    queue = []
    start_node.visited = True
    queue.append(start_node)

    while queue:
        current_node = queue.pop(0)
        print(current_node.value)
        for neighbor in current_node.neighbors:
            if not neighbor.visited:
                neighbor.visited = True
                queue.append(neighbor)

def reset_visited(nodes):
    for node in nodes:
        node.visited = False

root = Node('J')
b = Node('E')
c = Node('F')
d = Node('A')
e = Node('C')
f = Node('G')

root.neighbors = [b, c]
b.neighbors = [d, e]
c.neighbors = [f]
all_nodes = [root, b, c, d, e, f]
print("BFS traversal output:")
bfs(root)

reset_visited(all_nodes)