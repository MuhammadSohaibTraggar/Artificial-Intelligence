class Node:
    def __init__(self, value):
        self.value = value
        self.child = []  

    def add_child(self, child_node):
        self.child.append(child_node)

    def dfs(self, start, goal):
        visited = []  
        stack = [start] 
        while stack:
            current = stack.pop() 
            
            if current not in visited:
                visited.append(current)
                print(f"Visiting {current.value}")
                if current.value == goal:
                    print(f"Goal {goal} found!")
                    return visited
                for neighbor in reversed(current.child):
                    if neighbor not in visited:
                        stack.append(neighbor)

        print(f"Goal {goal} not found.")
        return visited

a = Node('A')
b = Node('B')
c = Node('C')
d = Node('D')
e = Node('E')
f = Node('F')
g = Node('G')

a.add_child(b)
a.add_child(c)
b.add_child(d)
b.add_child(e)
c.add_child(f)
c.add_child(g)

# Running DFS from node A to find node F
dfs = Node('DFS')  
visited_nodes = a.dfs(a, 'F')
print("Visited Nodes:", [node.value for node in visited_nodes])
