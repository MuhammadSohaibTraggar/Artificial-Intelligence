# task 06
# bfs without using queue and node
graph = {
    'D': ['E', 'F'],
    'E': ['A', 'C'],
    'F': ['G'],
    'A': [],
    'C': [],
    'G': []
}

def bfs(graph, start_node):
    visited = set()
    stack = [start_node]

    while stack:
        current_node = stack.pop()
        print(current_node)
        visited.add(current_node)
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                stack.append(neighbor)

print("BFS traversal output (without queue and node):")
bfs(graph, 'D')