# task 01
# min_max Algorithm

import math

def minimax(curDepth, nodeIndex, maxTurn, scores, targetDepth):
    if curDepth == targetDepth:
        print(f"Returning leaf node value {scores[nodeIndex]} at depth {curDepth}")
        return scores[nodeIndex]
    
    if maxTurn:
        print(f"Maximizer's turn at node index {nodeIndex}, depth {curDepth}")
        best = max(
            minimax(curDepth + 1, nodeIndex * 2, False, scores, targetDepth),
            minimax(curDepth + 1, nodeIndex * 2 + 1, False, scores, targetDepth)
        )
        print(f"Maximizer selects {best} at node index {nodeIndex}, depth {curDepth}")
        return best
    else:
        print(f"Minimizer's turn at node index {nodeIndex}, depth {curDepth}")
        best = min(
            minimax(curDepth + 1, nodeIndex * 2, True, scores, targetDepth),
            minimax(curDepth + 1, nodeIndex * 2 + 1, True, scores, targetDepth)
        )
        print(f"Minimizer selects {best} at node index {nodeIndex}, depth {curDepth}")
        return best

# Test the minimax algorithm
scores = [4,3,5,2,7,9,7,2]
treeDepth = int(math.log(len(scores), 2)) 

print("\nStarting Minimax Algorithm...\n")
optimal_value = minimax(0, 0, True, scores, treeDepth)
print(f"\nThe optimal value is: {optimal_value}")
