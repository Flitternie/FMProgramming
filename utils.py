import random
import torch
import numpy as np 
from collections import deque, defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def order_tree(tree):
    # Build the graph and compute in-degrees of nodes
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    node_keys = {}

    # Assign indices to the nodes based on their position in the list for easy access
    for idx, (node, children) in enumerate(tree):
        node_keys[idx] = node
        for child in children:
            graph[child].append(idx)
            in_degree[idx] += 1

    # Find all nodes with no incoming edges (roots of the original positions in tree)
    queue = deque([i for i in range(len(tree)) if in_degree[i] == 0])
    topological_order = []
    root_node = None

    # Mapping from old indices to new indices
    index_mapping = {}

    while queue:
        node = queue.popleft()
        index_mapping[node] = len(topological_order)  # Map old index to the current position
        topological_order.append(node)
        # In the last iteration, this will be the root node since it's the only node without children processed last
        root_node = node
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles (which should not happen in a valid tree structure)
    if len(topological_order) != len(tree):
        raise Exception("There is a cycle in the tree!")

    # Rebuild tree using the new indices
    reordered_tree = []
    for old_index in topological_order:
        node_name, old_children = tree[old_index]
        new_children = [index_mapping[child] for child in old_children]
        reordered_tree.append((node_name, new_children))

    # Return the ordered list of nodes and the index of the root node (new index)
    return reordered_tree, index_mapping[root_node]