import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import permutations
import random


# Function to load G_{r} from a GraphML file
def load_graph_from_graphml(graphml_path):
    return nx.read_graphml(graphml_path)


# Random Walk sampling method
def random_walk(mygraph, subgraph_size, max_iter=100):
    progress = 0
    current_node = np.random.choice(list(mygraph.nodes()))
    network_sub_nodes = [current_node]

    iterations = 0  # Track iterations without progress
    while len(network_sub_nodes) < subgraph_size:
        neighbors = list(mygraph.neighbors(current_node))

        if neighbors:
            next_node = np.random.choice(neighbors)
        else:
            next_node = np.random.choice(list(mygraph.nodes()))

        # Check if the next node is already in the subgraph
        if next_node not in network_sub_nodes:
            network_sub_nodes.append(next_node)
            progress += 1
            iterations = 0  # Reset iteration counter since progress was made
        else:
            iterations += 1  # Increment if no new node was added

        # If stuck, reset to a random new node after max_iter iterations
        if iterations >= max_iter:
            potential_nodes = set(mygraph.nodes()) - set(network_sub_nodes)
            if potential_nodes:
                next_node = np.random.choice(list(potential_nodes))
                # print(f"Resetting to a new node: {current_node}")
            else:
                print("No more new nodes to reset to; exiting.")
                break
            iterations = 0  # Reset iteration count after switching nodes

        # Update the current node
        current_node = next_node

    network_sub = mygraph.subgraph(network_sub_nodes).copy()
    print("Random walk completed.")
    return network_sub


# Randomly delete edges with a specified probability
def delete_edges_randomly(graph, deletion_probability):
    modified_graph = graph.copy()
    edges_to_remove = [edge for edge in graph.edges if random.random() < deletion_probability]
    modified_graph.remove_edges_from(edges_to_remove)
    return modified_graph


# Create degree matrix for alignment
def create_neighbor_degree_matrix(graph, max_degree):
    num_nodes = len(graph.nodes())
    matrix = np.zeros((num_nodes, max_degree))

    for i, node in enumerate(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        neighbor_degrees = sorted([graph.degree(neighbor) for neighbor in neighbors], reverse=True)
        matrix[i, :len(neighbor_degrees)] = neighbor_degrees

    return matrix, list(graph.nodes())


# Align two matrices for graph alignment (Munkres)
def align_matrices(matrix1, matrix2):
    cost_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))

    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[0]):
            cost_matrix[i, j] = np.linalg.norm(matrix1[i] - matrix2[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


# Perform graph alignment
def graph_alignment(graph1, graph2):
    max_degree = max(
        max([len(list(graph1.neighbors(node))) for node in graph1.nodes()]),
        max([len(list(graph2.neighbors(node))) for node in graph2.nodes()])
    )
    matrix1, nodes1 = create_neighbor_degree_matrix(graph1, max_degree)
    matrix2, nodes2 = create_neighbor_degree_matrix(graph2, max_degree)

    row_ind, col_ind = align_matrices(matrix1, matrix2)
    node_mapping = {nodes1[i]: nodes2[j] for i, j in zip(row_ind, col_ind)}
    return node_mapping


# Compute the signature vector for node equivalence (use this if counting geometrically)
def compute_signature_vector(graph, node):
    neighbors = list(graph.neighbors(node))
    signature_vector = []

    for neighbor in neighbors:
        neighbor_degrees = [graph.degree(n) for n in graph.neighbors(neighbor)]
        signature_vector.append(sorted(neighbor_degrees))

    return signature_vector


# Check if two nodes are equivalent by comparing their signature vectors
def are_nodes_equivalent(graph, node1, node2):
    if graph.degree(node1) in [1, 2, 3] and graph.degree(node2) in [1, 2, 3]:
        signature1 = compute_signature_vector(graph, node1)
        signature2 = compute_signature_vector(graph, node2)
        for perm in permutations(signature1):
            if list(perm) == signature2:
                return True
            else:
                return False
    else:
        return False


# Calculate correct predictions based on alignment
def calculate_correct_predictions(alignment, graph):
    correct_prediction_count = 0
    for node1 in alignment:
        predicted_node2 = alignment[node1]
        if predicted_node2 == node1 or are_nodes_equivalent(graph, node1, predicted_node2):
            correct_prediction_count += 1
    return correct_prediction_count


# Run the full alignment experiment
def run_alignment_experiment(graph, target_node_count, deletion_probability):
    # Step 1: Sample G_{s} = G_{1}
    sampled_subgraph = random_walk(graph, target_node_count)
    print("First subgraph created.")
    print("G1 node count: ", len(sampled_subgraph.nodes()))
    print("G1 edge count: ", len(sampled_subgraph.edges()))

    # Step 2: Randomly delete edges in G_{1} to get G_{2}
    final_subgraph = delete_edges_randomly(sampled_subgraph, deletion_probability)
    print("Final G2 subgraph created.")
    print("G2 node count: ", len(final_subgraph.nodes()))
    print("G2 edge count: ", len(final_subgraph.edges()))

    # Step 3: Alignment with DMC
    alignment = graph_alignment(sampled_subgraph, final_subgraph)

    # Step 4: Calculate the number of correct predictions
    correct_predictions = calculate_correct_predictions(alignment, sampled_subgraph)
    return correct_predictions


# Main execution
graphml_path = "/Path/to/combined_ppi.graphml"
main_graph = load_graph_from_graphml(graphml_path)
print("G_{r} loaded.")
target_node_count = 3890
deletion_probability = 0.01

round = 0
for i in range(10):
    round += 1
    print("Round ", round)
    correct_predictions = run_alignment_experiment(main_graph, target_node_count, deletion_probability)
    # Print results
    print("Correct predictions:", correct_predictions)


