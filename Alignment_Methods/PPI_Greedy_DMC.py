import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import random


# Function to load G_{r} from a GraphML file
def load_graph_from_graphml(graphml_path):
    return nx.read_graphml(graphml_path)


# Random walk to create G_{s}
def random_walk(mygraph, subgraph_size, max_iter=100):
    progress = 0
    current_node = np.random.choice(list(mygraph.nodes()))
    network_sub_nodes = [current_node]

    print(f"Starting random walk from node: {current_node}")

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
            else:
                print("No more new nodes to reset to; exiting.")
                break
            iterations = 0  # Reset iteration count after switching nodes

        # Update the current node
        current_node = next_node

    network_sub = mygraph.subgraph(network_sub_nodes).copy()
    print("Random walk completed.")
    return network_sub



# Randomly delete edges in a graph
def delete_edges_randomly(graph, deletion_probability=0.01):
    modified_graph = graph.copy()
    edges_to_remove = [edge for edge in graph.edges if random.random() < deletion_probability]
    modified_graph.remove_edges_from(edges_to_remove)
    return modified_graph


# Create degree matrix
def create_neighbor_degree_matrix(graph, max_degree):
    num_nodes = len(graph.nodes())
    matrix = np.zeros((num_nodes, max_degree))

    for i, node in enumerate(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        neighbor_degrees = sorted([graph.degree(neighbor) for neighbor in neighbors], reverse=True)
        matrix[i, :len(neighbor_degrees)] = neighbor_degrees

    return matrix, list(graph.nodes())


def auction_phase(cost_matrix, epsilon=0.1):
    n = cost_matrix.shape[0]
    prices = np.zeros(n)
    assignment = [-1] * n  # Initially unassigned

    for i in range(n):
        if assignment[i] == -1:
            best_obj = None
            second_best_obj = None
            best_profit = float('-inf')
            second_best_profit = float('-inf')

            # Find best and second best objects for person i
            for j in range(n):
                profit = cost_matrix[i][j] - prices[j]
                if profit > best_profit:
                    second_best_profit = best_profit
                    second_best_obj = best_obj
                    best_profit = profit
                    best_obj = j
                elif profit > second_best_profit:
                    second_best_profit = profit

            # Update price of best object
            if best_obj is not None:
                prices[best_obj] += epsilon + (best_profit - second_best_profit)
                assignment[i] = best_obj

    return assignment, prices

# Function to perform Hungarian method after auction phase
def hungarian_refinement(cost_matrix, prices):
    reduced_cost_matrix = cost_matrix - prices
    row_ind, col_ind = linear_sum_assignment(reduced_cost_matrix)
    return row_ind, col_ind

# Align two matrices for graph alignment
def align_matrices(matrix1, matrix2, epsilon=0.1):
    cost_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))
    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[0]):
            cost_matrix[i, j] = np.linalg.norm(matrix1[i] - matrix2[j])

    initial_assignment, prices = auction_phase(cost_matrix, epsilon)

    # Perform the Hungarian method for refinement
    row_ind, col_ind = hungarian_refinement(cost_matrix, prices)
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


# Calculate correct predictions based on alignment
def calculate_correct_predictions(alignment, graph):
    correct_prediction_count = 0
    for node1, predicted_node2 in alignment.items():
        if predicted_node2 == node1:
            correct_prediction_count += 1
    return correct_prediction_count


# Run the alignment experiment
def run_alignment_experiment(graph, target_node_count, deletion_probability):
    sampled_subgraph = random_walk(graph, target_node_count)
    print("First subgraph created with node count:", len(sampled_subgraph.nodes()), " Edge Count: ", len(sampled_subgraph.edges()))

    modified_subgraph = delete_edges_randomly(sampled_subgraph, deletion_probability=deletion_probability)
    print("Second subgraph created with random edge deletion applied. Node count:", len(modified_subgraph.nodes()), " Edge Count: ", len(modified_subgraph.edges()))

    alignment = graph_alignment(sampled_subgraph, modified_subgraph)
    correct_predictions = calculate_correct_predictions(alignment, sampled_subgraph)
    return correct_predictions


# Main execution
graphml_path = "/Path/to/combined_ppi.graphml"
main_graph = load_graph_from_graphml(graphml_path)
print("Graph loaded from GraphML file.")

target_node_count = 3890
deletion_probability = 0.01
for i in range(10):
    correct_predictions = run_alignment_experiment(main_graph, target_node_count, deletion_probability)
    print(f"Round {i + 1}: Correct predictions - {correct_predictions}")

