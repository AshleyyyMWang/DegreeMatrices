import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


def create_graph_from_edges(edge_path):
    # Load the graph from the edge list
    G = nx.read_edgelist(edge_path, nodetype=int)
    return G


# Load the graph from a GraphML file
def load_graph_from_graphml(graphml_path):
    return nx.read_graphml(graphml_path)


# Function to create neighbor degree matrix for graph alignment
def create_neighbor_degree_matrix(graph, max_degree):
    num_nodes = len(graph.nodes())
    matrix = np.zeros((num_nodes, max_degree))

    for i, node in enumerate(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        neighbor_degrees = sorted([graph.degree(neighbor) for neighbor in neighbors], reverse=True)
        matrix[i, :len(neighbor_degrees)] = neighbor_degrees

    return matrix, list(graph.nodes())


# Function to align two matrices
def align_matrices(matrix1, matrix2):
    cost_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))

    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[0]):
            cost_matrix[i, j] = np.linalg.norm(matrix1[i] - matrix2[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


# Function to perform graph alignment
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


def convert_node_types(node_list, target_type):
    return [target_type(node) for node in node_list]


# Function to create sample subgraphs for alignment experiment
def create_sample_subgraphs(graph, node_list, percentage):

    # Example usage:
    node_list = convert_node_types(node_list, int)

    common_part = int(len(node_list) * percentage)
    diff_part = node_list[common_part:]
    num_diff = len(diff_part)

    sub_nodes_1 = node_list[:common_part] + diff_part[:int(0.5 * num_diff)]
    sub_nodes_2 = node_list[:common_part] + diff_part[int(0.5 * num_diff):]

    subgraph_1 = graph.subgraph(sub_nodes_1)
    subgraph_2 = graph.subgraph(sub_nodes_2)

    return subgraph_1, subgraph_2


# Function to calculate correct predictions based on alignment
def calculate_correct_predictions(alignment, subgraph_nodes, precision_half):
    correct_prediction_count = 0
    size = len(subgraph_nodes)
    test_precision = 2 * precision_half + 1
    upper_bound = size - precision_half - 1

    aligned_nodes = list(alignment.values())

    for node in alignment:
        prediction = alignment[node]
        actual = node
        precision_list = []
        if actual in aligned_nodes:
            if precision_half != 0:
                actual_index = aligned_nodes.index(actual)
                if precision_half <= actual_index <= upper_bound:
                    precision_list = aligned_nodes[actual_index - precision_half: actual_index + precision_half + 1]
                else:
                    if actual_index < precision_half:
                        precision_list = aligned_nodes[:test_precision]
                    elif actual_index > upper_bound:
                        precision_list = aligned_nodes[-test_precision:]
            else:
                precision_list = [actual]
            if prediction in precision_list:
                correct_prediction_count += 1

    return correct_prediction_count


# Function to run the alignment experiment
def run_alignment_experiment(graph, node_list, percentages, precision_half_list):
    results = {}

    for percentage in percentages:
        subgraph_1, subgraph_2 = create_sample_subgraphs(graph, node_list, percentage)
        alignment = graph_alignment(subgraph_1, subgraph_2)
        correct_counts = []

        for ph in precision_half_list:
            correct_prediction_count = calculate_correct_predictions(alignment, node_list, ph)
            correct_counts.append(correct_prediction_count)

        results[percentage] = correct_counts

    return results


# Function to generate results table
def generate_results_table(results, precision_half_list, num_nodes_list):
    table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{| l | " + " | ".join(["l" for _ in precision_half_list]) + " |}\n\n\\hline \n"
    table += "Precision to $n$ & " + " & ".join([str(2 * ph + 1) for ph in precision_half_list]) + " \\\\  \\hline \n"

    for percentage, correct_counts in results.items():
        num_nodes = num_nodes_list[percentages.index(percentage)]
        correct_percentages = [round(count / num_nodes, 4) for count in correct_counts]
        table += f"$p = {int(percentage * 100)}\\%$ & " + " & ".join(map(str, correct_percentages)) + " \\\\   \\hline\n"

    table += "\n\\end{tabular}\n\\caption{Correct Percentage for Facebook Degree Matrices Alignment}\n\\end{table}"
    return table


# Create the graph using only the edge list
edges_path = "/Users/ashleywang/Desktop/facebook/1912.edges"
mygraph_facebook = create_graph_from_edges(edges_path)
print("Full graph created from edge list.")

# Example usage
graphml_path = "/Users/ashleywang/Desktop/facebook_700.graphml"  # Replace with the actual file path

# Load the graph from the GraphML file
facebook_graph = load_graph_from_graphml(graphml_path)
print("Sampled graph loaded from GraphML file.")

node_list = list(facebook_graph.nodes())

# Define parameters
percentages = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
precision_half_list = [0, 1, 2, 3, 4, 5, 6, 7, 15]
num_nodes_list = [int(len(node_list) * p) for p in percentages]

# Run the alignment experiment
results = run_alignment_experiment(mygraph_facebook, node_list, percentages, precision_half_list)

# Generate the results table
results_table = generate_results_table(results, precision_half_list, num_nodes_list)
print(results_table)
