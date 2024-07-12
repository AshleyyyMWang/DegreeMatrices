import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


def create_sample_graph(edge_path, node_path):
    mygraph = nx.read_edgelist(edge_path)
    with open(node_path, "r") as node_file_f:
        for lines in node_file_f:
            parts_f = lines.strip().split(maxsplit=1)
            node_id_f = parts_f[0]
            node_name_f = parts_f[1] if len(parts_f) > 1 else None
            if node_id_f in mygraph:
                mygraph.nodes[node_id_f]['name'] = node_name_f
            else:
                print(f"Node {node_id_f} not found in edge list.")
    return mygraph


def create_neighbor_degree_matrix(graph, max_degree):
    num_nodes = len(graph.nodes())
    matrix = np.zeros((num_nodes, max_degree))

    for i, node in enumerate(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        neighbor_degrees = sorted([graph.degree(neighbor) for neighbor in neighbors], reverse=True)
        matrix[i, :len(neighbor_degrees)] = neighbor_degrees

    return matrix, list(graph.nodes())


def align_matrices(matrix1, matrix2):
    cost_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))

    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[0]):
            cost_matrix[i, j] = np.linalg.norm(matrix1[i] - matrix2[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


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


def create_sample_subgraphs(graph, node_list, percentage):
    common_part = int(len(node_list) * percentage)
    diff_part = node_list[common_part:]
    num_diff = len(diff_part)

    sub_nodes_1 = node_list[:common_part] + diff_part[:int(0.5 * num_diff)]
    sub_nodes_2 = node_list[:common_part] + diff_part[int(0.5 * num_diff):]

    subgraph_1 = graph.subgraph(sub_nodes_1)
    subgraph_2 = graph.subgraph(sub_nodes_2)

    return subgraph_1, subgraph_2


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
    print(f"Correctly aligns {correct_prediction_count} pairs up to precision of {test_precision}.")
    return correct_prediction_count


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


# Load the graphs
mygraph_flickr = create_sample_graph("/Path/to/flickr.edges", "/Path/to/flickr.nodes")
flickr_5000 = nx.read_graphml("/Path/to/flickr_graph.graphml")
flickr_5000_node_list = list(flickr_5000.nodes())

# Define parameters
percentages = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
precision_half_list = [0, 1, 2, 3, 4, 5, 6, 7, 15]
num_nodes_list = [4250, 4375, 4500, 4635, 4750, 4875]

# Run the alignment experiment
results = run_alignment_experiment(mygraph_flickr, flickr_5000_node_list, percentages, precision_half_list)
