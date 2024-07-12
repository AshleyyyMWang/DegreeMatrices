import networkx as nx
import numpy as np


# change variables "percentage" and "precision_half" for different experiments.


def create_sample_graph(edge_path, node_path):
    mygraph = nx.read_edgelist(edge_path)
    node_file_path_f = node_path
    node_file_f = open(node_file_path_f, "r")
    for lines in node_file_f:
        parts_f = lines.strip().split(maxsplit=1)
        node_id_f = parts_f[0]
        node_name_f = parts_f[1] if len(parts_f) > 1 else None
        if node_id_f in mygraph:
            mygraph.nodes[node_id_f]['name'] = node_name_f
        else:
            print(f"Node {node_id_f} not found in edge list.")
    node_file_f.close()
    return mygraph


mygraph_flickr = create_sample_graph("/Path/to/flickr.edges", "/Path/to/flickr.nodes")
print("Flickr sample graph loaded.")
flickr_5000 = nx.read_graphml("/Path/to/flickr_graph.graphml")
flickr_5000_node_list = list(flickr_5000.nodes())


# FIRST VARIABLE.
percentage = 0.7


common_part = 5000 * percentage
diff_part = []
for j in range(int(common_part), 5000):
    diff_part.append(flickr_5000_node_list[j])
num_diff = len(diff_part)


# creating the two subgraphs
flickr1_sub_nodes = []
for i in range(int(common_part)):
    flickr1_sub_nodes.append(flickr_5000_node_list[i])
for k in range(int(0.5 * num_diff)):
    flickr1_sub_nodes.append(diff_part[k])
flickr_sub_1 = mygraph_flickr.subgraph(flickr1_sub_nodes)
print("First flickr subgraph created.")


flickr2_sub_nodes = []
for a in range(int(common_part)):
    flickr2_sub_nodes.append(flickr_5000_node_list[a])
for b in range(int(0.5 * num_diff), num_diff):
    flickr2_sub_nodes.append(diff_part[b])
flickr_sub_2 = mygraph_flickr.subgraph(flickr2_sub_nodes)
print("Second flickr subgraph created.")


# now we start alignment by first comparing the degree of each node
def get_degree_to_node_map(graph):
    degree_node_map = {}
    for point in graph.nodes():
        degree_0 = graph.degree(point)
        if degree_0 not in degree_node_map:
            degree_node_map[degree_0] = []
        degree_node_map[degree_0].append(point)
    return degree_node_map


map_flickr_1 = get_degree_to_node_map(flickr_sub_1)
map_flickr_2 = get_degree_to_node_map(flickr_sub_2)
print("The degree to node maps for flickr_1 and flickr_2 are loaded. ")


# aligning from the smallest possible degree to largest
degree_ordered_1 = sorted(map_flickr_1)
degree_ordered_2 = sorted(map_flickr_2)


# this function is for creating a dictionary, with node as key and average degree of neighbors as output
def average_neighbor_degree(G):
    avg_degs = {}
    for nod in G.nodes():
        neighbors = list(G.neighbors(nod))
        if neighbors:  # Check if the node has neighbors
            avg_degree = np.mean([G.degree(neighbor) for neighbor in neighbors])
        else:
            avg_degree = 0
        avg_degs[nod] = avg_degree
    return avg_degs


def reordering_avg_nd(ordered_degree_list, dn_map, some_sub, new_list):
    for degree in ordered_degree_list:
        node_list = dn_map[degree]
        sub_nodes_graph = some_sub.subgraph(node_list)
        avg_degrees = average_neighbor_degree(sub_nodes_graph)
        sorted_nodes = sorted(avg_degrees, key=avg_degrees.get, reverse=True)
        for node in sorted_nodes:
            new_list.append(node)
    return new_list


flickr_new_1 = []
flickr_new_1 = reordering_avg_nd(degree_ordered_1, map_flickr_1, flickr_sub_1, flickr_new_1)
print("Flickr_1 node reordering done.")

flickr_new_2 = []
flickr_new_2 = reordering_avg_nd(degree_ordered_2, map_flickr_2, flickr_sub_2, flickr_new_2)
print("Flickr_2 node reordering done.")


degree_align_dict = dict(zip(flickr_new_1, flickr_new_2))


# SECOND VARIABLE.
precision_half_list = [0,1,2,3,4,5,6,7,15]

for ph in precision_half_list:
    precision_half = ph

    correct_prediction_count = 0
    size = int(common_part + 0.5 * len(diff_part))
    test_precision = 2 * precision_half + 1
    upper_bound = size - precision_half - 1

    for f in degree_align_dict:
        prediction = degree_align_dict[f]
        actual = f
        precision_list = []
        if actual in flickr_new_2:
            if precision_half != 0:
                actual_index = flickr_new_2.index(actual)
                # check index range.
                if precision_half <= actual_index <= upper_bound:
                    for a in range(actual_index - precision_half, actual_index + precision_half + 1):
                        precision_list.append(flickr_new_2[a])
                else:
                    if actual_index < precision_half:
                        for b in range(test_precision):
                            precision_list.append(flickr_new_2[b])
                    elif actual_index > upper_bound:
                        for c in range(size - test_precision, size):
                            precision_list.append(flickr_new_2[c])
            elif precision_half == 0:
                precision_list.append(f)
            if prediction in precision_list:
                correct_prediction_count += 1

    print("Correctly aligns ", correct_prediction_count, " pairs up to precision of ", test_precision, ".")
