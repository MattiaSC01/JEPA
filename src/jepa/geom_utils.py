import time
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from GraphRicciCurvature.OllivierRicci import OllivierRicci

# Compute the k-nearest neighbors graph of the provided embeddings. This allows us to discretize the manifold.
def build_knn_graph(embeddings, k=5):
    dists = cdist(embeddings, embeddings)
    sorted_dists = dists.argsort(axis=-1)
    knn = sorted_dists[:, 1:k + 1]
    G = nx.Graph() # create the graph
    for i in range(embeddings.shape[0]): # add nodes
        for j in knn[i]:
            G.add_edge(i, j) # add edges

    return G

# Computes Gromov's delta hyperbolicity of a graph
def delta_hyperbolicity(G, num_samples=5000):
    curr_time = time.time()
    hyps = []
    for i in range(num_samples):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    # print('Time for delta-hyp: ', time.time() - curr_time)
    return max(hyps)

# Computes Ricci curvature of the graph edges and return the values
def oricci_curvature(G):
    curr_time = time.time()
    orc = OllivierRicci(G, alpha=0.5, method='Sinkhorn', verbose="TRACE")
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()
    ricci_curvatures = nx.get_edge_attributes(G_orc, "ricciCurvature").values()
    print('Time for OR curvature calculation: ', time.time() - curr_time)
    del G_orc 
    return list(ricci_curvatures)

# TODO: compute intrinsic dimension of the embeddings
def intrinsic_dimension(embeddings, G):
    pass


