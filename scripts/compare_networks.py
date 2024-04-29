
import os
import sys
sys.path.append(os.getcwd())

from typing import Tuple

import networkx as nx

from grn_eval.network import Network


PARAMS = {
    "grnboost2": {
        "u_col": "TF",
        "v_col": "target",
        "edge_weight_col": "importance",
        "index_col": 0
    },
    "scenic": {
        "u_col": "TF",
        "v_col": "target",
        "edge_weight_col": None,
        "index_col": 0
    },
    "ground_truth": {
        "u_col": "TF",
        "v_col": "target",
        "edge_weight_col": None
    }
}


def get_common_nodes(
    network_1: Network, network_2: Network
) -> Tuple[Network, Network]:

    shared_nodes = set.intersection(
        set(network_1.graph.nodes), set(network_2.graph.nodes)
    )
    subg_1 = network_1.graph.subgraph(shared_nodes)
    print(
        f"network 1: {len(network_1.graph.edges)} -> {len(subg_1.edges)} edges"
    )
    subg_2 = network_2.graph.subgraph(shared_nodes)
    print(f"network 2: {len(network_2.graph.edges)} -> {len(subg_2.edges)} edges")

    return shared_nodes


def jaccard_distance(network_1: Network, network_2: Network) -> float:

    shared_nodes = get_common_nodes(network_1, network_2)
    shared_edges = set.intersection(
        set(network_1.graph.edges), set(network_2.graph.edges)
    )
    all_edges = set.union(
        set(network_1.graph.edges), set(network_2.graph.edges)
    )
    all_edges = set(
        [edge for edge in all_edges
         if edge[0] in shared_nodes and edge[1] in shared_nodes]
    )
    jaccard = len(shared_edges) / len(all_edges)

    import pdb; pdb.set_trace()

    return 1 - jaccard


if __name__ == "__main__":

    fp_1 = sys.argv[1]
    method_1 = sys.argv[2][2:]

    if method_1 in ("grnboost2", "scenic", "ground_truth"):
        network_1 = Network.from_edgelist(fp_1, **PARAMS[method_1])
    elif method_1 == "scode":
        network_1 = Network.from_matrix(fp_1)

    fp_2 = sys.argv[3]
    method_2 = sys.argv[4][2:]

    if method_2 in ("grnboost2", "scenic", "ground_truth"):
        network_2 = Network.from_edgelist(fp_2, **PARAMS[method_2])
    elif method_2 == "scode":
        network_2 = Network.from_matrix(fp_2)

    import pdb; pdb.set_trace()

    print(jaccard_distance(network_1, network_2))
