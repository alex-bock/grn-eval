
import os
import sys
sys.path.append(os.getcwd())

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


if __name__ == "__main__":

    fp = sys.argv[1]
    method = sys.argv[2][2:]

    if method in ("grnboost2", "scenic", "ground_truth"):
        network = Network.from_edgelist(fp, **PARAMS[method])
    elif method == "scode":
        network = Network.from_matrix(fp)

    basename = ".".join(fp.split(".")[:-1])

    edgelist_fp = f"{basename}.edgelist"
    if not os.path.exists(edgelist_fp):
        network.to_edgelist(edgelist_fp, as_i=True)

    network.degree_distribution(fig_path=f"{basename}_degrees.png")
    network.run_hits()

    network.load_geneset_assignments(fp="./data/genesets.csv")

    node2vec_fp = f"{basename}_node2vec.emb"
    if not os.path.exists(node2vec_fp):
        network.generate_node2vec_embeddings(emb_fp=node2vec_fp)
    network.load_embeddings(type="node2vec", emb_fp=node2vec_fp)

    network.node_df.to_csv(f"{basename}_nodes.csv")
    network.geneset_df.to_csv(f"{basename}_genesets.csv")
