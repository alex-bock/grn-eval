
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
        "sep": "\t"
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

    edgelist_fp = fp.replace(".csv", ".edgelist")
    if not os.path.exists(edgelist_fp):
        network.to_edgelist(edgelist_fp)

    print(len(network.graph.edges))
    network.degree_distribution()
    network.run_hits()
    import pdb; pdb.set_trace()

    network.load_geneset_assignments("./data/genesets.csv")

    network.load_node2vec_embeddings()
    network.plot_embeddings(
        type="node2vec", geneset="WP_CONSTITUTIVE_ANDROSTANE_RECEPTOR_PATHWAY"
    )

    # network.plot_embeddings(
    #     geneset="WP_G_PROTEIN_SIGNALING_PATHWAYS", type="struc2vec"
    # )
    # network.draw(geneset="WP_G_PROTEIN_SIGNALING_PATHWAYS")
