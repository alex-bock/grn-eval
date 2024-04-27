
from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


class Network:

    def __init__(self):

        self.embeddings = dict()

        return

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, u_col: str, v_col: str, edge_weight_col: str,
        edge_weight_threshold: float = None
    ) -> Network:

        network = cls()
        if edge_weight_threshold is not None:
            df = df[df[edge_weight_col] > edge_weight_threshold]
        df.rename(
            {u_col: "u", v_col: "v", edge_weight_col: "weight"}, axis=1,
            inplace=True
        )

        network.edge_df = df
        graph = nx.from_pandas_edgelist(
            df, source="u", target="v", edge_attr="weight"
        )
        network.graph = nx.convert_node_labels_to_integers(
            graph, label_attribute="symbol"
        )
        network.node_df = pd.DataFrame.from_dict(
            dict(network.graph.nodes(data=True)), orient="index"
        )

        return network

    def to_edgelist(self, fp: str, data: bool = False):

        nx.write_edgelist(self.graph, fp, data=data)

        return

    def degree_distribution(self):

        degrees = nx.degree(self.graph)
        degrees = [degrees[i] for i in range(len(degrees))]
        fig = px.histogram(x=degrees)
        fig.show()

        return

    def load_geneset_assignments(self, fp: str):

        geneset_df = pd.read_csv(fp)
        self.node_df["genesets"] = self.node_df.symbol.apply(
            lambda x: geneset_df[geneset_df.gene_symbol == x].gs_name.unique().tolist()
        )

        genesets = set()
        for geneset_list in self.node_df.genesets:
            genesets.update(geneset_list)
        self.geneset_df = pd.DataFrame(index=list(genesets))

        geneset_counts = defaultdict(int)
        for geneset_list in self.node_df.genesets:
            for geneset in geneset_list:
                geneset_counts[geneset] += 1
        self.geneset_df["count"] = geneset_counts

        geneset_weights = defaultdict(float)
        for geneset in self.geneset_df.index:
            geneset_nodes = self.node_df[self.node_df.genesets.apply(lambda x: geneset in x)].index.values
            if len(geneset_nodes) == 0:
                continue
            subgraph = self.graph.subgraph(geneset_nodes)
            subgraph_edges = nx.get_edge_attributes(subgraph, "weight").items()
            geneset_weights[geneset] = sum([item[1] for item in subgraph_edges])
            if geneset == "WP_G_PROTEIN_SIGNALING_PATHWAYS":
                import pdb; pdb.set_trace()
            if len(subgraph_edges) > 0:
                 geneset_weights[geneset] /= len(subgraph_edges)

        self.geneset_df["weight"] = geneset_weights

        return

    def load_node2vec_embeddings(self):

        self.embeddings["node2vec"] = np.zeros((len(self.graph), 128))

        node2vec = Node2Vec(self.graph)
        embeddings = node2vec.fit()

        for i in self.graph.nodes:
            self.embeddings["node2vec"][i] = embeddings.wv[i]

        return

    def load_embeddings(self, fp: str, type: str):

        with open(fp, "r") as f:
            lines = f.readlines()

        n_nodes, n_dim = int(lines[0].strip().split()[0]), int(lines[0].strip().split()[1])
        assert(n_nodes == len(self.graph.nodes))
        self.embeddings[type] = np.zeros((n_nodes, n_dim))

        for line in lines[1:]:
            vals = line.strip().split()
            self.embeddings[type][int(vals[0])] = [float(val) for val in vals[1:]]

        return

    def plot_embeddings(self, type: str, geneset: str = None):

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.embeddings[type])

        if geneset is None:
            fig = px.scatter(x=reduced[:, 0], y=reduced[:, 1])
            fig.show()
        else:
            fig = px.scatter(x=reduced[:, 0], y=reduced[:, 1], color=self.node_df.genesets.apply(lambda x: geneset in x))
            fig.show()

    def draw(self, geneset: str = None):

        if geneset is None:
            graph = self.graph
        else:
            graph = self.graph.subgraph(
                self.node_df[self.node_df.genesets.apply(lambda x: geneset in x)].index.values
            )

        edges, weights = zip(*nx.get_edge_attributes(graph, "weight").items())

        nx.draw(graph, pos=nx.drawing.layout.circular_layout(graph), edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues)
        plt.show()
