
from __future__ import annotations

from collections import defaultdict
import json

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
    def from_edgelist(
        cls, fp: str, u_col: str, v_col: str, edge_weight_col: str,
        edge_weight_threshold: float = None, sep: str = ",",
        index_col: int = None
    ) -> Network:

        df = pd.read_csv(fp, sep=sep, index_col=index_col)
        if edge_weight_threshold is not None:
            df = df[df[edge_weight_col] > edge_weight_threshold]

        if edge_weight_col is not None:
            df.rename(columns={edge_weight_col: "weight"}, inplace=True)
            edge_weight_col = "weight"
        else:
            df["weight"] = 1.0

        network = cls()
        network.graph = nx.from_pandas_edgelist(
            df, source=u_col, target=v_col, edge_attr="weight",
            create_using=nx.DiGraph()
        )
        network.node_df = pd.DataFrame(index=network.graph.nodes)
        network.node_df["i"] = range(len(network.node_df))

        return network

    @classmethod
    def from_matrix(cls, fp: str) -> Network:

        df = pd.read_csv(fp, index_col=0)

        network = cls()
        network.graph = nx.from_numpy_matrix(
            df.values, create_using=nx.DiGraph()
        )
        network.node_df = pd.DataFrame.from_dict(
            {name: i for i, name in enumerate(df.columns)}, orient="index",
            columns=["symbol"]
        )

        return network

    def to_edgelist(self, fp: str, data: bool = False, as_i: bool = False):

        if as_i:
            graph = nx.relabel_nodes(self.graph, self.node_df.i, copy=True)
        else:
            graph = self.graph

        nx.write_edgelist(graph, fp, data=data)

        return

    def degree_distribution(self, fig_path: str = None, weighted: bool = False):

        degrees = dict(
            nx.degree(self.graph, weight="weight" if weighted else None)
        )
        self.node_df["degree"] = degrees
        fig = px.histogram(x=self.node_df.degree.values)

        if fig_path is not None:
            fig.write_image(fig_path)
        else:
            fig.show()

        return

    def run_hits(self):

        hubs, hits = nx.hits(self.graph)
        self.node_df["hub"] = hubs
        self.node_df["hit"] = hits

        return

    def load_geneset_assignments(self, fp: str):

        geneset_df = pd.read_csv(fp)
        self.node_df["genesets"] = self.node_df.index.map(
            lambda x: geneset_df[
                geneset_df.gene_symbol == x
            ].gs_name.unique().tolist()
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
            geneset_nodes = self.node_df[
                self.node_df.genesets.apply(lambda x: geneset in x)
            ].index.values
            if len(geneset_nodes) == 0:
                continue
            subgraph = self.graph.subgraph(geneset_nodes)
            subgraph_edges = nx.get_edge_attributes(subgraph, "weight").items()
            geneset_weights[geneset] = sum([item[1] for item in subgraph_edges])
            if len(subgraph_edges) > 0:
                 geneset_weights[geneset] /= len(subgraph_edges)

        self.geneset_df["weight"] = geneset_weights

        return

    def generate_node2vec_embeddings(self, emb_fp: str, weighted: bool = False):

        model = Node2Vec(self.graph, weight_key="weight" if weighted else None)
        result = model.fit()

        with open(emb_fp, "w") as f:
            for node in self.node_df.index.values:
                f.write(
                    " ".join(
                        [node] + list([str(x) for x in result.wv[node]])
                    ) + "\n"
                )

        return

    def load_embeddings(
        self, type: str, emb_fp: str, i_indexed: bool = False,
        header: bool = False
    ):

        with open(emb_fp, "r") as f:
            lines = f.readlines()

        if header:
            lines = lines[1:]

        dim = len(lines[0].split()) - 1
        self.embeddings[type] = np.zeros((len(self.node_df), dim))

        for line in lines:
            vals = line.split()
            i = vals[0]
            embedding = vals[1:]
            if not i_indexed:
                i = self.node_df.loc[i].i
            self.embeddings[type][i] = embedding

        return

    def draw(self, geneset: str = None):

        if geneset is None:
            graph = self.graph
        else:
            graph = self.graph.subgraph(
                self.node_df[
                    self.node_df.genesets.apply(lambda x: geneset in x)
                ].index.values
            )

        edges, weights = zip(*nx.get_edge_attributes(graph, "weight").items())

        nx.draw(
            graph, pos=nx.drawing.layout.circular_layout(graph), edgelist=edges,
            edge_color=weights, edge_cmap=plt.cm.Blues
        )
        plt.show()
