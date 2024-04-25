
from __future__ import annotations

import networkx as nx
import pandas as pd


class Network:

    def __init__(self):

        return

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, u_col: str, v_col: str, edge_weight_col: str,
        edge_weight_threshold: float = None
    ) -> Network:

        network = cls()
        if edge_weight_threshold is not None:
            df = df[df[edge_weight_col] > edge_weight_threshold]
        network.df = df
        network.graph = nx.from_pandas_edgelist(
            df, source=u_col, target=v_col, edge_attr=edge_weight_col
        )

        return network

    def visualize(self):

        return
