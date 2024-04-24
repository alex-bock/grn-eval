
from __future__ import annotations

import pandas as pd


class Network:

    def __init__(self):

        return

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Network:

        network = cls()
        network.df = df

        return network

    def visualize(self):

        return
