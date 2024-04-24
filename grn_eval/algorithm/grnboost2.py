
from distributed import LocalCluster, Client

from arboreto.algo import grnboost2
import pandas as pd

from .algorithm import Algorithm
from ..dataset.scrna_seq import scRNASeq


class GRNBoost2(Algorithm):

    def __init__(self):

        super().__init__()

        return

    def run(self, dataset: scRNASeq) -> pd.DataFrame:

        cluster = LocalCluster()
        client = cluster.get_client()

        return grnboost2(dataset.df, verbose=True, client_or_address=client)
