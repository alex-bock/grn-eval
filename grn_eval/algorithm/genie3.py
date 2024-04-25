
from distributed import LocalCluster

from arboreto.algo import genie3
import pandas as pd

from .algorithm import Algorithm
from ..dataset.scrna_seq import scRNASeq


class GENIE3(Algorithm):

    def __init__(self):

        super().__init__()

        return

    def run(self, dataset: scRNASeq) -> pd.DataFrame:

        cluster = LocalCluster()
        client = cluster.get_client()

        return genie3(dataset.df, verbose=True, client_or_address=client)
