
from __future__ import annotations
from typing import Any, Dict

import pandas as pd
import scanpy as sc

from .dataset import Dataset


class scRNASeq(Dataset):

    def __init__(self):

        super().__init__()

        self.adata = None
        self.df = None

        return

    @classmethod
    def from_csv(
        cls, csv_path: str, delimiter: str = ",", transpose: bool = True,
        pd_kwargs: Dict[str, Any] = None
    ) -> scRNASeq:

        if pd_kwargs is None:
            pd_kwargs = {}

        dataset = cls()
        dataset.adata = sc.read_csv(csv_path, delimiter=delimiter)
        dataset.df = pd.read_csv(csv_path, sep=delimiter, **pd_kwargs)

        if transpose:
            dataset.df = dataset.df.T

        return dataset

    @classmethod
    def from_h5(cls, h5_path: str) -> scRNASeq:

        dataset = cls()
        dataset.adata = sc.read_10x_h5(h5_path)
        dataset.df = dataset.adata.to_df()

        return dataset
