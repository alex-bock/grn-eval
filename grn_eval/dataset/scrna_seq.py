
from __future__ import annotations
from typing import Any, Dict

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
        dataset.adata.var_names_make_unique()
        dataset.df = dataset.adata.to_df()

        if transpose:
            dataset.df = dataset.df.T

        return dataset

    @classmethod
    def from_h5ad(cls, h5_path: str) -> scRNASeq:

        dataset = cls()
        dataset.adata = sc.read_h5ad(h5_path)
        dataset.adata.var_names_make_unique()
        dataset.df = dataset.adata.to_df()

        return dataset
