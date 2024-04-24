
from __future__ import annotations
from typing import Any, Dict

from anndata import AnnData
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
        dataset.adata.var_names_make_unique()
        dataset.adata = dataset._filter_adata(dataset.adata)
        dataset.df = dataset.adata.to_df()

        if transpose:
            dataset.df = dataset.df.T

        return dataset

    @classmethod
    def from_h5(cls, h5_path: str) -> scRNASeq:

        dataset = cls()
        dataset.adata = sc.read_10x_h5(h5_path)
        dataset.adata.var_names_make_unique()
        dataset.adata = dataset._filter_adata(dataset.adata)
        dataset.df = dataset.adata.to_df()

        return dataset
    
    def _filter_adata(self, adata: AnnData) -> AnnData:

        # adata.var["mt"] = adata.var_names.str.startswith("MT-")
        # sc.pp.calculate_qc_metrics(
        #     adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        # )

        # n_counts_filter = 2500
        # mito_filter = 5

        # adata = adata[adata.obs.n_genes_by_counts < n_counts_filter, :]
        # adata = adata[adata.obs.pct_counts_mt < mito_filter, :].copy()

        sc.pp.filter_genes(adata, min_cells=3)

        return adata
