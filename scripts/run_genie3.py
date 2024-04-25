
import os
import sys
sys.path.append(os.getcwd())

from grn_eval.dataset import scRNASeq
from grn_eval.algorithm import GENIE3


if __name__ == "__main__":

    data_fp = sys.argv[1]
    outpath = sys.argv[2]

    if data_fp.endswith(".csv"):
        dataset = scRNASeq.from_csv(
            data_fp, transpose=True, pd_kwargs={"index_col": 0}
        )
    elif data_fp.endswith(".h5ad"):
        dataset = scRNASeq.from_h5ad(data_fp)

    algorithm = GENIE3()
    result = algorithm.run(dataset)
    result.to_csv(outpath)
