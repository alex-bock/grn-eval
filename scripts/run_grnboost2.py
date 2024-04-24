
import os
import sys
sys.path.append(os.getcwd())

from grn_eval.dataset import scRNASeq
from grn_eval.algorithm import GRNBoost2
from grn_eval.network import Network


if __name__ == "__main__":

    data_fp = sys.argv[1]

    if data_fp.endswith(".csv"):
        dataset = scRNASeq.from_csv(
            data_fp, transpose=True, pd_kwargs={"index_col": 0}
        )
    elif data_fp.endswith(".h5"):
        dataset = scRNASeq.from_h5(data_fp)

    algorithm = GRNBoost2()
    result = algorithm.run(dataset)
    network = Network.from_df(result)

    print(network.df)
    network.df.to_csv("./result.csv")
