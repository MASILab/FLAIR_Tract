import pandas as pd
import pathlib as pl
import sys

def get_prequal_path(p):
    arr = p.split("BIDS")
    subarr = arr[1].split("/")
    parent = pl.Path(arr[0]).joinpath("BIDS", subarr[0],subarr[1],"derivatives",subarr[2], subarr[3])
    prequal_dirs = list(filter(lambda pp: pp.is_dir(), parent.glob("PreQual*")))
    if len(prequal_dirs) == 1:
        return prequal_dirs[0]
    res = list(filter(lambda pp: pl.re.match(r"(PreQual$|PreQualrun-2|PreQualacq-double)", pp.name), parent.glob("PreQual*")))
    return str(res[0]) if len(res) else "None"



if __name__ == "_main_":
    datalist = sys.argv[1]
    df = pd.read_csv(datalist)
    df['prequal_path'] = get_prequal_path(df['path'])

    # Need to manually change subjects with None in prequal column. (col 16)
    df.to_csv("../resources/dataset/df_with_prequal.csv", index=False)
