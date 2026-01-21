from plumbum import local
import pathlib as pl
import pandas as pd


def hierarch_pq(grp):
    if grp.path.str.contains("double").any():
        ret = grp[grp.path.str.contains("double")]
    elif grp.path.str.contains("PreQualrun-2").any():
        ret = grp[grp.path.str.contains("PreQualrun-2")]
    elif grp.path.str.contains("HARDI").any():
        ret = grp[grp.path.str.contains("HARDI")]
    else:
        ret = grp[grp.path.str.endswith("PreQual")]
    return ret


def order_group_by_session(grp):
    g = grp.sort_values("session")
    return g.iloc[-1]


def propagate_anat_across_rows(grp):
    g = grp.sort_values("session")
    g.iat[0, -1] = g.iat[1, -1]
    g.iat[0, -2] = g.iat[1, -2]
    return g


def get_ordered_subject_data(grp: pd.DataFrame):
    grp = grp.sort_values("session")
    g_end = grp.iloc[1:, :]
    g_end = g_end[~(g_end.flair.isna()) & ~(g_end.mprage.isna())]
    if len(g_end):
        scan2 = g_end.sample(1)
        scan1_set = grp.iloc[: scan2.index[0], :]
        scan1 = scan1_set[scan1_set.session != scan2.session.values[0]].sample(
            1
        )
        if len(scan1):
            return (scan1.index[0], scan2.index[0])
    return (-1, -1)


def prequal_is_valid(p: str):
    if not pl.Path(p).joinpath("PREPROCESSED").exists():
        return False
    return True


rg = local["rg"]
prequal_list = "/nfs2/harmonization/BIDS/ALL_PREQUALS_11_20_25.txt"
pattern = r"(\bNACC\b|BIOCARD|WRAP\b|BLSA|WASHU|HABSHD)"
prequal_dirs = list(
    filter(
        lambda s: pl.Path(s).is_dir() & pl.Path(s).exists(),
        sorted(
            rg("--regexp", pattern, prequal_list).split(),
            key=lambda s: s.split("/")[4],
        ),
    )
)
ser = pd.Series(sorted(prequal_dirs, key=lambda s: "".join(s.split("/")[6:7])))
df = ser.to_frame(name="path")
df.loc[:, ["valid_prequal"]] = df.path.apply(prequal_is_valid)
df = df[df.valid_prequal]
df["dataset"] = df.path.map(lambda s: s.split("/")[4])
df["subject"] = df.path.map(lambda s: s.split("/")[6])
df["session"] = df.path.map(lambda s: s.split("/")[7])
df["sub_sess_count"] = df.groupby("subject")["session"].count()

df.drop(columns="sub_sess_count", inplace=True)
sub_sess_count = df.groupby("subject")["session"].nunique()
sub_sess_count.name = "sub_sess_count"
df = df.merge(sub_sess_count, on="subject")
df_sub_gr1 = df[df.sub_sess_count > 1]
fdfind = local["fdfind"]
if not pl.Path("FLAIR_LIST_01-12-2026.csv").exists():
    flair_list = fdfind(
        "FLAIR.nii.gz",
        "--base-directory=/nfs2/harmonization/BIDS",
        "--search-path=BIOCARD",
        "--search-path=WASHU",
        "--search-path=HABSHD",
        "--search-path=WRAP",
        "--search-path=NACC",
        "--search-path=BLSA",
        "--exclude=WRAPnew$",
        "--exclude=SCAN$",
        "--exclude=dwi",
        "--exclude=func",
        "--exclude=perf",
        "--absolute-path",
    )
    flair_list = flair_list.split()
    flair_ser = pd.Series(flair_list, name="flair")
    flair_df = flair_ser.to_frame()
    flair_df.to_csv("FLAIR_LIST_01-12-2026.csv")
else:
    flair_df = pd.read_csv("FLAIR_LIST_01-12-2026.csv")
flair_df["subject"] = flair_df.flair.map(lambda s: s.split("/")[5])
flair_df["session"] = flair_df.flair.map(lambda s: s.split("/")[6])

mprage = []
for row in df.itertuples():
    p = list(pl.Path(row.path).parent.parts)
    _ = p.pop(5)
    pth = pl.Path("/").joinpath(*p[0:])
    res: list[str] = fdfind(
        r"(MPRAGE|T1).*nii.gz", f"--base-directory={pth}", "--absolute-path"
    ).split()
    mprage.extend(res)

t1_df = pd.Series(mprage, name="mprage").to_frame()
t1_df["subject"] = t1_df.mprage.map(lambda s: s.split("/")[5])
t1_df["session"] = t1_df.mprage.map(lambda s: s.split("/")[6])

df_flair_subsess = df_sub_gr1.merge(
    flair_df, on=["subject", "session"], how="left"
)
df_flair_t1 = df_flair_subsess.merge(
    t1_df, on=["subject", "session"], how="left"
)

paired_data = df_flair_t1.groupby("subject").apply(
    get_ordered_subject_data, include_groups=False
)
paired_data = paired_data.to_frame(name="pair").reset_index()

subject_indexes = []
for pair in paired_data[paired_data.pair != -1].itertuples():
    subject_indexes.extend(list(pair.pair))

subject_series = (
    df_flair_t1[df_flair_t1.index.isin(subject_indexes)]
    .loc[:]
    .drop_duplicates(subset="subject")
    .groupby(by="dataset")["subject"]
    .sample(25)
)
selected_indexes = df_flair_t1[df_flair_t1.index.isin(subject_indexes)]
selected_data = selected_indexes[
    selected_indexes.subject.isin(subject_series.values)
]

selected_data = (
    selected_data.groupby(by="subject")
    .apply(propagate_anat_across_rows, include_groups=False)
    .reset_index()
)
selected_data = selected_data[
    [
        "dataset",
        "subject",
        "session",
        "path",
        "flair",
        "mprage",
        "valid_prequal",
    ]
]
selected_data.to_csv("CURRENT_DATASET_01-20-2026.csv", index=False)
