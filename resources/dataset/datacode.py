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
dedup_flair_df = flair_df[["subject", "flair"]].drop_duplicates(
    subset="subject"
)

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
t1_df = t1_df.drop_duplicates(subset="subject")

df_flair_subsess = df_sub_gr1.merge(dedup_flair_df, on=["subject"], how="left")
df_flair_t1 = df_flair_subsess.merge(t1_df, on=["subject"], how="left")

subject_list = (
    df_flair_t1[
        ~(df_flair_t1.dataset.isin(("OASIS4", "WRAPnew", "SCAN")))
        & (df_flair_t1.flair.notna())
    ]
    .groupby("dataset")
    .subject.sample(25)
)
df_flair_t1[df_flair_t1.subject.isin(subject_list)].to_csv(
    "UPDATED_DATASET_01-12-2026.csv", index=False
)
df_flair_t1.groupby(by=["subject", "session"]).apply(
    hierarch_pq, include_groups=False
).reset_index().sort_values(by=["dataset", "subject", "session"]).drop(
    columns="level_2"
k.groupby(by="subject").apply(lambda grp: grp.iloc[:2, :], include_groups=False)
