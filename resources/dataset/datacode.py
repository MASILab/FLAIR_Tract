from plumbum import local
import pathlib as pl
import pandas as pd

rg = local["rg"]
prequal_list = "/nfs2/harmonization/BIDS/ALL_PREQUALS_11_20_25.txt"
pattern = "(NACC|BIOCARD|OASIS4|WRAP|BLSA|WASHU|HABSHD)"
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
flair_list = fdfind(
    "FLAIR.nii.gz",
    "--base-directory=/nfs2/harmonization/BIDS",
    "--search-path=BIOCARD",
    "--search-path=WASHU",
    "--search-path=HABSHD",
    "--search-path=WRAP",
    "--search-path=NACC",
    "--search-path=OASIS4",
    "--search-path=BLSA",
)
flair_list = flair_list.split()
flair_ser = pd.Series(flair_list, name="flair")
flair_df = flair_ser.to_frame()
flair_df["subject"] = flair_df.flair.map(lambda s: s.split("/")[1])
flair_df["session"] = flair_df.flair.map(lambda s: s.split("/")[2])
df_flair_subsess = df_sub_gr1.merge(flair_df, on=["subject", "session"], how='left')  # left because dont want to drop if no flair->session

# Drop oasis because only 11 unique subs with >1 prequal

df_flair_subsess[df_flair_subsess.dataset != 'OASIS4'].groupby('dataset').subject.sample(25)
subject_list = df_flair_subsess[df_flair_subsess.dataset != 'OASIS4'].groupby('dataset').subject.sample(25)
df_flair_subsess[df_flair_subsess.subject.isin(subject_list)]
df_flair_subsess[df_flair_subsess.subject.isin(subject_list)].to_csv("UPDATED_DATASET_12-29-2025.csv", index=False)

