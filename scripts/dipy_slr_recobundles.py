from plumbum import local
from pathlib import Path
import fire

dipy_slr = local["dipy_slr"]
dipy_recobundles = local["dipy_recobundles"]
dipy_convert_tractogram = local["dipy_convert_tractogram"]

# dipy_recobundles "moved.trk" "/fs5/p_masi/schwat1/spie_flair_tract_extension/resources/atlases/bundleseg/atlases/atlas/pop_average/*.trk" --force --mix_names --out_dir "rb_output"
# dipy_slr Atlas_30_Bundles/whole_brain/whole_brain_MNI.trk ../T1_test50to250.trk --force


def run_pipeline(
    mni_fixed_tractogram, moving_tractogram, tract_atlases,modality, num_threads=-5
):

    slr_out_dir = Path(moving_tractogram).parent.joinpath(f"{modality}_slr_out_dir")
    if not slr_out_dir.exists():
        slr_out_dir.mkdir()
    dipy_slr_arguments = [
        mni_fixed_tractogram,
        moving_tractogram,
        "--force",
        "--qbx_thr",
        "50","35","25","15",
        "--num_threads",
        "-5",
        "--out_dir",
        str(slr_out_dir),
    ]
    # dipy_slr(dipy_slr_arguments)

    moved_trk = str([*slr_out_dir.rglob("moved.tr*")][0])

    recobundles_out_dir = Path(moving_tractogram).parent.joinpath(f"{modality}_rb_out")
    if not recobundles_out_dir.exists():
        recobundles_out_dir.mkdir()
    dipy_recobundles_arguments = [
        moved_trk,
        tract_atlases + "/*.trk",
        "--force",
        "--mix_names",
        "--out_dir",
        str(recobundles_out_dir),
    ]

    # dipy_recobundles(dipy_recobundles_arguments)

    trks = filter(lambda x: x.stat().st_size > 1000, recobundles_out_dir.glob("*.trk"))
    

    for trk_file in trks:
        dipy_convert_tractogram(["--reference", moved_trk, "--out_tractogram", str(trk_file).replace(".trk", ".tck"), str(trk_file)])
        # trk = load_tractogram(str(trk_file), str(moved_trk))
        # _ = save_tractogram(trk, recobundles_out_dir.joinpath(trk_file.name.replace(".trk", ".tck")))


if __name__ == "__main__":
    fire.Fire(run_pipeline)
