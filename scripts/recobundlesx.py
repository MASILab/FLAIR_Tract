from plumbum import local
import yaml
import sys
from pathlib import Path
from typing import TypedDict


class ConfigType(TypedDict):
    derivatives_dir: str | Path  # tractoflow_folder
    input_trk_path: str | Path
    subj_T1: str | Path
    model_T1: str | Path
    models_config: str | Path
    atlases: str | Path
    minL: int
    maxL: int
    nb_total_executions: int  # len(model_clustering_thr) * len(bundle_pruning_thr) * len(tractogram_clustering_thr) = max total executions (see json).
    thresh_dist: str  # "10 12" Whole brain clustering threshold (in mm) for QuickBundles.
    seed: int  # Random number generator initialisation.
    minimal_vote: float  # Saving streamlines if recognized often enough.
    itk_threads: int
    ants_sif_path: str | Path
    scil_procs: int


# get config file
conf_path = sys.argv[1]
with open(conf_path, "r") as conf:
    config: ConfigType = yaml.safe_load(conf)

input_trk_path = Path(config["input_trk_path"])

if input_trk_path.is_file() and input_trk_path.exists():
    input_trk_list = input_trk_path.read_text().split()

else:
    fdfind = local["fdfind"]
    input_trk_list = fdfind("-e 'trk' . ", config["derivatives_dir"]).split()


subj_T1_dir = str(Path(config["subj_T1"]).parent)
model_dir = str(Path(config["model_T1"]).parent)
subj_data_dir = str(config["derivatives_dir"])

ants_bind = ",".join([subj_T1_dir, model_dir, subj_data_dir])

ants_sif_path = str(config["ants_sif_path"])

itk_threads = int(config["itk_threads"])
apptainer_cmd = local["apptainer"][
    "exec",
    "-c",
    "-e",
    "--bind",
    ants_bind,
    "--env",
    f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={itk_threads}",
    ants_sif_path,
]

antsRegistrationSyNQuick = apptainer_cmd["antsRegistrationSyNQuick.sh"]


minL = int(config["minL"])
maxL = int(config["maxL"])


scil_filter_streamlines_by_length = local["scil_tractogram_filter_by_length"]

atlases = str(config["atlases"])

scil_procs = int(config["scil_procs"])

min_vote = int(config["minimal_vote"])
scil_recognize_multi_bundles = local["scil_tractogram_segment_with_bundleseg"]


for subj_trk in input_trk_list:
    output_path = Path(subj_trk).parent
    rbx_folder = output_path.joinpath("rbx_folder")
    rbx_folder.mkdir(exist_ok=True)
    model_to_subj_out = rbx_folder.joinpath("model_to_subj_anat")
    model_to_subj = str(model_to_subj_out) + "0GenericAffine.mat"

    antsRegistrationSyNQuick(
        "-d",
        "3",
        "-f",
        config["subj_T1"],
        "-m",
        config["model_T1"],
        "-t",
        "a",
        "-n",
        4,
        "-o",
        str(model_to_subj_out),
    )

    subj_filtered_tracking = output_path.joinpath("Tracking")
    subj_filtered_tracking.mkdir(exist_ok=True)
    subj_filtered_trk = str(
        subj_filtered_tracking.joinpath("filtered_length.trk")
    )
    scil_filter_streamlines_by_length(
        "--minL", minL, "--maxL", maxL, subj_trk, subj_filtered_trk
    )

    multi_bundles = rbx_folder.joinpath("multi_bundles")
    multi_bundles.mkdir(exist_ok=True)
    models_config = str(config["models_config"])
    atlases = str(config["atlases"]) + "/*"

    scil_recognize_multi_bundles(
        subj_filtered_trk,
        models_config,
        atlases,
        model_to_subj,
        "--out_dir",
        str(multi_bundles),
        "--processes",
        scil_procs,
        "--seed",
        int(config["seed"]),
        "--minimal_vote_ratio",
        min_vote,
        "-v",
        "DEBUG",
        "--inverse",
        "--exploration_mode",
        "-f",
    )
