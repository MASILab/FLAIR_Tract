from pathlib import Path
import nibabel as nib
import numpy as np
from plumbum import local
import yaml
from collections.abc import Sequence
from typing import Any, TypedDict


class ConfigType(TypedDict):
    derivatives_dir: str | Path
    t1_path: str | Path
    atlas_path: str | Path
    proj_path: str | Path


def register_flair2T1(
    total_paths: Sequence[str | Path],
    config_path: str = "process_flair_config.yaml",
    ants_sif_path: str | Path = "/fs5/p_masi/schwat1/spie_flair_tract_extension/resources/sifs/Ants.sif",
) -> None:
    """Register FLAIR image to T1 image using ANTs via Apptainer."""
    # Read configuration from config
    with open(config_path, "r") as f:
        config: ConfigType = yaml.safe_load(f)
    derivatives_dir = Path(config["derivatives_dir"])
    atlas_path_base = Path(config["atlas_path"])

    for this_path in total_paths:
        this_path = Path(this_path)
        print(this_path)

        # Extract subject/session information from the BIDS path
        # Assuming the path format is like: /some/path/sub-<id>/ses-<session>...
        parts = this_path.parts
        subject_id: str | None = None
        session_id: str | None = None

        for part in parts:
            if part.startswith("sub-"):
                subject_id = part
            elif part.startswith("ses-"):
                session_id = part
                break

        if not subject_id or not session_id:
            print(f"Could not extract subject/session from path: {this_path}")
            continue


        # Construct T1 path in derivatives structure
        t1_path = Path(config["t1_path"])

        # Find FLAIR image in the BIDS path
        flair_paths = list(this_path.rglob("*FLAIR.nii.gz"))

        if len(flair_paths) != 1:
            print("got errors when finding flair image")
            print([str(p) for p in flair_paths])
            continue

        flair_path = flair_paths[0]

        # Define output path in derivatives folder
        output_path = derivatives_dir / subject_id / session_id
        output_path.mkdir(parents=True, exist_ok=True)
        output_name = output_path / "flair_registered2_T1_N4_mni_1mm"
        print(output_name)

        # Prepare bind mounts for Apptainer
        project_path = config["proj_path"]
        bind_mounts = [
            str(t1_path.parent),  # T1 directory
            project_path,
            str(output_path),  # Output directory
        ]

        # Create bind string
        bind_string = ",".join(bind_mounts)

        # Run ANTs registration via Apptainer
        apptainer_cmd = local["apptainer"][
            "exec",
            "-c",
            "-e",
            "--bind",
            bind_string,
            "--env",
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1",
            str(ants_sif_path),
        ]
        ants_registration = apptainer_cmd["antsRegistrationSyN.sh"]
        ants_registration(
            "-d",
            "3",
            "-f",
            str(t1_path),
            "-m",
            str(flair_path),
            "-t",
            "r",
            "-o",
            str(output_name),
        )


def register_atlas_seg_2_flair(
    total_paths: Sequence[str | Path],
    config_path: str = "process_flair_config.yaml",
    fs5_flair_copy_path: str | Path | None = None,
    compute_fusion_only: bool = False,
    ants_sif_path: str | Path = "/fs5/p_masi/schwat1/spie_flair_tract_extension/resources/sifs/Ants.sif",
) -> None:
    """Register atlas segmentations to FLAIR space and perform fusion via Apptainer."""
    # Read configuration from config
    with open(config_path, "r") as f:
        config: ConfigType = yaml.safe_load(f)
    derivatives_dir = Path(config["derivatives_dir"])
    atlas_path_base = Path(config["atlas_path"])

    target_files: dict[str, tuple[int, str]] = {
        # "T1_seg_mni_1mm.nii.gz": (0, "NearestNeighbor"),
        "T1_5tt.nii.gz": (3, "Linear"),
        # "T1_tractseg_mni_1mm.nii.gz": (3, "Linear"),
        # "T1_slant_mni_1mm.nii.gz": (0, "NearestNeighbor"),
        # "T1_seed_mni_1mm.nii.gz": (0, "NearestNeighbor"),
    }

    for path in total_paths:
        path = Path(path)
        print(path)

        # Extract subject/session information from the BIDS path
        # Assuming the path format is like: /some/path/sub-<id>/ses-<session>...
        parts = path.parts
        subject_id: str | None = None
        session_id: str | None = None

        for part in parts:
            if part.startswith("sub-"):
                subject_id = part
            elif part.startswith("ses-"):
                session_id = part
                break

        if not subject_id or not session_id:
            print(f"Could not extract subject/session from path: {path}")
            continue

        # Set the flair folder path to the derivatives subdirectory
        this_flair_folder_path = (
            derivatives_dir / subject_id / session_id
        )

        # Compute registration
        if not compute_fusion_only:
            # Atlas T1 path (in atlas directory)
            atlas_t1_path = Path(config["t1_path"])

            # FLAIR registered to T1 (output from first step)
            flair_path = (
                this_flair_folder_path
                / "flair_registered2_T1_N4_mni_1mmWarped.nii.gz"
            )

            # Create registration result folder
            registration_result_folder = (
                this_flair_folder_path / "T1_atlas_registered"
            )
            registration_result_folder.mkdir(exist_ok=True)

            if not flair_path.exists():
                print("flair image not found: ", flair_path)
                continue

            # Prepare bind mounts for Apptainer
            project_path = config["proj_path"]
            bind_mounts = [
                project_path,
                str(atlas_t1_path.parent),  # Atlas T1 directory
                str(registration_result_folder),  # Output directory
            ]

            bind_string = ",".join(bind_mounts)

            # Register atlas T1 to FLAIR space via Apptainer
            apptainer_cmd = local["apptainer"][
                "exec",
                "-c",
                "-e",
                "--bind",
                bind_string,
                "--env",
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1",
                str(ants_sif_path),
            ]
            ants_registration = apptainer_cmd["antsRegistrationSyN.sh"]
            # ants_registration(
            #     "-d",
            #     "3",
            #     "-f",
            #     str(flair_path),
            #     "-m",
            #     str(atlas_t1_path),
            #     "-t",
            #     "s",
            #     "-o",
            #     str(registration_result_folder / "atlas_T1_registered2_flair"),
            # )

            # # Apply transforms to each target file
            for target_file in target_files.keys():
                this_file = atlas_path_base / target_file
                print(this_file)

                # Prepare bind mounts for Apptainer for antsApplyTransforms
                bind_mounts = [
                    str(flair_path.parent),  # FLAIR directory (reference)
                    str(this_file.parent),  # Target file directory
                    str(
                        registration_result_folder
                    ),  # Transform files directory
                    str(registration_result_folder),  # Output directory
                ]

                bind_string = ",".join(bind_mounts)

                # Apply transforms via Apptainer
                apptainer_cmd = local["apptainer"][
                    "exec",
                    "-c",
                    "-e",
                    "--bind",
                    bind_string,
                    "--env",
                    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1",
                    str(ants_sif_path),
                ]
                ants_apply_transforms = apptainer_cmd["antsApplyTransforms"]
                ants_apply_transforms(
                    "-d",
                    "3",
                    "-e",
                    str(target_files[target_file][0]),
                    "-n",
                    target_files[target_file][1],
                    "-i",
                    str(this_file),
                    "-o",
                    str(
                        registration_result_folder
                        / f"{target_file.replace('.nii.gz', '_2flair.nii.gz')}"
                    ),
                    "-t",
                    str(
                        registration_result_folder
                        / "atlas_T1_registered2_flair1Warp.nii.gz"
                    ),
                    "-t",
                    str(
                        registration_result_folder
                        / "atlas_T1_registered2_flair0GenericAffine.mat"
                    ),
                    "-r",
                    str(flair_path),
                )

        # Compute fusion
        # Since we only have one atlas now, we just need to copy the transformed files to final fusion outputs
        registration_result_folder = (
            this_flair_folder_path / "T1_atlas_registered"
        )
        flair_path = (
            this_flair_folder_path
            / "flair_registered2_T1_N4_mni_1mmWarped.nii.gz"
        )

        if not flair_path.exists():
            print("FLAIR image not found for fusion step:", flair_path)
            continue

        reference_nii = nib.load(str(flair_path))

        for target in target_files.keys():
            output_file_path = (
                this_flair_folder_path
                / f"{Path(target).name.replace('.nii.gz', '_2flair_fusion.nii.gz')}"
            )

            # Input file from the single atlas transformation
            input_file_path = (
                registration_result_folder
                / f"{target.replace('.nii.gz', '_2flair.nii.gz')}"
            )

            if not input_file_path.exists():
                print(f"Input file does not exist: {input_file_path}")
                continue

            # Load the single transformed file
            nii = nib.load(str(input_file_path))
            data = nii.get_fdata()

            # For a single file, fusion is just copying the data
            fused_map = data.copy()

            # Preserve original data type
            original_dtype = nii.get_data_dtype()
            fused_map = fused_map.astype(original_dtype)

            # Create and save the fused NIfTI image
            fused_nii = nib.Nifti1Image(
                fused_map,
                affine=reference_nii.affine,
                header=reference_nii.header,
            )
            nib.save(fused_nii, str(output_file_path))
