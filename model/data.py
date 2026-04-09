<<<<<<< HEAD
import os
=======
# Dataset Definitions
# Written by Leon Cai(GOAT)
# Modified by Tian Yu
# MASI Lab
# Summer 2023

# Set Up

>>>>>>> 28f1f798228a5e6cce958fc63dcb545323809b18
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset


class DT1Dataset(Dataset):
    """
    A dataset class that caches NIfTI data to disk to avoid repeated I/O.

    This class checks for a pre-processed .npy file in a cache directory.
    If the cache exists and is valid, it loads from disk. If not, it loads 
    the NIfTI, saves the .npy cache, and returns the data.

    Attributes
    ----------
    dt1_dirs : List[Path]
        List of directories containing subject data.
    num_batches : int
        Number of batches for streamline data indexing.
    cache_root : Path
        Absolute directory path where cached .npy files are stored.
    base_data_path : Path
        Base directory where the subject data resides.
    """

    def __init__(
        self,
        dt1_dirs: List[str],
        num_batches: int,
        cache_root: str = "./cache",
        base_data_path: Optional[str] = None,
    ):
        """
        Initialize the PersistentDT1Dataset.

        Parameters
        ----------
        dt1_dirs : List[str]
            List of relative directory paths for each subject relative to base_data_path.
            If base_data_path is None, these are treated as absolute paths.
        num_batches : int
            Number of batches for streamline data indexing.
        cache_root : str, optional
            Directory path for storing cached data files. Will be resolved to absolute.
        base_data_path : str, optional
            Base directory for the input data. If None, dt1_dirs are treated as absolute.
        """
        super(DT1Dataset, self).__init__()
        self.dt1_dirs = [Path(d) for d in dt1_dirs]
        self.num_batches = num_batches
        
        # CRITICAL: Resolve to absolute path to ensure consistency across workers
        self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

        if base_data_path is not None:
            self.base_data_path = Path(base_data_path).resolve()
        else:
            self.base_data_path = None

    def _get_cache_path(self, subject_id: str, file_name: str) -> Path:
        """
        Generate the cache file path for a specific subject and file.

        Parameters
        ----------
        subject_id : str
            The identifier for the subject (directory name).
        file_name : str
            The original file name to be cached.

        Returns
        -------
        Path
            The absolute path to the cached .npy file.
        """
        safe_subject_id = subject_id.replace("/", "_").replace("\\", "_")
        safe_file_name = file_name.replace(".nii.gz", ".npy").replace(".nii", ".npy")
        return self.cache_root / f"{safe_subject_id}_{safe_file_name}"

    def _load_or_cache_data(self, file_path: Path, subject_id: str) -> np.ndarray:
        """
        Load data from cache or generate cache from NIfTI.

        This function handles cache creation with process-unique temporary files
        to prevent race conditions during multiprocessing. It ensures all paths
        are absolute to avoid working directory discrepancies.

        Parameters
        ----------
        file_path : Path
            Path to the source NIfTI file.
        subject_id : str
            Subject identifier for cache naming.

        Returns
        -------
        np.ndarray
            The loaded image data.

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        RuntimeError
            If cache writing fails.
        """
        cache_path = self._get_cache_path(subject_id, file_path.name).resolve()

        # Attempt to load from cache if it exists
        if cache_path.exists():
            try:
                return np.load(str(cache_path))
            except (FileNotFoundError, OSError):
                # Handle corrupted files by removing and rebuilding
                cache_path.unlink(missing_ok=True)

        # Verify source file exists before attempting load
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        try:
            img = nib.load(file_path)
            data = img.get_fdata()
        except Exception as e:
            raise RuntimeError(f"Failed to load NIfTI {file_path}: {e}") from e

        # CRITICAL: Ensure cache directory exists in worker process
        # This handles cases where worker processes start before directory propagation
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Generate a process-unique temporary filename to prevent collisions
        # if the same subject is processed by multiple workers simultaneously.
        # Using .resolve() ensures the path is absolute.
        temp_suffix = f".tmp.{os.getpid()}.npy"
        temp_path = cache_path.with_suffix(temp_suffix)

        try:
            # Save to temporary file
            # np.save expects a string or file-like object
            np.save(str(temp_path), data)

            # Verify temp file exists explicitly before moving
            # This catches potential NFS/local disk latency issues
            if not temp_path.exists():
                raise FileNotFoundError(f"Temp file not visible after save: {temp_path}")

            # Atomically move temp file to final cache location
            # Path.replace is atomic on POSIX systems when on the same filesystem
            temp_path.replace(cache_path)

        except FileNotFoundError as e:
            # Handle race condition: another worker may have completed the cache
            if cache_path.exists():
                # Another worker won the race; safe to use their cache
                temp_path.unlink(missing_ok=True)
                return np.load(str(cache_path))
            else:
                # Genuine error: temp file missing and no cache exists
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Failed to write cache {cache_path}: {e}") from e
        except Exception as e:
            # Clean up temp file on any other error
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to write cache {cache_path}: {e}") from e

        return data

    def __getitem__(self, index: int) -> tuple:
        dt1_dir = self.dt1_dirs[index]
        ses = dt1_dir.name
        subject_id = dt1_dir.parent.name + "_" + ses
        
        # Construct full input directory path
        if self.base_data_path:
            in_dir = self.base_data_path / dt1_dir
        else:
            in_dir = dt1_dir

        # Define file paths
        fod_file = in_dir / "dwmri_fod_mni_trix.nii.gz"
        t1_file = in_dir / "flair_registered2_T1_N4_mni_1mmWarped.nii.gz"
        mask_file = in_dir / "T1_seg_mni_1mm_2flair_fusion.nii.gz"
        act_file = in_dir / "T1_5tt_2flair_fusion.nii.gz"
        tseg_file = in_dir / "T1_tractseg_2flair_fusion.nii.gz"
        slant_file = in_dir / "T1_slant_2flair_fusion.nii.gz"

        # Load or cache each volume
        fod_img = self._load_or_cache_data(fod_file, subject_id)
        t1_img = self._load_or_cache_data(t1_file, subject_id)
        mask_img = self._load_or_cache_data(mask_file, subject_id).astype(bool)
        act_img = self._load_or_cache_data(act_file, subject_id)[:, :, :, :-1]
        tseg_img = self._load_or_cache_data(tseg_file, subject_id)
        slant_img = self._load_or_cache_data(slant_file, subject_id)

        # Tensor conversion
        fod = torch.from_numpy(
            np.expand_dims(np.transpose(fod_img, axes=(3, 0, 1, 2)), axis=0)
        ).float()

        mask_sum = np.sum(mask_img)
        median_val = np.median(t1_img[mask_img]) if mask_sum > 0 else 1.0

        t1_ten = torch.from_numpy(
            np.expand_dims(t1_img / median_val, axis=(0, 1))
        ).float()

        act_ten = torch.from_numpy(
            np.expand_dims(np.transpose(act_img, axes=(3, 0, 1, 2)), axis=0)
        ).float()

        tseg_ten = torch.from_numpy(
            np.expand_dims(np.transpose(tseg_img, axes=(3, 0, 1, 2)), axis=0)
        ).float()

        slant_ten = torch.from_numpy(
            np.expand_dims(np.transpose(slant_img, axes=(3, 0, 1, 2)), axis=0)
        ).float()

        ten_2mm = torch.cat((t1_ten, act_ten, tseg_ten, slant_ten), dim=1)
        brain = torch.from_numpy(np.expand_dims(mask_img, axis=(0, 1))).float()

        # Streamlines loading (memory mapped)
        b = np.random.randint(0, self.num_batches)
        trk_path = in_dir / "packed_trk_data"

        step_file = trk_path / "step_{:06}.pt".format(b)
        if not step_file.exists():
            raise FileNotFoundError(f"Streamline file not found: {step_file}")

        step = torch.load(
            str(step_file),
            weights_only=False,
            mmap=True
        )
        trid = torch.load(
            str(trk_path / "trid_{:06}.pt".format(b)),
            weights_only=False,
            mmap=True
        )
        trii = torch.load(
            str(trk_path / "trii_{:06}.pt".format(b)),
            weights_only=False,
            mmap=True
        )
        mask = torch.load(
            str(trk_path / "mask_{:06}.pt".format(b)),
            weights_only=False,
            mmap=True
        )

        return ten_2mm, fod, brain, step, trid, trii, mask

    def __len__(self) -> int:
        return len(self.dt1_dirs)


def unload(ten_2mm, fod, brain, step, trid, trii, mask):
    """
    Unpack and process batched data.

    This function expects batched inputs (typically from a DataLoader) and
    extracts the first element or processes the PackedSequence attributes.

    Parameters
    ----------
    ten_2mm : torch.Tensor
        Batched tensor of 2mm data.
    fod : torch.Tensor
        Batched tensor of FOD data.
    brain : torch.Tensor
        Batched tensor of brain masks.
    step : torch.Tensor
        Batched step data.
    trid : nn.utils.rnn.PackedSequence
        Batched PackedSequence for trid.
    trii : nn.utils.rnn.PackedSequence
        Batched PackedSequence for trii.
    mask : torch.Tensor
        Batched mask data.

    Returns
    -------
    tuple
        Unpacked data elements.
    """
    ten_2mm = ten_2mm[0]
    fod = fod[0]
    brain = brain[0]
    step = step[0]
    
    # Reconstruct PackedSequence from the first batch element
    # Ensure attributes exist before accessing to prevent AttributeError
    if hasattr(trid, 'data') and trid.data.dim() > 0:
        trid = nn.utils.rnn.PackedSequence(
            trid.data[0],
            batch_sizes=trid.batch_sizes[0],
            sorted_indices=trid.sorted_indices[0],
            unsorted_indices=trid.unsorted_indices[0],
        )
    
    if hasattr(trii, 'data') and trii.data.dim() > 0:
        trii = nn.utils.rnn.PackedSequence(
            trii.data[0],
            batch_sizes=trii.batch_sizes[0],
            sorted_indices=trii.sorted_indices[0],
            unsorted_indices=trii.unsorted_indices[0],
        )
        
    mask = mask[0]

    return ten_2mm, fod, brain, step, trid, trii, mask
