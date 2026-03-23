# Dataset Definitions
# Written by Leon Cai
# Modified by Tian Yu
# MASI Lab
# Summer 2023

# Set Up

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import nibabel as nib

from utils import onehot


class DT1Dataset(Dataset):
    def __init__(self, dt1_dirs, num_batches):
        super(DT1Dataset, self).__init__()
        self.dt1_dirs = dt1_dirs
        self.num_batches = num_batches

    def __getitem__(self, index):
        dt1_dir = self.dt1_dirs[index]

        # Try doing more image stuff in RAM
        #in_dir = os.path.dirname(dt1_dir)
        in_dir=os.path.join("/valiant02/masi/schwat1/projects/spie_flair_tract_extension/FLAIR_processing/BIDS_format/derivatives",dt1_dir)
        fod_file = os.path.join(in_dir, "dwmri_fod_mni_trix.nii.gz")
        # t1_file = os.path.join(in_dir, "T1_N4_mni_1mm.nii.gz")
        # mask_file = os.path.join(in_dir, "T1_seg_mni_1mm.nii.gz")
        # act_file = os.path.join(in_dir, "T1_5tt_mni_1mm.nii.gz")
        #t1_file = os.path.join(in_dir, "T1_N4_mni_2mm.nii.gz")
        t1_file = os.path.join(in_dir, "flair_registered2_T1_N4_mni_1mmWarped.nii.gz")
        if 0:
            mask_file = os.path.join(in_dir, "T1_seg_mni_2mm.nii.gz")
            act_file = os.path.join(in_dir, "T1_5tt_mni_2mm.nii.gz")
            tseg_file = os.path.join(in_dir, "T1_tractseg_mni_2mm.nii.gz")
            slant_file = os.path.join(in_dir, "T1_slant_mni_2mm.nii.gz")
        elif 0:
            mask_file = os.path.join(in_dir, "T1_seg_mni_2mm_2flair.nii.gz")
            act_file = os.path.join(in_dir, "T1_5tt_mni_2mm_2flair.nii.gz")
            tseg_file = os.path.join(in_dir, "T1_tractseg_mni_2mm_2flair.nii.gz")
            slant_file = os.path.join(in_dir, "T1_slant_mni_2mm_2flair.nii.gz")
        else:
            mask_file = os.path.join(in_dir, "T1_seg_mni_1mm_2flair_fusion.nii.gz")
            act_file = os.path.join(in_dir, "T1_5tt_2flair_fusion.nii.gz")
            tseg_file = os.path.join(in_dir, "T1_tractseg_2flair_fusion.nii.gz")
            slant_file = os.path.join(in_dir, "T1_slant_2flair_fusion.nii.gz")

        fod_img = nib.load(fod_file).get_fdata()
        # t1_img = nib.load(t1_file).get_fdata()[:-1, :-1, :-1]
        # mask_img = nib.load(mask_file).get_fdata()[:-1, :-1, :-1].astype(bool)
        # act_img = nib.load(act_file).get_fdata()[:-1, :-1, :-1, :-1]
        t1_img = nib.load(t1_file).get_fdata()
        mask_img = nib.load(mask_file).get_fdata().astype(bool)
        act_img = nib.load(act_file).get_fdata()[:, : , :, :-1]
        tseg_img = nib.load(tseg_file).get_fdata()
        slant_img = nib.load(slant_file).get_fdata()

        fod = torch.FloatTensor(
            np.expand_dims(np.transpose(fod_img, axes=(3, 0, 1, 2)), axis=0)
        )

        t1_ten = torch.FloatTensor(
            np.expand_dims(t1_img / np.median(t1_img[mask_img]), axis=(0, 1))
        )
        act_ten = torch.FloatTensor(
            np.expand_dims(np.transpose(act_img, axes=(3, 0, 1, 2)), axis=0)
        )
        tseg_ten = torch.FloatTensor(
            np.expand_dims(np.transpose(tseg_img, axes=(3, 0, 1, 2)), axis=0)
        )
        slant_ten = torch.FloatTensor(
            np.expand_dims(np.transpose(slant_img, axes=(3, 0, 1, 2)), axis=0)
        )
        ten_2mm = torch.cat(
            (t1_ten, act_ten, tseg_ten, slant_ten), dim=1)  # everything at 2mm
        # ten_1mm = torch.cat((t1_ten, act_ten), dim=1)  # trying only using...
        # ten_2mm = torch.cat((tseg_ten, slant_ten), dim=1)  # ...SLANT/WML at 2mm

        brain = torch.FloatTensor(np.expand_dims(mask_img, axis=(0, 1)))

        # Streamlines (must be read from disk)

        b = np.random.randint(0, self.num_batches)
        step = torch.load(os.path.join(in_dir,"packed_trk_data", "step_{:06}.pt".format(b)), weights_only=False)
        trid = torch.load(os.path.join(in_dir,"packed_trk_data", "trid_{:06}.pt".format(b)), weights_only=False)
        trii = torch.load(os.path.join(in_dir,"packed_trk_data", "trii_{:06}.pt".format(b)), weights_only=False)
        mask = torch.load(os.path.join(in_dir,"packed_trk_data", "mask_{:06}.pt".format(b)), weights_only=False)
        return (ten_2mm, fod, brain, step, trid, trii, mask)
        # return (ten_1mm, ten_2mm, fod, brain, step, trid, trii, mask)

    def __len__(self):
        return len(self.dt1_dirs)


def unload(ten_2mm, fod, brain, step, trid, trii, mask):
    # def unload(ten, step, trid, trii, mask, tdi):

    # ten_1mm = ten_1mm[0]
    ten_2mm = ten_2mm[0]
    fod = fod[0]
    brain = brain[0]
    step = step[0]
    trid = nn.utils.rnn.PackedSequence(
        trid.data[0],
        batch_sizes=trid.batch_sizes[0],
        sorted_indices=trid.sorted_indices[0],
        unsorted_indices=trid.unsorted_indices[0],
    )
    trii = nn.utils.rnn.PackedSequence(
        trii.data[0],
        batch_sizes=trii.batch_sizes[0],
        sorted_indices=trii.sorted_indices[0],
        unsorted_indices=trii.unsorted_indices[0],
    )
    mask = mask[0]

    return ten_2mm, fod, brain, step, trid, trii, mask
    # return ten, fod, brain, step, trid, trii, mask
