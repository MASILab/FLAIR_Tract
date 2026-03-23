# Utilities
# Leon Cai
# MASI Lab
# July 21, 2022

# Set Up

import numpy as np
import torch, io, gzip
import contextlib

# Default Context (as opposed to torch.no_grad())

@contextlib.contextmanager
def default_context():
    yield

# Function Definitions

def vox2step(streamline_vox):

    streamline_step = np.diff(streamline_vox, axis=0)
    streamline_step = streamline_step / np.sqrt(np.sum(streamline_step ** 2, axis=1, keepdims=True))
    return streamline_step

def step2axis(streamline_step):

    flip = streamline_step[:, 2] < 0
    streamline_axis = streamline_step
    streamline_axis[flip, :] = -streamline_axis[flip, :]
    return streamline_axis

def axis2step(curr_axis, prev_step=None):

    if prev_step is None:
        flip = np.round(np.random.rand(curr_axis.shape[0])).astype(bool)
    else:
        flip = np.sum(curr_axis * prev_step, axis=-1) < 0
    curr_step = curr_axis
    curr_step[flip, :] = -curr_step[flip, :]
    return curr_step

def vox2coor(vox): # Does not protect against out of bounds (i.e. will return negative coordinates or coordinates larger than voxel boundaries)

    coor = np.round(vox).astype(int) 
    return coor

def coor2idx(coor, img): # Returns index of -1 for invalid coordinates

    invalid_idx = np.logical_or(np.any(coor < 0, axis=1), np.any(coor > img.shape, axis=1))
    idx = np.ravel_multi_index(tuple(np.transpose(coor, axes=(1, 0))), img.shape, mode='clip')
    idx[invalid_idx] = -1
    return idx

def vox2trid(vox):

    trid = vox - np.floor(vox) # this can be parallelized (aka already is)...
    return trid

def vox2trii(vox, img): # ...can we parallelize this? We can pack the sequence and then run through this and then unpack it!

    offset = np.transpose(np.array([[[0, 0, 0], 
                                     [1, 0, 0], 
                                     [0, 1, 0],
                                     [0, 0, 1], 
                                     [1, 1, 0],  
                                     [1, 0, 1], 
                                     [0, 1, 1],
                                     [1, 1, 1]]]), axes=(0, 2, 1)).astype(int)
    tric = np.expand_dims(np.floor(vox).astype(int), axis=2) + offset
    trii = np.stack([coor2idx(tric[:, :, c], img) for c in range(8)], axis=1)
    return trii

def triinterp(img, trid, trii, fourth_dim=True):

    assert len(img.shape) == 3 or len(img.shape) == 4, 'img must be a 3D or 4D array (x, y, z, c).'
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=3)
    else:
        fourth_dim = True
    img = np.reshape(img, (-1, img.shape[-1]))

    # Source: https://www.wikiwand.com/en/Trilinear_interpolation

    xd = np.expand_dims(trid[:, 0], axis=1)
    yd = np.expand_dims(trid[:, 1], axis=1)
    zd = np.expand_dims(trid[:, 2], axis=1)
    
    c000 = img[trii[:, 0], :]
    c100 = img[trii[:, 1], :]
    c010 = img[trii[:, 2], :]
    c001 = img[trii[:, 3], :]
    c110 = img[trii[:, 4], :]
    c101 = img[trii[:, 5], :]
    c011 = img[trii[:, 6], :]
    c111 = img[trii[:, 7], :]

    c00 = c000*(1-xd) + c100*xd
    c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd
    c11 = c011*(1-xd) + c111*xd

    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd

    c = c0*(1-zd) + c1*zd

    if not fourth_dim:
        c = c[:, 0]

    return c

def streamline2network(streamline_vox, t1_img):

    streamline_step = vox2step(streamline_vox)          # Cartesian steps 1,...,n-1
    streamline_trid = vox2trid(streamline_vox)          # Distance between two nearest voxels 1,...,n
    streamline_trii = vox2trii(streamline_vox, t1_img)  # Raveled indices for neighboring 8 voxels 1,...,n

    return streamline_trid[:-1, :], streamline_trii[:-1, :], streamline_step

def len2mask(streamlines_length):

    batch_size = len(streamlines_length)
    mask = np.ones((np.max(streamlines_length), len(streamlines_length))) # always batch_first = false
    for b in range(batch_size):
        mask[streamlines_length[b]:, b] = 0
    return mask

def save_tensor_gz(x, fname):

    xb = io.BytesIO()
    torch.save(x, xb)
    xb.seek(0)
    with gzip.open(fname, 'wb') as xbgz:
        xbgz.write(xb.read())

def load_tensor_gz(fname):

    with gzip.open(fname, 'rb') as ybgz:
        yb = io.BytesIO(ybgz.read())
        y = torch.load(yb)
    return y

def onehot(ten, num_classes): # 115 classes for FS aparc+aseg

    oh_ten = torch.nn.functional.one_hot(ten, num_classes=num_classes).unsqueeze(0)
    oh_ten = torch.permute(oh_ten, dims=(0, 4, 1, 2, 3))
    return oh_ten