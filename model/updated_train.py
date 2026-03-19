import torch
import torch.nn as nn
from dipy.io.streamline import load_tractogram, save_tractogram, StatefulTractogram
from dipy.io.stateful_tractogram import Space
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

from tqdm import tqdm
import numpy as np
import nibabel as nib
import pathlib as pl
import os
import sys
import yaml

from utils import streamline2network, len2mask, triinterp


# Global variable for worker processes (set by initializer)
_worker_t1_img = None
_worker_t1_shape = None
_worker_t1_dtype = None
_t1_shm_name = None


def _init_worker(t1_img_path, shm_name=None):
    """
    Initialize worker process: load T1 image once per worker.
    
    If shm_name is provided, attach to shared memory instead of loading from disk.
    """
    global _worker_t1_img, _worker_t1_shape, _worker_t1_dtype, _t1_shm_name
    
    if shm_name is not None:
        # Attach to shared memory (zero-copy)
        _t1_shm_name = shm_name
        # We need to know shape/dtype; pass via environment or global
        _worker_t1_shape = tuple(map(int, os.environ.get('T1_SHAPE', '').split(',')))
        _worker_t1_dtype = np.dtype(os.environ.get('T1_DTYPE', 'float32'))
        shm = shared_memory.SharedMemory(name=shm_name)
        _worker_t1_img = np.ndarray(_worker_t1_shape, dtype=_worker_t1_dtype, buffer=shm.buf)
    else:
        # Fallback: load from disk once per worker (still better than per-streamline)
        _worker_t1_img = nib.load(t1_img_path).get_fdata()


def _process_streamline_chunk(args):
    """
    Worker function to process a chunk of streamlines.
    
    Args:
        args: tuple of (streamline_chunk, cuts_chunk)
    
    Returns:
        list of tuples: [(step, trid, trii, length), ...] for all streamlines in chunk
    """
    global _worker_t1_img
    streamline_chunk, cuts_chunk = args
    
    results = []
    for streamline_vox, cut in zip(streamline_chunk, cuts_chunk):
        # Process the streamline segment(s)
        segments = [streamline_vox]
        for seg in segments:
            trid, trii, step = streamline2network(seg, _worker_t1_img)
            results.append((step, trid, trii, step.shape[0]))
    
    return results


if __name__ == '__main__':

    with open("prep_pt_config.yaml", 'r') as yml:
        config = yaml.safe_load(yml.read())
    
    in_dir = sys.argv[1]
    num_streamlines = config["num_streamlines"]
    batch_size = config["batch_size"]
    t1_file = config["t1_file"]
    num_workers = config.get("num_workers", 8)
    chunk_size = config.get("chunk_size", 1000)  # Process N streamlines per task

    # Parse inputs
    print('prep_pt.py: Parsing inputs...')
    assert os.path.exists(in_dir), 'Input directory does not exist.'
    
    trk_file = str(list(pl.Path(in_dir).rglob("moved.trk"))[0])
    assert os.path.exists(trk_file) and os.path.exists(t1_file), 'Input directory missing files.'

    out_dir = os.path.join(in_dir, 'packed_trk_data')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    num_streamlines = int(num_streamlines)
    batch_size = int(batch_size)
    num_batches = np.ceil(num_streamlines / batch_size).astype(int)

    print('prep_pt.py: Input Directory:   {}'.format(in_dir))
    print('prep_pt.py: Number of Batches: {}'.format(num_batches))
    print('prep_pt.py: Output Directory:  {}'.format(out_dir))
    print('prep_pt.py: Using {} workers, chunk_size={}'.format(num_workers, chunk_size))

    # Load T1 image for shared memory setup
    print('prep_pt.py: Loading T1 image...')
    t1_img = nib.load(t1_file).get_fdata()
    t1_shape = t1_img.shape
    t1_dtype = str(t1_img.dtype)
    
    # Create shared memory for zero-copy access (optional but recommended for large images)
    use_shared_memory = True  # Set to False if you encounter issues
    shm = None
    shm_name = None
    
    if use_shared_memory and t1_img.nbytes < 2 * 1024**3:  # Only if <2GB
        try:
            shm = shared_memory.SharedMemory(create=True, size=t1_img.nbytes)
            shm_name = shm.name
            # Copy data into shared memory
            shared_array = np.ndarray(t1_shape, dtype=t1_img.dtype, buffer=shm.buf)
            shared_array[:] = t1_img[:]
            print(f'prep_pt.py: Created shared memory "{shm_name}" for T1 image ({t1_img.nbytes / 1024**2:.1f} MB)')
        except Exception as e:
            print(f'prep_pt.py: Shared memory creation failed ({e}), falling back to per-worker loading')
            use_shared_memory = False
            shm_name = None
    
    # Set environment variables for worker initializer (shape/dtype for shared memory)
    os.environ['T1_SHAPE'] = ','.join(map(str, t1_shape))
    os.environ['T1_DTYPE'] = t1_dtype

    # Load tractography
    print('prep_pt.py: Loading tractography...')
    tractogram = load_tractogram(trk_file, reference='same', to_space=Space.VOX)
    streamlines = tractogram.streamlines
    shuffle_idxs = np.linspace(0, len(streamlines)-1, len(streamlines)).astype(int)
    np.random.shuffle(shuffle_idxs)
    streamlines = streamlines[shuffle_idxs]
    
    if num_streamlines < len(streamlines):
        streamlines = streamlines[:num_streamlines]

    # Pre-generate random cuts for reproducibility
    cuts = []
    for sl in streamlines:
        if sl.shape[0] > 4:
            cuts.append(np.random.randint(2, sl.shape[0]-2))
        else:
            cuts.append(2)

    # Chunk streamlines for batched processing
    chunks = []
    for i in range(0, len(streamlines), chunk_size):
        end_i = min(i + chunk_size, len(streamlines))
        chunks.append((streamlines[i:end_i], cuts[i:end_i]))
    
    print(f'prep_pt.py: Processing {len(streamlines)} streamlines in {len(chunks)} chunks...')
    
    # Collect all processed results
    all_steps = []
    all_trids = []
    all_triis = []
    all_lens = []
    
    with ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=_init_worker,
        initargs=(t1_file if not use_shared_memory else None, shm_name)
    ) as executor:
        # Submit chunked tasks
        futures = [executor.submit(_process_streamline_chunk, chunk) for chunk in chunks]
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(chunks), 
                          desc='prep_pt.py: Processing streamlines'):
            try:
                results = future.result()
                for step, trid, trii, length in results:
                    all_steps.append(torch.FloatTensor(step))
                    all_trids.append(torch.FloatTensor(trid))
                    all_triis.append(torch.LongTensor(trii))
                    all_lens.append(length)
            except Exception as e:
                print(f"Error processing chunk: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Cleanup shared memory
    if shm is not None:
        shm.close()
        shm.unlink()
        print('prep_pt.py: Cleaned up shared memory')

    # Save in batches
    print('prep_pt.py: Saving batches...')
    for b in tqdm(range(num_batches), desc='prep_pt.py: Saving batches'):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, len(all_steps))
        
        batch_steps = all_steps[start_idx:end_idx]
        batch_trids = all_trids[start_idx:end_idx]
        batch_triis = all_triis[start_idx:end_idx]
        batch_lens = all_lens[start_idx:end_idx]
        
        if batch_steps:
            torch.save(nn.utils.rnn.pad_sequence(batch_steps, batch_first=False), 
                      os.path.join(out_dir, f'step_{b:06}.pt'), 
                      _use_zip_format=True)
            torch.save(nn.utils.rnn.pack_sequence(batch_trids, enforce_sorted=False), 
                      os.path.join(out_dir, f'trid_{b:06}.pt'),
                      _use_zip_format=True)
            torch.save(nn.utils.rnn.pack_sequence(batch_triis, enforce_sorted=False), 
                      os.path.join(out_dir, f'trii_{b:06}.pt'),
                      _use_zip_format=True)
            torch.save(torch.FloatTensor(len2mask(batch_lens)), 
                      os.path.join(out_dir, f'mask_{b:06}.pt'),
                      _use_zip_format=True)
    
    print(f"prep_pt.py: Preprocessing complete. Saved {num_batches} batches to {out_dir}")
