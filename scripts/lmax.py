"""
    Use this code to check compatibility with dwi2response/dwi2fod. It confirms there are enough unique directions per shell. 
"""
import sys
from pathlib import Path
import numpy as np

def cluster_shells(bvals, tolerance=50):
    """
    Cluster b-values into shells based on a tolerance threshold.
    
    Parameters
    ----------
    bvals : np.ndarray
        1D array of b-values.
    tolerance : float
        Maximum difference to consider b-values part of the same shell.
        
    Returns
    -------
    dict
        Dictionary mapping shell b-value (mean) to list of indices.
    """
    # Sort bvals and keep track of original indices
    sorted_indices = np.argsort(bvals)
    sorted_bvals = bvals[sorted_indices]
    
    shells = {}
    current_shell_bvals = []
    current_shell_indices = []
    
    if len(sorted_bvals) == 0:
        return shells
        
    current_val = sorted_bvals[0]
    
    for i, bval in zip(sorted_indices, sorted_bvals):
        if len(current_shell_bvals) == 0:
            current_shell_bvals.append(bval)
            current_shell_indices.append(i)
        else:
            if abs(bval - np.mean(current_shell_bvals)) <= tolerance:
                current_shell_bvals.append(bval)
                current_shell_indices.append(i)
            else:
                # Save previous shell
                shell_key = int(round(np.mean(current_shell_bvals)))
                shells[shell_key] = current_shell_indices
                # Start new shell
                current_shell_bvals = [bval]
                current_shell_indices = [i]
    
    # Save last shell
    if current_shell_bvals:
        shell_key = int(round(np.mean(current_shell_bvals)))
        shells[shell_key] = current_shell_indices
        
    return shells

def calculate_max_lmax(n_directions):
    """
    Calculate max even SH degree supported by n_directions.
    Returns 0 if insufficient for L=2.
    """
    if n_directions < 6:
        return 0
    
    max_l = 0
    for l in range(2, 20, 2):
        required = (l + 1) * (l + 2) // 2
        if n_directions >= required:
            max_l = l
        else:
            break
    return max_l

def check_dwi_compatibility(bvals_path, bvecs_path=None):
    """
    Analyze b-values to detect potential MRtrix processing failures.
    """
    bvals_path = Path(bvals_path)
    if not bvals_path.exists():
        print(f"Error: File not found {bvals_path}")
        return

    bvals = np.loadtxt(bvals_path)
    if bvals.ndim > 1:
        bvals = bvals.squeeze()
        
    shells = cluster_shells(bvals)
    
    print(f"Total volumes: {len(bvals)}")
    print(f"Detected shells: {len(shells)}")
    print("-" * 40)
    
    failure_detected = False
    
    # Sort shells by b-value
    sorted_shell_keys = sorted(shells.keys())
    
    for bval in sorted_shell_keys:
        indices = shells[bval]
        n_vols = len(indices)
        
        # Check for b0
        if bval < 50:
            print(f"b=0 Shell: {n_vols} volumes")
            continue
            
        lmax = calculate_max_lmax(n_vols)
        status = "OK"
        
        # MRtrix CSD requires at least L=2 (6 directions)
        if lmax < 2:
            status = "FAIL (Too few directions for L=2)"
            failure_detected = True
            
        print(f"b={bval} Shell: {n_vols} volumes -> Max L={lmax} [{status}]")
        
    print("-" * 40)
    if failure_detected:
        print(f"FAIL_PATH: {bvals_path}")
        print("WARNING: One or more shells have insufficient directions for CSD.")
    else:
        print("All non-zero shells support at least L=2.")

    

if __name__ == "__main__":
    data_file = Path(sys.argv[1])
    bval_file = data_file.parent.joinpath("dwmri.bval")
    bvec_file = data_file.parent.joinpath("dwmri.bvec")

    check_dwi_compatibility(bval_file, bvec_file)
        
