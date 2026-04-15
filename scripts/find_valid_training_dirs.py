from pathlib import Path
from tqdm.rich import tqdm
import random

pth = Path("/valiant02/masi/schwat1/projects/spie_flair_tract_extension/FLAIR_processing/BIDS_format/derivatives/")
req_files = ["T1_seg_mni_1mm_2flair_fusion.nii.gz","T1_5tt_2flair_fusion.nii.gz","T1_tractseg_2flair_fusion.nii.gz","T1_slant_2flair_fusion.nii.gz","flair_registered2_T1_N4_mni_1mmWarped.nii.gz","dwmri_fod_mni_trix.nii.gz"]

valid = []
for subject in tqdm(pth.glob("sub-*"), total=150):
    found_files = dict()
    for file in req_files:
        result = set(subject.rglob(file))
        if len(result) > 1:
            for r in result:
                key = r.parent.name
                if key in found_files:
                    found_files[key].append(r)
                else:
                    found_files[key] = [r]
        elif len(result) == 1:
            entry = result.pop()
            key = entry.parent.name
            if key in found_files:
                found_files[key].append(entry)
            else:
                found_files[key] = [entry]
        else:
            continue
            
    for key in found_files:
        if len(found_files[key]) == len(req_files):
            sub = str(Path("/").joinpath(*list(found_files[key][0].parts[1:11]), '\n'))
            valid.append(sub)

with open("/fs5/p_masi/schwat1/spie_flair_tract_extension/model/valid_model_subjects.txt", "w") as out_file:
    out_file.writelines(valid)

    
inference_path = Path("/fs5/p_masi/schwat1/spie_flair_tract_extension/model/valid_inference_subjects.txt")
train_path = Path("/fs5/p_masi/schwat1/spie_flair_tract_extension/model/valid_train_subjects.txt")
val_path = Path("/fs5/p_masi/schwat1/spie_flair_tract_extension/model/valid_val_subjects.txt")

if inference_path.exists():
    inference_path.unlink()
if train_path.exists():
    train_path.unlink()
if val_path.exists():
    val_path.unlink()

train_val = []

for path in valid:
    line = str(path)
    if 'blsa' in line.lower():
        with open(inference_path, "a") as inference_file:
            inference_file.write(line)
    else:
        train_val.append(line)


train_selection = random.sample(train_val, k=100)
with open(train_path, "w") as train_file:
    train_file.writelines(train_selection)

val_selection = set(train_val).difference(train_selection)
with open(val_path, "w") as val_file:
    val_file.writelines(val_selection)
