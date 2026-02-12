from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space
import sys
trk_file = sys.argv[1]
sft = load_tractogram(trk_file, 'same', bbox_valid_check=False)
# Ensure the tractogram is in the correct space for saving as .tck
# .tck files assume RAS+ (world) coordinates and do not store voxel-to-RAS transform,
# so we convert to world space if needed.
if sft.space != Space.RASMM:
    sft.to_rasmm()
# Save as .tck
tck_file = trk_file.removesuffix("trk") + "tck"
save_tractogram(sft, tck_file)


