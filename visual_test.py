from pathlib import Path

import matplotlib.pyplot as plt
from monai.visualize import matshow3d
import numpy as np
import SimpleITK as sitk

volume = sitk.GetArrayFromImage(sitk.ReadImage(Path("test_prostate_data_dir/10019_1000019_t2w.mha")))
fig = plt.figure()
matshow3d(volume, fig=fig, title="3D Volume", frames_per_row=3, cmap='gray')
plt.show()