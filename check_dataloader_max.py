import os
from pathlib import Path
from pprint import pprint
import torchio as tio
import SimpleITK as sitk

import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.transforms import Compose, ToTensord, EnsureTyped, NormalizeIntensityd, Transform, ConcatItemsd, \
    ScaleIntensityRangePercentilesd, ClipIntensityPercentiles, ClipIntensityPercentilesD
from monai.visualize import matshow3d
from tqdm import tqdm

from train_Coarse import get_dataloaders

# dir = Path('./Task205_picai_lesion/labelsTr')
# ex = dir.exists()
#
# args={'batch_size': 2}
# train_dataloader, val_dataloader = get_dataloaders(args)
# tbar = tqdm(train_dataloader)
# zero = 0
# one = 1
# empty_list = []
# for step, data3D in enumerate(tbar):
#     tbar.set_description(f"ID: {data3D['id'][0]}")
#     t2w = data3D["t2w"]  # [B, 1, 128, 128, 128]
#     adc = data3D["adc"]
#     dwi = data3D["dwi"]
#     gt3D = data3D["mask"]
#
#     # print(f'gt3D shape is {gt3D.shape}')  # [1, 1, 128, 128, 128]
#
#     gt3D = gt3D[0,0]
#     gt3D = gt3D > 0
#     if gt3D.sum() > 0:
#         zero += 1
#     else:
#         fig1 = plt.figure()
#         matshow3d(gt3D[0, 0], fig=fig1, title=f"{data3D['id'][0]}")
#
#         # fig2 = plt.figure()
#         # matshow3d(t2w[0, 0], fig=fig2, title="3D Volume", cmap="gray")
#
#         plt.show()
#
#         print(f'gt3D shape is {gt3D.shape}')
#
#         one += 1
#         empty_list.append(data3D['id'][0])
#
# print(zero, one)
# pprint(empty_list)
#
#
class TorchIOCropOrPadTransform(Transform):
    def __init__(self, target_shape=(128, 128, 128)):
        super().__init__()
        self.crop_or_pad = tio.CropOrPad(target_shape=target_shape, mask_name='mask')

    def __call__(self, data):
        subject = tio.Subject(
            t2w=tio.ScalarImage(tensor=data["t2w"]),
            adc=tio.ScalarImage(tensor=data["adc"]),
            dwi=tio.ScalarImage(tensor=data["dwi"]),
            mask=tio.LabelMap(tensor=data["mask"]),
        )

        out = self.crop_or_pad(subject)

        return {
            "t2w": out["t2w"].data,    # torch.Tensor, shape: (1, D, H, W)
            "adc": out["adc"].data,
            "dwi": out["dwi"].data,
            "mask": out["mask"].data,
        }


transform=Compose([
    lambda data: {  # -- 1
        "t2w": sitk.GetArrayFromImage(sitk.ReadImage(data["t2w"]))[None, :, :, :],
        "adc": sitk.GetArrayFromImage(sitk.ReadImage(data["adc"]))[None, :, :, :],
        "dwi": sitk.GetArrayFromImage(sitk.ReadImage(data["dwi"]))[None, :, :, :],
        "mask": sitk.GetArrayFromImage(sitk.ReadImage(data["mask"]))[None, :, :, :],
    },
    ToTensord(keys=["t2w", "adc", "dwi", "mask",]),  # -- 2
    TorchIOCropOrPadTransform(),
    EnsureTyped(keys=["t2w", "adc", "dwi"], dtype=torch.float),
    EnsureTyped(keys=["mask"], dtype=torch.long),
    # ConcatItemsd(keys=["t2w", "adc", "dwi"], name="image", dim=0),
    ClipIntensityPercentilesD(
        keys=["t2w", "adc", "dwi"],
        lower=0.5, upper=99.5,  # 0.5–99.5 百分位
        sharpness_factor=None,  # None=硬裁剪
        channel_wise=True  # 每通道单独计算百分位
    ),
    NormalizeIntensityd(keys=["t2w", "adc", "dwi"], nonzero=True, channel_wise=False),
    ])


t2w =   '/home/lib/PycharmProjects/SAM-Med3D/data/Task205_picai_lesion/imagesTr/10140_1000142_0000.nii.gz'
adc =  '/home/lib/PycharmProjects/SAM-Med3D/data/Task205_picai_lesion/imagesTr/10140_1000142_0001.nii.gz'
dwi =  '/home/lib/PycharmProjects/SAM-Med3D/data/Task205_picai_lesion/imagesTr/10140_1000142_0002.nii.gz'
label = '/home/lib/PycharmProjects/SAM-Med3D/data/Task205_picai_lesion/labelsTr/10140_1000142.nii.gz'

label_npy = sitk.GetArrayFromImage(sitk.ReadImage(t2w))
print(f'未经transform之前，最大值为{np.max(label_npy)}')


################# Transforming ################
data = {"t2w": t2w,
        "adc": adc,
        "dwi": dwi,
        "mask": label}
output = transform(data)
t2w_t = output["t2w"]
mask_t = output["mask"]
################# Transforming ################


print(f'经过transform之后，最大值为{torch.max(t2w_t)}')

print(mask_t.shape)
#
# fig = plt.figure()
# matshow3d(mask_t, fig=fig, title=f"hh")
# plt.show()


