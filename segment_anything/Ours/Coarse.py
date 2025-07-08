from pathlib import Path

import medim
import torch
import torch.nn as nn

from segment_anything import sam_model_registry3D


class SAM_Coarse_Seg(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()
        ckpt_path = "../../work_dir/Task2203_picai_gland/sam_model_142_step_dice:0.9792470335960388_best.pth"
        assert not Path(ckpt_path).exists(), f"{ckpt_path} does not exist"
        self.sam_med3d_model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to('cuda:0')
        self.sam_med3d_model.eval()

        # Fine-tuning parameters
        for p in self.parameters():
            p.requires_grad = True

        # Frozen SAM-Med3D parameters
        for p in self.sam_med3d_model.parameters():
            p.requires_grad = False


    def forward(self, image, labels):
        t2w = image[:, 0:1, :, :, :]  # [1, 1， 128, 128, 128]
        adc = image[:, 1:2, :, :, :]
        dwi = image[:, 2:3, :, :, :]

        t2w_outputs = self.image_embed(t2w)  # [1, 384, 8, 8, 8]
        adc_outputs = self.image_embed(adc)
        dwi_outputs = self.image_embed(dwi)

        # Feature fusion
        # TODO
        feature_map = torch.randn([1, 384, 8, 8, 8])

    @torch.no_grad()
    def image_embed(self, image3D):  # image [1, 1, 128, 128, 128]
        image_embedding = self.sam_med3d_model.image_encoder(image3D)
        return image_embedding
