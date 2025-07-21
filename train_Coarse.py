import argparse
import json
import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path
from pprint import pprint
import torchio as tio
import numpy as np
from matplotlib import pyplot as plt
from monai.data import Dataset, DataLoader
from monai.losses import DiceCELoss
import matplotlib.pyplot as plt
from monai.visualize import matshow3d
import torch.nn.functional as F
from monai.transforms import (Compose, Transform, EnsureTyped,
                              ConcatItemsd, NormalizeIntensityd, ToTensord, ClipIntensityPercentilesD)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from segment_anything.Ours.common import loralib as lora
from utils.infer_utils import random_sample_next_click
from utils_our import get_network
import SimpleITK as sitk

import torch
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Using TorchIO images without a torchio\.SubjectsLoader.*",
    category=UserWarning,
    module=r"torchio\.data\.image"
)


from utils.click_method import get_next_click3D_torch_2
from utils.data_paths import img_datas

# #######    Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='adalora_try_2_try')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='ckpt/sam_med3d.pth')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--mode', type=str, default='sam_adalora')

# train
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()
device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = os.path.join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = os.path.join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

join = os.path.join


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
            "id": data["id"]           # id 保持不变
        }

def get_dataloaders(args):
    # ['/home/lib/PycharmProjects/SAM-Med3D/data/Task204_picai_lesion']
    task_dir = Path(img_datas[0])
    assert task_dir.exists(), "Please check the path of dataset"
    imagesTr = task_dir / 'imagesTr'
    labelsTr = task_dir / 'labelsTr'


    ids = [item.split('.')[0] for item in os.listdir(str(labelsTr))]  # 413
    # Dividing train and val dataset according to [0.85:0.15]
    train_ids, val_ids = train_test_split(ids, test_size=0.15, random_state=42)


    assert (len(train_ids)>0 and len(val_ids)>0) ,\
        "Please check the number of train and val datasets"

    # generate datasets for Monai dataloader dictionary version
    # Train
    train_list = []
    for id in train_ids:
        patient = {
            "id": id,
            "t2w": imagesTr / f'{id}_0000.nii.gz',
            "adc": imagesTr / f'{id}_0001.nii.gz',
            "dwi": imagesTr / f'{id}_0002.nii.gz',
            "mask": labelsTr / f'{id}.nii.gz',
        }

        assert all(path.exists() for path in [patient["t2w"], patient["adc"], patient["dwi"]]), \
            "Please check the path of t2w, adc, dwi"
        train_list.append(patient)

    # Val
    val_list = []
    for id in val_ids:
        patient = {
            "id": str(id),
            "t2w": imagesTr / f'{id}_0000.nii.gz',
            "adc": imagesTr / f'{id}_0001.nii.gz',
            "dwi": imagesTr / f'{id}_0002.nii.gz',
            "mask": labelsTr / f'{id}.nii.gz',
        }
        assert all(path.exists() for path in [patient["t2w"], patient["adc"], patient["dwi"]]), \
            "Please check the path of t2w, adc, dwi"
        val_list.append(patient)

    # Train
    train_dataset = Dataset(data=train_list, transform=Compose([
        lambda data: {  # -- 1
            'id': data["id"],
            "t2w": sitk.GetArrayFromImage(sitk.ReadImage(data["t2w"]))[None, : , :, :],
            "adc": sitk.GetArrayFromImage(sitk.ReadImage(data["adc"]))[None, : , :, :],
            "dwi": sitk.GetArrayFromImage(sitk.ReadImage(data["dwi"]))[None, : , :, :],
            "mask": sitk.GetArrayFromImage(sitk.ReadImage(data["mask"]))[None, : , :, :],
        },
        ToTensord(keys=["t2w", "adc", "dwi", "mask", "id"]),  # -- 2
        TorchIOCropOrPadTransform(),
        EnsureTyped(keys=["t2w", "adc", "dwi"], dtype=torch.float),
        EnsureTyped(keys=["mask", "id"], dtype=torch.long),
        # ConcatItemsd(keys=["t2w", "adc", "dwi"], name="image", dim=0),
        ClipIntensityPercentilesD(
            keys=["t2w", "adc", "dwi"],
            lower=0.5, upper=99.5,  # 0.5–99.5 percentiles
            sharpness_factor=None,
            channel_wise=False
        ),
        NormalizeIntensityd(keys=["t2w", "adc", "dwi"], nonzero=True, channel_wise=False),
    ]))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Val
    val_dataset = Dataset(data=val_list, transform=Compose([
        lambda data: {  # -- 1
            'id': data["id"],
            "t2w": sitk.GetArrayFromImage(sitk.ReadImage(data["t2w"]))[None, :, :, :],
            "adc": sitk.GetArrayFromImage(sitk.ReadImage(data["adc"]))[None, :, :, :],
            "dwi": sitk.GetArrayFromImage(sitk.ReadImage(data["dwi"]))[None, :, :, :],
            "mask": sitk.GetArrayFromImage(sitk.ReadImage(data["mask"]))[None, :, :, :],
        },
        ToTensord(keys=["t2w", "adc", "dwi", "mask", "id"]),  # -- 2
        TorchIOCropOrPadTransform(),
        EnsureTyped(keys=["t2w", "adc", "dwi"], dtype=torch.float),
        EnsureTyped(keys=["mask", "id"], dtype=torch.long),
        # ConcatItemsd(keys=["t2w", "adc", "dwi"], name="image", dim=0),
        ClipIntensityPercentilesD(
            keys=["t2w", "adc", "dwi"],
            lower=0.5, upper=99.5,  # 0.5–99.5 percentiles
            sharpness_factor=None,
            channel_wise=False
        ),
        NormalizeIntensityd(keys=["t2w", "adc", "dwi"], nonzero=True, channel_wise=True)
    ]))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return train_dataloader, val_dataloader


class BaseTrainer:
    def  __init__(self, model, train_dataloaders, val_dataloader, args):

        self.model = model
        self.dataloaders = train_dataloaders
        self.val_dataloader = val_dataloader
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0

        self.best_val_dice = 0.0     # New ：Best Val DIce
        self.best_val_loss = np.inf  # New：Best Val

        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []

        self.val_dices = []  # New：Val dice record

        self.ious = []
        self.val_interval = getattr(args, 'val_interval', 1)  # New : val interval parameter
        self.seg_loss = None
        self.optimizer = None
        self.lr_scheduler = None
        ######   Initializing Method  ######
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()

        # if (args.resume):
        #     self.init_checkpoint(
        #         join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        # else:
        #     self.init_checkpoint(self.args.checkpoint)
        self.start_epoch = 0

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def set_optimizer(self):
        sam_model = self.model
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam_model.parameters()),
                                           lr=self.args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=self.args.weight_decay
                                           )


    def get_dice_score(self, prev_masks, gt3D):

        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()


    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "dices": self.dices,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
                "used_datas": img_datas,
            }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))



    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)


    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks,
                                   size=gt3D.shape[-3:],
                                   mode='trilinear',
                                   align_corners=False)
        return low_res_masks, prev_masks


    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)  # 初始化为全零掩码
        low_res_masks = F.interpolate(prev_masks.float(),
                                      size=(args.img_size // 4, args.img_size // 4,
                                            args.img_size // 4))  # 下采样至1/4大小
        random_insert = np.random.randint(2, 9)  # 在2~8次点击中随机选一次, 作用：强制模型在随机某一步不依赖用户点击（提升鲁棒性）。
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model,
                                                               image_embedding,
                                                               gt3D,
                                                               low_res_masks,
                                                               points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model,
                                                               image_embedding,
                                                               gt3D,
                                                               low_res_masks,
                                                               points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss


    @torch.no_grad()
    def val_epoch(self, epoch, num_clicks=1):
        # infer one by one , batch size is 1 ! !
        self.model.eval()
        sam_model = self.model

        epoch_dice = 0.

        tbar = tqdm(self.val_dataloader, desc=f'VAL {epoch}')

        for data3D in tbar:
            # ─── 数据取出 ───────────────────────────────
            t2w = data3D["t2w"].to(device)
            adc = data3D["adc"].to(device)
            dwi = data3D["dwi"].to(device)
            gt3D = data3D["mask"].to(device)

            if gt3D is not None and (gt3D == 0).all() and num_clicks > 0:
                print("Warning: roi_gt is empty. Prediction will be empty.")
                return np.zeros_like(t2w.cpu().numpy()), None  # Return None for low_res_mask

            with torch.amp.autocast("cuda"):
                # ─── 前向提取特征 ──────────────────────
                t2w_embed = sam_model.image_encoder(t2w)
                adc_embed = sam_model.image_encoder(adc)
                dwi_embed = sam_model.image_encoder(dwi)
                image_embeddings = sam_model.feature_fusion(t2w_embed, adc_embed, dwi_embed)

                points_coords, points_labels = torch.zeros(1, 0, 3).to(device), torch.zeros(1, 0).to(device)

                # Start with empty prev_mask for click [1, 1, 128, 128, 128] --> [1, 128, 128, 128]
                current_prev_mask_for_click_generation = torch.zeros_like(t2w, device=device)[:, 0, ...]
                prev_low_res_mask = torch.zeros(1, 1, # similar to current_prev_mask_for_click_generation // 4
                                                t2w.shape[2] // 4,  # [1, 1, 32, 32, 32]
                                                t2w.shape[3] // 4,
                                                t2w.shape[4] // 4, device=device, dtype=torch.float)

                for _ in range(num_clicks):
                    new_points_co, new_points_la = random_sample_next_click(
                        current_prev_mask_for_click_generation.squeeze(0).cpu(),  # Expects HWD tensor
                        gt3D[0, 0].cpu()  # Expects HWD tensor
                    )
                    new_points_co, new_points_la = new_points_co.to(device), new_points_la.to(device)
                    points_coords = torch.cat([points_coords, new_points_co], dim=1)
                    points_labels = torch.cat([points_labels, new_points_la], dim=1)

                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=[points_coords, points_labels],
                        boxes=None,
                        masks=prev_low_res_mask,
                    )

                    # 10. 掩膜解码和更新低分辨率掩膜
                    low_res_masks, _ = sam_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    # Update prev_low_res_mask for next iteration's prompt encoder input
                    prev_low_res_mask = low_res_masks.detach()

                    # For click generation, use the upscaled version of the current prediction
                    current_prev_mask_for_click_generation = F.interpolate(low_res_masks,
                                                                           size=t2w.shape[-3:],
                                                                           mode='trilinear',
                                                                           align_corners=False)
                    current_prev_mask_for_click_generation = torch.sigmoid(current_prev_mask_for_click_generation) > 0.5

                    # 11. 生成最终的高分辨率掩膜
                    # Final high-resolution mask from the last low_res_masks
                final_masks_hr = F.interpolate(low_res_masks,  # Use the final low_res_masks
                                               size=t2w.shape[-3:],
                                               mode='trilinear',
                                               align_corners=False)

                medsam_seg_prob = torch.sigmoid(final_masks_hr)
                medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
                medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)  # (128, 128, 128)
                new_mask = np.expand_dims(np.expand_dims(medsam_seg_mask, axis=0), axis=0)

                epoch_dice += self.get_dice_score(new_mask, gt3D)

        print(f'Val EPOCH: {epoch},  Val Dice: {epoch_dice / len(self.val_dataloader):.4f}')
        logger.info(f'Val Epoch\t {epoch}\t :  val dice: {epoch_dice / len(self.val_dataloader):.4f}')
        return round(epoch_dice / len(self.val_dataloader), 4)




    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        step_loss = 0
        epoch_dice = 0

        self.model.train()
        sam_model = self.model
        self.args.rank = -1
        tbar = tqdm(self.dataloaders)
        self.optimizer.zero_grad()

        for step, data3D in enumerate(tbar):
            # tbar.set_description(f"ID: {data3D['id'][0]}")
            t2w = data3D["t2w"].to(device)  # [B, 1, 128, 128, 128]
            adc = data3D["adc"].to(device)
            dwi = data3D["dwi"].to(device)
            gt3D = data3D["mask"].to(device)

            with torch.amp.autocast("cuda"):
                t2w_image_embeddings = sam_model.image_encoder(t2w)  # [2, 384, 8, 8, 8]
                adc_image_embeddings = sam_model.image_encoder(adc)
                dwi_image_embeddings = sam_model.image_encoder(dwi)

                image_embedding = sam_model.feature_fusion(t2w_image_embeddings,
                                                            adc_image_embeddings,
                                                            dwi_image_embeddings)

                self.click_points = []
                self.click_labels = []

                pred_list = []

                prev_masks, loss = self.interaction(sam_model,
                                                    image_embedding,
                                                    gt3D,
                                                    num_clicks=11)

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks, gt3D)
                cur_loss = loss.item()

                # ------- LoRA orthogonal regularization
                if self.args.mode == 'sam_adalora':
                    ortho_reg = lora.compute_orth_regu(sam_model, regu_weight=0.1)
                    loss = loss + ortho_reg

                loss /= self.args.accumulation_steps

                self.scaler.scale(loss).backward()

                if step % self.args.accumulation_steps == 0 and step != 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.args.mode == 'sam_adalora':
                        self.rankallocator.update_and_mask(self.model.image_encoder, global_step=step)

                    self.optimizer.zero_grad()

                    print_loss = step_loss / self.args.accumulation_steps
                    step_loss = 0
                    # print(f'prev_masks shape is {prev_masks.shape}, gt3D shape is {gt3D.shape}')
                    print_dice = self.get_dice_score(prev_masks, gt3D)
                else:
                    step_loss += cur_loss

                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.5:
                            self.save_checkpoint(epoch,
                                                 sam_model.state_dict(),
                                                 describe=f'{epoch}_step_dice:{print_dice}_best')
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss


        epoch_loss /= step + 1
        epoch_dice /= step + 1

        return epoch_loss, epoch_iou, epoch_dice, pred_list


    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()

    def plot_result_mix(self, plot_data1, plot_data2, label1, label2, description, save_name):
        plt.plot(plot_data1, label=label1)
        plt.plot(plot_data2, label=label2)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.legend(loc='best')
        plt.savefig(join(str(MODEL_SAVE_PATH), f'{save_name}.png'))
        plt.close()


    def train(self):
        self.scaler = torch.amp.GradScaler("cuda")
        if self.args.mode == 'default':
            # Frozen SAM-Med3D parameters
            for p in self.model.parameters():
                p.requires_grad = False

            # train encoder weights
            for p in self.model.image_encoder.parameters():
                p.requires_grad = True

            for p in self.model.feature_fusion.parameters():
                p.requires_grad = True
        elif self.args.mode == 'sam_adapter':
            for n, value in self.model.named_parameters():
                if 'Adapter' not in n:
                    value.requires_grad = False
                else:
                    value.requires_grad = True
        elif self.args.mode == 'sam_adalora':

            lora.mark_only_lora_as_trainable(self.model.image_encoder)
            # Initialize the RankAllocator
            self.rankallocator = lora.RankAllocator(
                self.model.image_encoder,
                lora_r=4,
                target_rank=8,
                init_warmup=500,
                final_warmup=1500,
                mask_interval=10,
                total_step=self.args.num_epochs * len(self.dataloaders),
                beta1=0.85,
                beta2=0.85
            )
        else:
            for n, value in self.model.named_parameters():
                value.requires_grad = True

        model_total_params = sum(p.numel() for p in self.model.parameters())
        model_grad_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(
            "Total params: {0}M\t Gradient Parameters: {1}M".format(model_total_params / 1e6,
                                                                    model_grad_params / 1e6))

        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')


            ################### Train #################
            num_clicks = np.random.randint(1, 21)  # 实际未参与train
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)

            self.lr_scheduler.step()
            self.losses.append(epoch_loss)
            self.dices.append(epoch_dice)

            print(f'EPOCH: {epoch}, Loss: {epoch_loss}  Dice: {epoch_dice}')
            logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')

            # New: Validating val dataset at {parameter} interval
            if (epoch + 1) % self.val_interval == 0:
                print_val_dice = self.val_epoch(epoch, num_clicks=num_clicks)
                self.val_dices.append(print_val_dice)


            state_dict = self.model.state_dict()

            # save latest checkpoint
            self.save_checkpoint(epoch, state_dict, describe='latest')

            # save train loss best checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint(epoch, state_dict, describe='loss_best')

            # save train dice best checkpoint
            if epoch_dice > self.best_dice:
                self.best_dice = epoch_dice
                self.save_checkpoint(epoch, state_dict, describe='dice_best')

            self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
            self.plot_result_mix(self.dices, self.val_dices, 'Train_Dice', 'Val_Dice',
                                 'Total Dice', 'Dice')
            # self.plot_result(self.dices, 'Dice', 'Dice')
            # self.plot_result(self.val_dices, 'Val Dice', 'Val_Dice')

        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')

def main():
    # 打印args里面的参数
    # pprint(vars(args))

    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)

    # Load datasets
    train_dataloader, val_dataloader = get_dataloaders(args)
    print("Dataloader is Comming")

    # Build model
    ckpt_path = "/home/lib/PycharmProjects/SAM-Med3D/ckpt/sam_med3d_turbo.pth"
    net = get_network(args=args, use_gpu=True)

    print("Using", torch.cuda.device_count(), "GPUs!")
    print(" ============================================= ")

    model = net.to(device)

    trainer = BaseTrainer(model, train_dataloader, val_dataloader, args)

    # Train
    trainer.train()

if __name__ == '__main__':
    main()