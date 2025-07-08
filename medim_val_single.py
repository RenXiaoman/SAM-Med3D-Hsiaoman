# -*- encoding: utf-8 -*-
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai.visualize import matshow3d

import medim

from utils.infer_utils import validate_paired_img_gt
from utils.metric_utils import compute_metrics, print_computed_metrics

if __name__ == "__main__":
    ''' 1. prepare the pre-trained model with local path or huggingface url '''
    ckpt_path = "work_dir/Task2203_picai_gland/sam_model_dice_best.pth"
    if Path(ckpt_path).exists():
        print("The model has been downloaded.")

    # or you can use a local path like:
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)

    ''' 2. read and pre-process your input data '''
    # img_path = "./test_data/amos_val_toy_data/imagesVa/amos_0013.nii.gz"
    # gt_path = "./test_data/amos_val_toy_data/labelsVa/amos_0013.nii.gz"
    # out_path = "./test_data/amos_val_toy_data/pred/amos_0013.nii.gz"

    prostate_img_path = "data/Task2203_picai_gland/imagesTr/10001_1000001.nii.gz"
    prostate_gt_path = "data/Task2203_picai_gland/labelsTr/10001_1000001.nii.gz"
    prostate_output_path = "test_prostate_data_dir/100001_gland.nii.gz"

    # 使用assert，来判断输入图像和标签的shape是否一致，不一致则报错
    assert sitk.ReadImage(prostate_img_path).GetSize() == sitk.ReadImage(prostate_gt_path).GetSize(), "The shape of input image and label is not equal!"
    # 使用assert，来判断输入图像和标签的spacing是否一致，不一致则报错

    ''' 3. infer with the pre-trained SAM-Med3D model '''
    print("Validation start! plz wait for some times.")
    # validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=1)
    validate_paired_img_gt(model, prostate_img_path, prostate_gt_path, prostate_output_path, num_clicks=1, target_spacing=(1.5, 1.5, 1.5))
    print("Validation finish! plz check your prediction.")

    ''' 4. compute the metrics of your prediction with the ground truth '''
    # metrics = compute_metrics(
    #     gt_path=gt_path,
    #     pred_path=out_path,
    #     metrics=['dice'],
    #     classes=None,q
    # )
    metrics = compute_metrics(
        gt_path=prostate_gt_path,
        pred_path=prostate_output_path,
        metrics=['dice'],
        classes=None,
    )
    print('nihao')
    print_computed_metrics(metrics)
