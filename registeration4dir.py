from pathlib import Path
from typing import List, Tuple
import SimpleITK as sitk
from tqdm import tqdm
from utils.registeration_utils import register_image_pair
import shutil

import multiprocessing as mp
from tqdm.contrib.concurrent import process_map


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def collect_case_ids(task_dir: Path, split: str = "imagesTr") -> List[str]:
    """
    收集 nnUNet 文件名前两段组成的 case-id 列表。
    例:10005_1000005_0000.nii.gz  →  10005_1000005

    Parameters
    ----------
    task_dir : 任务根目录，如 ./data/Task205_picai_lesion
    split    : 子目录名(imagesTr / imagesTs / imagesVal …）

    Returns
    -------
    list[str] : 去重后的 case-id 列表（顺序为字典序）
    """
    img_dir = task_dir / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} 不存在！")

    # 用集合去重，再转成列表，最后排序
    id_set = {
        "_".join(p.stem.split("_")[:2])
        for p in img_dir.iterdir()
        if p.is_file()
    }
    return sorted(id_set)


def process_case(args: Tuple[str, Path, Path]) -> str:
    cid, images_dir, out_dir = args

    # SimpleITK 在子进程里多线程可能过度；限制为 1
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    # 路径
    t2w_path = images_dir / f"{cid}_0000.nii.gz"
    adc_path = images_dir / f"{cid}_0001.nii.gz"
    dwi_path = images_dir / f"{cid}_0002.nii.gz"

    # ---------- ADC 注册 ----------
    adc_reg_img, _, _ = register_image_pair(
        fixed_img=t2w_path,
        moving_img=adc_path,
        to_float32=True,
        verbose=False,
    )

    # ---------- DWI 注册 ----------
    dwi_reg_img, _, _ = register_image_pair(
        fixed_img=t2w_path,
        moving_img=dwi_path,
        to_float32=True,
        verbose=False,
    )

    # ---------- 保存结果 ----------
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) T2W 直接复制
    shutil.copy2(t2w_path, out_dir / t2w_path.name)
    # 2) 写配准后的影像
    sitk.WriteImage(adc_reg_img, str(out_dir / adc_path.name))
    sitk.WriteImage(dwi_reg_img, str(out_dir / dwi_path.name))

    return cid   # 返回标记供调试

# We emply registration process for nnUnet-based  dir 
# ------------------------------------------------------------------
# 主要流程
# ------------------------------------------------------------------

TASK_DIR   = Path("./data/Task205_picai_lesion")
IMAGES_DIR = TASK_DIR / "imagesTr"
OUT_DIR    = TASK_DIR / "new_imagesTr"
OUT_DIR.mkdir(exist_ok=True, parents=True)

case_ids = collect_case_ids(TASK_DIR, "imagesTr")

print(f"Total IDs: {len(case_ids)}")
print(f'CPU count {mp.cpu_count()}')

process_args = [(cid, IMAGES_DIR, OUT_DIR) for cid in case_ids]
max_workers = max(1, mp.cpu_count() // 2)

# 并行跑，附带进度条
_ = process_map(
    process_case,
    process_args,
    max_workers=max_workers,
    chunksize=1,
    desc="Registering"
)

print("All cases finished ✔️")


# for cid in tqdm(case_ids, desc="Registering"):
#     # ---- 路径准备 ----
#     t2w_path = IMAGES_DIR / f"{cid}_0000.nii.gz"   # fixed
#     adc_path = IMAGES_DIR / f"{cid}_0001.nii.gz"   # moving-1
#     dwi_path = IMAGES_DIR / f"{cid}_0002.nii.gz"   # moving-2

    

#     # ---- 2. ADC 配准 ----
#     adc_reg_img, adc_tx, adc_metric = register_image_pair(
#         fixed_img  = t2w_path,
#         moving_img = adc_path,
#         to_float32 = True,
#         verbose    = False,
#     )
    

#     # ---- 3. DWI 配准 ----
#     dwi_reg_img, dwi_tx, dwi_metric = register_image_pair(
#         fixed_img  = t2w_path,
#         moving_img = dwi_path,
#         to_float32 = True,
#         verbose    = False,
#     )
#     # ---- 1. 复制 T2W 原图到新目录 ----
#     shutil.copy2(t2w_path, NEW_DIR / t2w_path.name)
#     sitk.WriteImage(adc_reg_img, str(NEW_DIR / adc_path.name))
#     sitk.WriteImage(dwi_reg_img, str(NEW_DIR / dwi_path.name))