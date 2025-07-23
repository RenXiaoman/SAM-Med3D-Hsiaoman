from pathlib import Path
import SimpleITK as sitk
from .reg_lib import register_spline           # 你已有的 B-Spline 配准函数

def get_gradient_features(image: sitk.Image) -> sitk.Image:
    """Return the average gradient of the 3D image in the (x, y, z) direction。"""
    grad = sitk.GradientImageFilter().Execute(image)
    gx = sitk.VectorIndexSelectionCast(grad, 0)
    gy = sitk.VectorIndexSelectionCast(grad, 1)
    gz = sitk.VectorIndexSelectionCast(grad, 2)
    return 0.33 * (gx + gy + gz)


def register_image_pair(
    fixed_img: sitk.Image | str | Path,
    moving_img: sitk.Image | str | Path,
    *,
    to_float32: bool = True,
    verbose: bool = False,
) -> tuple[sitk.Image, sitk.Transform, float]:
    """
    使用 B-Spline (register_spline) 将 moving 配准到 fixed。

    Parameters
    ----------
    fixed_img, moving_img : `SimpleITK.Image` or path-like
        固定影像 / 待配准影像 (NIfTI、MHA 等)。
    to_float32 : bool
        若为 True, 则强制将读入影像转换为 sitkFloat32。

    Returns
    -------
    registered_img : `SimpleITK.Image`
        按 transform 重采样后的 moving 影像 (在 fixed 空间)。
    transform : `SimpleITK.Transform`
        由 B-Spline 优化得到的变换，可持久化保存。
    metric : float
        最终 Mattes Mutual Information 指标值。
    """
    # ---------- Read / Check Type ----------
    if isinstance(fixed_img, (str, Path)):
        fixed_img = sitk.ReadImage(str(fixed_img), sitk.sitkFloat32 if to_float32 else None)
    if isinstance(moving_img, (str, Path)):
        moving_img = sitk.ReadImage(str(moving_img), sitk.sitkFloat32 if to_float32 else None)

     # if the input is a 'Image' rather than a path, explicit cast can also be performed
    if to_float32: 
        fixed_img  = sitk.Cast(fixed_img,  sitk.sitkFloat32)
        moving_img = sitk.Cast(moving_img, sitk.sitkFloat32)

    # ---------- 梯度特征 ----------
    fixed_grad   = get_gradient_features(fixed_img)
    moving_grad  = -get_gradient_features(moving_img)

    # ---------- B-Spline 配准 ----------
    transform, metric = register_spline(
        fixed_grad,
        moving_grad,
        verbose=verbose,
    )

    # ---------- 重采样 ----------
    registered_img = sitk.Resample(
        moving_img,            # 要重采样的影像
        fixed_img,             # 目标空间
        transform,             # 计算得到的变换
        sitk.sitkLinear,       # 插值方式
        0.0,                   # 默认空洞填充值
        # moving_img.GetPixelID()
        sitk.sitkFloat32
    )

    return registered_img, transform, metric


# if __name__ == "__main__":
#     OUT_PUT_DIR = Path("./register")      # 或 Path("src/register")
#     OUT_PUT_DIR.mkdir(parents=True, exist_ok=True)


#     # Example usage
#     fixed_image_path = "data/Task205_picai_lesion/imagesTr/10005_1000005_0000.nii.gz"
#     moving_image_path = "data/Task205_picai_lesion/imagesTr/10005_1000005_0001.nii.gz"
    
#     registered_image, transform, metric = register_image_pair(
#         fixed_image_path,
#         moving_image_path,
#         to_float32=True,
#         verbose=True
#     )
    
#     print(f"Registered Image: {registered_image}")
#     print(f"Transform: {transform}")
#     print(f"Metric: {metric}")

#     out_path = OUT_PUT_DIR / f"{Path(fixed_image_path).stem}_registered.nii.gz"
#     sitk.WriteImage(registered_image, str(out_path))   # 建议显式转成 str