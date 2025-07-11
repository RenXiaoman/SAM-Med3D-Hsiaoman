import torch
from segment_anything import sam_model_registry3D

def get_network(args, use_gpu :bool=True):
    sam_med3d_model = sam_model_registry3D['vit_b_ori'](args=args, checkpoint=None).to('cuda:0')
    if use_gpu:
        sam_med3d_model = sam_med3d_model.to(args.device)
    return sam_med3d_model