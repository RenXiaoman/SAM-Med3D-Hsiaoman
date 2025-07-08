import argparse

from segment_anything.Ours.Coarse import SAM_Coarse_Seg
import torch
import torch.nn as nn


# #######    Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='ckpt/sam_med3d.pth')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
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
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()
device = args.device


def model_train(args, net, train_loader, valid_loader):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=args.lr)  # weight_decay=0.0004)
    step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

def main():
    net = SAM_Coarse_Seg(ckpt_path="/home/lib/PycharmProjects/SAM-Med3D/ckpt/sam_med3d_turbo.pth")

    model_total_params = sum(p.numel() for p in net.parameters())
    model_grad_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("Total params: {0}M\t Gradient Parameters: {1}M".format(model_total_params / 1e5, model_grad_params / 1e6))
    print("Using", torch.cuda.device_count(), "GPUs!")

    net = net.to(device)

    input_tensor = torch.randn(1, 3, 128, 128, 128).to(device)
    y = None
    net(input_tensor, y)

if __name__ == '__main__':
    main()