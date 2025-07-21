import torch

from segment_anything.Ours.common.fusion import SEAttention3D

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SEAttention3D()
    model.init_weights()
    model.to(device)

    Batch_size = 1
    input = torch.randn(Batch_size, 384, 8, 8, 8).to(device)   # [B, C, D, H, W]

    output = model(input, input, input)
    print(output.shape)
