import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from Dataset import VideoFrameActionDataset
from Models.FrameEncoder import ImpalaCNN
from Models.InverseDynamicsModel import InverseDynamicsModel
from Models.VectorQuantizer import VectorQuantizerEMA
from Models.ForwardDynanmicsModel import UNetFDM
from Models.ActionEncoder import ActionDecoder
import torchvision.models as models


def VGGLoss(x, y, device):
    vgg = models.vgg19(pretrained=True).features[:16].eval().to(device)
    for p in vgg.parameters(): p.requires_grad = False
    return F.mse_loss(vgg(x), vgg(y))


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Models
    encoder = ImpalaCNN(embedding_dim=128, channel_multiplier=4).to(device)
    idm = InverseDynamicsModel(state_dim=128, latent_dim=32).to(device)
    vq = VectorQuantizerEMA(num_embeddings=64, embedding_dim=32,
                             commitment_cost=args.vq_commitment, decay=0.999).to(device)
    fdm = UNetFDM(latent_dim=32).to(device)
    action_decoder = ActionDecoder(latent_dim=32).to(device)

    # Load Pre-Trained Parameters
    encoder.load_state_dict(torch.load(os.path.join(args.model_dir, 'encoder.pth'), map_location=device))
    idm.load_state_dict(torch.load(os.path.join(args.model_dir, 'idm.pth'),     map_location=device))
    vq.load_state_dict(torch.load(os.path.join(args.model_dir, 'vq.pth'),       map_location=device))
    fdm.load_state_dict(torch.load(os.path.join(args.model_dir, 'fdm.pth'),     map_location=device))
    action_decoder.load_state_dict(torch.load(os.path.join(args.model_dir, 'action_decoder.pth'),
                                             map_location=device))

    # Evaluation Mode
    encoder.eval(); idm.eval(); vq.eval(); fdm.eval(); action_decoder.eval()

    # Load Data
    dataset = VideoFrameActionDataset(train=False, transform=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Losses
    total_recon, total_vq, total_stick, total_button, total_steps = 0,0,0,0,0

    with torch.no_grad():
        for i, ((s1, s2), stick_gt, button_gt) in enumerate(loader):
            s1 = s1.to(device); s2 = s2.to(device)
            stick_gt = stick_gt.to(device)
            button_gt = button_gt.to(device).float()

            e1 = encoder(s1)
            e2 = encoder(s2)
            z = idm(e1, e2)
            z_q, vq_loss = vq(z)
            s2_pred = fdm(s1, z_q)
            stick_pred, button_logits = action_decoder(z_q)

            recon = VGGLoss(s2_pred, s2, device).item()
            vq_l = vq_loss.item()
            stick_l = F.mse_loss(stick_pred, stick_gt).item()
            button_l = F.binary_cross_entropy_with_logits(button_logits.squeeze(), button_gt).item()

            total_recon += recon
            total_vq += vq_l
            total_stick += stick_l
            total_button += button_l
            total_steps += 1

            # 保存第一 batch 的可视化
            if i == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                grid_gt = vutils.make_grid(s2.cpu(), nrow=4, normalize=True)
                grid_pd = vutils.make_grid(s2_pred.cpu(), nrow=4, normalize=True)
                plt.imsave(os.path.join(args.output_dir, 'gt_grid.png'), grid_gt.permute(1,2,0).numpy())
                plt.imsave(os.path.join(args.output_dir, 'pred_grid.png'), grid_pd.permute(1,2,0).numpy())


    print("==== Test Results ====")
    print(f"Recon Loss: {total_recon/total_steps:.4f}")
    print(f"VQ Loss:    {total_vq/total_steps:.4f}")
    print(f"Stick Loss: {total_stick/total_steps:.4f}")
    print(f"Button Loss:{total_button/total_steps:.4f}")
    print(f"Results images saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Latent Action Model')
    parser.add_argument('--model_dir', type=str,
                        help='保存模型的目录路径', default= "C://Users//44753//Desktop//SavedModels")
    parser.add_argument('--output_dir', type=str, default='C://Users//44753//Desktop//EvalResults',
                        help='测试输出（图像和日志）保存目录')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='测试时的 batch size')
    parser.add_argument('--vq_commitment', type=float, default=0.05,
                        help='VQ commitment weight')
    args = parser.parse_args()
    test(args)
