from Dataset import VideoFrameActionDataset
from torch.utils.data import DataLoader
import torch
from Models.FrameEncoder import ImpalaCNN
from Models.InverseDynamicsModel import InverseDynamicsModel
from Models.VectorQuantizer import VectorQuantizerEMA
from Models.ForwardDynanmicsModel import UNetFDM
from Models.ActionEncoder import ActionDecoder

import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

# matplotlib unblock
plt.ion()


def plot_losses(loss_array, title="Training Loss", ylabel="Loss", xlabel="Epoch",save_dir=""):
    if isinstance(loss_array, torch.Tensor):
        loss_array = loss_array.detach().cpu().numpy()
    elif isinstance(loss_array, list) and isinstance(loss_array[0], torch.Tensor):
        loss_array = [x.detach().cpu().item() for x in loss_array]
    plt.figure(figsize=(10, 5))
    plt.plot(loss_array, label="Loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_dir:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir)
    plt.show(block=False)
    plt.pause(0.001)
    plt.close()


def show_images(tensor_batch, title):
    img_grid = vutils.make_grid(tensor_batch[:8].cpu(), nrow=4, normalize=True)
    plt.figure(figsize=(8, 4))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)
    plt.close()


if __name__ == '__main__':
    # Configurations
    #video_folder_path = "C:\\Users\\44753\\Desktop\\Videostest"
    #csv_folder_path   = "C:\\Users\\44753\\Desktop\\Maintest"
    save_dir          = "C:\\Users\\44753\\Desktop\\SavedModels"
    os.makedirs(save_dir, exist_ok=True)

    save_freq = 5         # save every 5 epoch
    patience = 3          # early stop patience

    # Data loader
    dataset     = VideoFrameActionDataset(train=True, transform=None)
    dataloader  = DataLoader(dataset, batch_size=64, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    encoder        = ImpalaCNN(embedding_dim=128, channel_multiplier=4).to(device)
    idm            = InverseDynamicsModel(state_dim=128, latent_dim=32).to(device)
    vq             = VectorQuantizerEMA(num_embeddings=64, embedding_dim=32, commitment_cost=0.05, decay=0.999).to(device)
    fdm            = UNetFDM(latent_dim=32).to(device)
    action_decoder = ActionDecoder(latent_dim=32).to(device)

    # VGG perceptual loss
    vgg_features = models.vgg19(pretrained=True).features[:16].eval().to(device)
    for p in vgg_features.parameters(): p.requires_grad = False
    def VGGLoss(x, y):
        return F.mse_loss(vgg_features(x), vgg_features(y))

    # Optimizer
    opt_action = optim.Adam(
        list(encoder.parameters()) +
        list(idm.parameters()) +
        list(vq.parameters()) + 
        list(action_decoder.parameters()),
        lr=1e-4
    )

    opt_fdm = optim.Adam(
        list(fdm.parameters()),
        lr=1e-4
    )

    num_epochs = 50
    best_loss = float('inf')
    epochs_no_improve = 0

    print("----------------Start Training-----------------")

    average_recon_loss = []
    average_vq_loss    = []
    average_stick_loss = []
    average_button_loss= []
    average_total_loss = []

    for epoch in range(num_epochs):
        total_loss = total_recon_loss = total_vq_loss = total_stick_loss = total_button_loss = 0
        total_steps = 0

        for (s1_img, s2_img), stick_action, button_action in dataloader:
            s1_img, s2_img = s1_img.to(device), s2_img.to(device)
            stick_action   = stick_action.to(device)
            button_action  = button_action.to(device).float()

            # Forward
            e1 = encoder(s1_img)
            e2 = encoder(s2_img)
            z  = idm(e1, e2)
            z_q, vq_loss = vq(z)
            s2_pred = fdm(s1_img, z_q)
            stick_pred, button_logits = action_decoder(z_q)

            # Losses
            recon_loss  = VGGLoss(s2_pred, s2_img)
            stick_loss  = F.mse_loss(stick_pred, stick_action)
            button_logits = button_logits.squeeze()
            button_loss   = F.binary_cross_entropy_with_logits(button_logits, button_action)
            loss = recon_loss + vq_loss + stick_loss + button_loss

            # Backward
            opt_action.zero_grad()
            opt_fdm.zero_grad()
            loss.backward()
            opt_action.step()
            opt_fdm.step()

            # Metrics
            total_recon_loss += recon_loss.item()
            total_vq_loss    += vq_loss.item()
            total_stick_loss += stick_loss.item()
            total_button_loss+= button_loss.item()
            total_loss       += loss.item()
            total_steps      += 1

            # Optional visualization
            if total_steps % 500 == 0:
                show_images(s2_img, "Ground Truth: s2_img")
                show_images(s2_pred, "Reconstructed: s2_pred")

        # Epoch statistics
        avg_loss = total_loss / total_steps
        average_recon_loss.append(total_recon_loss / total_steps)
        average_vq_loss.append(total_vq_loss / total_steps)
        average_stick_loss.append(total_stick_loss / total_steps)
        average_button_loss.append(total_button_loss / total_steps)
        average_total_loss.append(avg_loss)

        #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Total: {avg_loss:.4f} | "
            f"Recon: {average_recon_loss[-1]:.4f} | "
            f"VQ: {average_vq_loss[-1]:.4f} | "
            f"Stick: {average_stick_loss[-1]:.4f} | "
            f"Button: {average_button_loss[-1]:.4f}"
        )
        plot_losses(average_recon_loss, title="Average Recon Loss", save_dir=os.path.join(save_dir, "average_recon_loss.png"))
        plot_losses(average_vq_loss, title="Average VQ Loss", save_dir=os.path.join(save_dir, "average_vq_loss.png"))
        plot_losses(average_stick_loss, title="Average Stick Loss", save_dir=os.path.join(save_dir, "average_stick_loss.png"))
        plot_losses(average_button_loss, title="Average Button Loss", save_dir=os.path.join(save_dir, "average_button_loss.png"))
        plot_losses(average_total_loss, title="Average Total Loss", save_dir=os.path.join(save_dir, "average_total_loss.png"))

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered. Stopping training.")
                break

        # Auto-save every `save_freq` epochs
        if (epoch + 1) % save_freq == 0:
            print(f"-- Saving models at epoch {epoch+1} --")
            torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
            torch.save(idm.state_dict(),     os.path.join(save_dir, "idm.pth"))
            torch.save(vq.state_dict(),      os.path.join(save_dir, "vq.pth"))
            torch.save(fdm.state_dict(),     os.path.join(save_dir, "fdm.pth"))
            torch.save(action_decoder.state_dict(), os.path.join(save_dir, "action_decoder.pth"))

    plt.ioff()
    plot_losses(average_recon_loss, title="Average Recon Loss", save_dir=os.path.join(save_dir, "average_recon_loss.png"))
    plot_losses(average_vq_loss, title="Average VQ Loss", save_dir=os.path.join(save_dir, "average_vq_loss.png"))
    plot_losses(average_stick_loss, title="Average Stick Loss", save_dir=os.path.join(save_dir, "average_stick_loss.png"))
    plot_losses(average_button_loss, title="Average Button Loss", save_dir=os.path.join(save_dir, "average_button_loss.png"))
    plot_losses(average_total_loss, title="Average Total Loss", save_dir=os.path.join(save_dir, "average_total_loss.png"))

    print("----------------Training Done-----------------")

    # Final save (overwrite latest)
    torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
    torch.save(idm.state_dict(),     os.path.join(save_dir, "idm.pth"))
    torch.save(vq.state_dict(),      os.path.join(save_dir, "vq.pth"))
    torch.save(fdm.state_dict(),     os.path.join(save_dir, "fdm.pth"))
    torch.save(action_decoder.state_dict(), os.path.join(save_dir, "action_decoder.pth"))
    print(f"Models saved to {save_dir}")


    plt.ioff()
    plot_losses(average_recon_loss, title="Average Recon Loss", save_dir=os.path.join(save_dir, "average_recon_loss.png"))
    plot_losses(average_vq_loss, title="Average VQ Loss", save_dir=os.path.join(save_dir, "average_vq_loss.png"))
    plot_losses(average_stick_loss, title="Average Stick Loss", save_dir=os.path.join(save_dir, "average_stick_loss.png"))
    plot_losses(average_button_loss, title="Average Button Loss", save_dir=os.path.join(save_dir, "average_button_loss.png"))
    plot_losses(average_total_loss, title="Average Total Loss", save_dir=os.path.join(save_dir, "average_total_loss.png"))
