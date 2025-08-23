# Phase 2：Teacher & Student Policies with loss logging & plotting

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from Dataset import VideoFrameActionDataset
from Models.VectorQuantizer import VectorQuantizerEMA
from Models.InverseDynamicsModel import InverseDynamicsModel
from Models.FrameEncoder import ImpalaCNN
from Models.TeacherPolicy import TeacherPolicy
from Models.StudentPolicy import StudentPolicy

def train():

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    lr_t       = 1e-4
    lr_s       = 1e-4
    epochs_t   = 20
    epochs_s   = 5
    save_dir = "C:\\Users\\44753\\Desktop\\SavedModels"

    # —— Load and frozen models trained in phase 1 —— 
    encoder = ImpalaCNN(embedding_dim=128, channel_multiplier=4).to(device).eval()
    idm     = InverseDynamicsModel(state_dim=128, latent_dim=32).to(device).eval()
    vq      = VectorQuantizerEMA(num_embeddings=64, embedding_dim=32, commitment_cost=0.05, decay=0.999).to(device).eval()

    encoder.load_state_dict(torch.load('C://Users//44753//Desktop//SavedModels//encoder.pth', map_location=device))
    idm.load_state_dict(torch.load('C://Users//44753//Desktop//SavedModels//idm.pth', map_location=device))
    vq.load_state_dict(torch.load('C://Users//44753//Desktop//SavedModels//vq.pth', map_location=device))

    for p in encoder.parameters(): p.requires_grad = False
    for p in idm.parameters():     p.requires_grad = False
    for p in vq.parameters():      p.requires_grad = False

    # —— TeacherPolicy lam —— 
    teacher = TeacherPolicy(encoder).to(device).train()
    for name, p in teacher.named_parameters():
        if not name.startswith('lam'):
            p.requires_grad = False

    # —— StudentPolicy lam —— 
    student = StudentPolicy(encoder).to(device).train()
    for name, p in student.named_parameters():
        if not name.startswith('lam'):
            p.requires_grad = False

    # —— Load Data —— 
    dataset    = VideoFrameActionDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Arrays recording losses
    teacher_losses = []
    student_losses = []

    # 1) train TeacherPolicy.lam
    optimizer_t = torch.optim.Adam(teacher.lam.parameters(), lr=lr_t)
    for epoch in range(1, epochs_t+1):
        running_loss = 0.0
        for (img1, img2), _, _ in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            with torch.no_grad():
                s1    = encoder(img1)
                s2    = encoder(img2)
                z_hat = idm(s1, s2)
                z_q, _ = vq(z_hat)
            # Teacher predict latent action
            z_pred, _, _ = teacher(img1,img2)
            # MSE
            loss_t = torch.nn.functional.mse_loss(z_pred, z_q)
            optimizer_t.zero_grad()
            loss_t.backward()
            optimizer_t.step()
            running_loss += loss_t.item()

        avg_loss = running_loss / len(dataloader)
        teacher_losses.append(avg_loss)
        print(f"Teacher Epoch {epoch}/{epochs_t}, Loss: {avg_loss:.6f}")

    # Save Model
    torch.save(teacher.state_dict(), os.path.join(save_dir, "teacher.pth"))

    # 2) train StudentPolicy.lam
    optimizer_s = torch.optim.Adam(student.lam.parameters(), lr=lr_s)
    for epoch in range(1, epochs_s+1):
        running_loss = 0.0
        for (img1, img2), _, _ in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)

            with torch.no_grad():
                z_teacher, _, _ = teacher(img1,img2)
            # Student predict
            z_student = student(img1)
            # Distill Loss
            loss_s = torch.nn.functional.mse_loss(z_student, z_teacher)
            optimizer_s.zero_grad()
            loss_s.backward()
            optimizer_s.step()
            running_loss += loss_s.item()

        avg_loss = running_loss / len(dataloader)
        student_losses.append(avg_loss)
        print(f"Student Epoch {epoch}/{epochs_s}, Loss: {avg_loss:.6f}")

    # Save Model
    torch.save(student.state_dict(), os.path.join(save_dir, "student.pth"))

     # —— Plotting —— 
    # Teacher Plotting
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs_t+1), teacher_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Teacher Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('teacher_training_loss.png')
    plt.close()
    print("Saved teacher loss plot to 'teacher_training_loss.png'")

    # Student Plotting
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs_s+1), student_losses, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Student Distillation Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('student_distill_loss.png')
    plt.close()
    print("Saved student loss plot to 'student_distill_loss.png'")

if __name__ == '__main__':
    train()