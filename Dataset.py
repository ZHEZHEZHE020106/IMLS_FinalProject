import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
import imageio
import numpy as np
from PIL import Image
import pandas as pd


class VideoFrameActionDataset(Dataset):
    def __init__(self, train=True, transform=None):
        interval_frames = 10
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]) if transform is None else transform

        self.train = train

        if self.train:
            self.video_folder_path = "C:\\Users\\44753\\Desktop\\Videostest"
            self.csv_folder_path = "C:\\Users\\44753\\Desktop\\Maintest"
            self.cache_dir = "C:\\Users\\44753\\Desktop\\cache"
        else:
            self.video_folder_path = "C:\\Users\\44753\\Desktop\\VideoEval"
            self.csv_folder_path = "C:\\Users\\44753\\Desktop\\MainEval"
            self.cache_dir = "C:\\Users\\44753\\Desktop\\cache2"
        
        os.makedirs(self.cache_dir, exist_ok=True)

        self.all_videos = self.list_files_in_folder(self.video_folder_path)
        self.all_csv = self.list_files_in_folder(self.csv_folder_path)

        self.frames = []
        self.stick_labels = []
        self.button_labels = []

        for i in range(len(self.all_videos)):
            cache_file = os.path.join(self.cache_dir, f"video_{i:03d}.pt")
            if os.path.exists(cache_file):
                #print(f"Cache Exist: {cache_file}")
                print(f"Loading cache: {cache_file}")
                cached = torch.load(cache_file)
                self.frames.extend(cached["frames"])
                self.stick_labels.extend(cached["stick_labels"])
                self.button_labels.extend(cached["button_labels"])
                continue

            print(f"Processing video {i}: building data...")
            current_video_path = self.video_folder_path + "\\" + self.all_videos[i]
            current_csv_path = self.csv_folder_path + "\\" + self.all_csv[i]
            data = pd.read_csv(current_csv_path)

            reader = imageio.get_reader(current_video_path, 'ffmpeg')
            all_frames = [reader.get_data(j) for j in range(reader.count_frames())]
            reader.close()

            frame_ids = data["VFRAME_ID"].values.astype(int)
            stick_col = ['LSTICKX', 'LSTICKY', 'RSTICKX', 'RSTICKY', 'R2']
            button_col = ['L1', 'R1', 'L2', 'L3', 'R3', 'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'TRI', 'SQUARE', 'CROSS', 'CIRCLE', 'TOUCHPAD', 'TOUCHPOINT0_ID', 'TOUCHPOINT0_X', 'TOUCHPOINT0_y', 'TOUCHPOINT1_ID', 'TOUCHPOINT1_X', 'TOUCHPOINT1_y']

            stick_values = data[stick_col].values.astype(np.float32) / 255.0
            button_values = data[button_col].values.astype(np.uint8)

            
            video_frames = []
            video_stick = []
            video_button = []

            for j, frame_id in enumerate(frame_ids):
                if frame_id + interval_frames >= len(all_frames):
                    continue
                cframe1 = self.transform(Image.fromarray(all_frames[frame_id - 1]))
                cframe2 = self.transform(Image.fromarray(all_frames[frame_id + interval_frames - 1]))
                video_frames.append((cframe1, cframe2))
                video_stick.append(stick_values[j])
                video_button.append(button_values[j])

            self.frames.extend(video_frames)
            self.stick_labels.extend(video_stick)
            self.button_labels.extend(video_button)

            # Save Built Dataset
            torch.save({
                "frames": video_frames,
                "stick_labels": video_stick,
                "button_labels": video_button
            }, cache_file)
            print(f"Saved cache to {cache_file}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i], self.stick_labels[i], self.button_labels[i]

    def list_files_in_folder(self, folder_path):
        return sorted(os.listdir(folder_path))