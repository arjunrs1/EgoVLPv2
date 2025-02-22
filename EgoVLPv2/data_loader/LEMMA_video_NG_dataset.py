# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import random
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video


class LemmaFrameNarrationGrounding(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'train.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))
        self.metadata = metadata
    
    def _get_frame_folder_path(self, sample):
        video_id = sample['video_id']
        view_id = sample['clip_id'].split("_")[-2]
        frame_folder = os.path.join(self.data_dir, video_id, view_id)
        return frame_folder
    
    def _get_caption(self, idx, sample):
        return None
    
    def get_frame_ids(self, start_frame, end_frame, num_segments=32, jitter=True):
        seg_size = float(end_frame - start_frame - 1) / num_segments
        seq = []
        for i in range(num_segments):
            start = int(np.round(seg_size * i) + start_frame)
            end = int(np.round(seg_size * (i + 1)) + start_frame)
            end = min(end, end_frame)
            if jitter:
                frame_id = np.random.randint(low=start, high=(end + 1))
            else:
                frame_id = (start + end) // 2
            seq.append(frame_id)
        return seq
    
    def frame_loader(self, frame_folder, frame_ids):
        frames = []
        for frame_id in frame_ids:
            frame_path = os.path.join(frame_folder, f"img_{frame_id+1:05d}.jpg")
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert('RGB')
                frames.append(img)
            else:
                print(f"Warning: missing frame {frame_path}")
                frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))  # Placeholder black frame
        return frames
    
    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        cam_view = sample['clip_id'].split("_")[-2]
        
        start_frame = sample['start_frame']
        end_frame = sample['end_frame']
        frame_folder = self._get_frame_folder_path(sample)
        
        if not os.path.exists(frame_folder):
            print(f"Warning: missing frame folder {frame_folder}.")
            assert False
        
        frame_ids = self.get_frame_ids(start_frame, end_frame, num_segments=self.video_params['num_frames'], jitter=(self.split == 'train'))
        imgs = self.frame_loader(frame_folder, frame_ids)
        
        crop_size = self.video_params["input_res"]
        if self.split in ['test', 'val']:
            self.transforms = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        imgs = [self.transforms(img) for img in imgs]
        imgs = torch.stack(imgs)  # Shape: [T, C, H, W]
        #imgs = imgs.permute(1, 0, 2, 3)  # Shape: [C, T, H, W]
        
        meta_arr = {'video_uid': sample['video_id'], 'clip_uid': sample['clip_id'], 'view_name': cam_view, 'dataset': self.dataset_name}
        data = {'video': imgs, 'text': "None", 'meta': meta_arr}
        return data


if __name__ == "__main__":
    kwargs = dict(
        dataset_name="EgoExo4D_video_NG",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4, #TODO: Need to increase this after verifying pipeline
        "loading": "lax"
        },
        data_dir="/datasets01/egoexo4d/v2/takes/",
        meta_dir="/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/exo_video_intervals/",
        tsfms=init_video_transform_dict()['test'],
        reader='cv2_epic',
        split='train'
    )
    dataset = EgoExo4DVideoNarrationGrounding(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())
