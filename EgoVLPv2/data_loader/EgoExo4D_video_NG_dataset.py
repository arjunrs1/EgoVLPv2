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
import decord
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video


class EgoExo4DVideoNarrationGrounding(TextVideoDataset): 
    def _load_metadata(self):
        split_files = {
            'train': 'train.csv',
            'val': 'val.csv',
            'test': 'test.csv'
        }

        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))
        self.metadata = metadata

    def _get_video_path(self, sample):
        video_id = sample['video_id']
        take_name, exo_cam = video_id.rsplit("_", 1)
        rel_video_fp = '{}/frame_aligned_videos/downscaled/448/{}.mp4'.format(take_name, exo_cam)
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, idx, sample):
        return None

    def get_frame_ids(self, start_frame, end_frame, num_segments=32, jitter=True):
        seg_size = float(end_frame - start_frame - 1) / num_segments
        seq = []
        for i in tqdm(range(num_segments)):
            start = int(np.round(seg_size * i) + start_frame)
            end = int(np.round(seg_size * (i + 1)) + start_frame)
            end = min(end, end_frame)
            if jitter:
                frame_id = np.random.randint(low=start, high=(end + 1))
            else:
                frame_id = (start + end) // 2
            seq.append(frame_id)
        return seq

    def video_loader_by_frames(self, vid, frame_ids):
        vr = decord.VideoReader(vid)
        try:
            frames = vr.get_batch(frame_ids).numpy()
            frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
        except (IndexError, decord.DECORDError) as error:
            print(frame_ids)
            exit()
            print(error)
            print("Erroneous video: ", vid)
            raise ValueError()
            frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
        return torch.stack(frames, dim=0)

    def datetime2sec(self, st):
        hh, mm, ss = st.split(':')
        return int(hh) * 3600 + int(mm) * 60 + float(ss)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]

        start_frame = sample['start_frame']
        end_frame = sample['end_frame']
        if self.video_id is None:
            video_fp, _ = self._get_video_path(sample)

        frame_sample = 'rand'
        if self.split in ['test', 'val']:
            frame_sample = 'uniform'
        fix_start = None

        if True:
            if os.path.exists(video_fp):
                frame_ids = self.get_frame_ids(start_frame, end_frame, num_segments=self.video_params['num_frames'], jitter=(self.split == 'train'))
                imgs = self.video_loader_by_frames(video_fp, frame_ids)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        if self.split in ['test', 'val']:
            crop_size = self.video_params["input_res"] 
            self.transforms = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                ])
        else:
            crop_size = self.video_params["input_res"]
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
                transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                ])

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                #imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = imgs.permute(3,0,1,2) # T H W C -> C T H W
                imgs = self.transforms(imgs)
                imgs = imgs.permute(1,0,2,3)
                #imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)
            
        meta_arr = {'video_uid': sample['video_id'], 'clip_uid': sample['clip_id'], 'dataset': self.dataset_name}
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
