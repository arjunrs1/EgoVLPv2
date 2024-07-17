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


class TextNarrationGrounding(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'EPIC_100_train.csv',
            'val': 'EPIC_100_train.csv',
            'test': 'EPIC_100_train.csv'
        }
        """ split_files = {
            'train': 'EPIC_100_retrieval_train.csv',
            'val': 'EPIC_100_retrieval_test.csv',            # there is no test
            'test': 'EPIC_100_retrieval_test.csv'
        }
        split_files_sentence = {
            'train': 'EPIC_100_retrieval_train_sentence.csv',
            'val': 'EPIC_100_retrieval_test_sentence.csv',  # there is no test
            'test': 'EPIC_100_retrieval_test_sentence.csv'
        } """
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))
        #restrict to just one video_id for now
        #metadata = metadata[metadata.video_id == self.video_id]
        #metadata = metadata.sort_values(by='start_timestamp')

        # Sort within each 'video_id' group
        metadata['video_id_order'] = pd.Categorical(metadata['video_id'], categories=metadata['video_id'].unique(), ordered=True)
        metadata = metadata.sort_values(by=['video_id_order', 'start_timestamp']).drop('video_id_order', axis=1)

        self.metadata = metadata

    def _get_video_path(self, sample):
        pid, vid = sample[1:3]
        rel_video_fp = '{}/{}.MP4'.format(pid, vid)
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)

        return full_video_fp, rel_video_fp

    def _get_caption(self, idx, sample):
        return sample[8]
        # return sentence, relevancy score, idx
        if self.split == 'train':
            positive_list = np.where(self.relevancy_mat[idx] > self.relevancy)[0].tolist()
            if positive_list != []:
                pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                    return self.metadata_sentence.iloc[pos][1], self.relevancy_mat[idx][pos], pos
            return sample[8], 1, 0

        elif self.split in ['val', 'test']:
            return sample[8], 1, -1

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

        video_fp, _ = self._get_video_path(sample)
        caption = self._get_caption(item, sample)

        #video_loading = self.video_params.get('loading', 'strict')
        start_timestamp, end_timestamp = self.datetime2sec(sample[4]), self.datetime2sec(sample[5])
        fps = decord.VideoReader(video_fp).get_avg_fps()
        start_frame = int(np.round(fps * start_timestamp))
        end_frame = int(np.ceil(fps * end_timestamp))

        frame_sample = 'rand'
        if self.split in ['test', 'val']:
            frame_sample = 'uniform'
        fix_start = None

        if True:
            if os.path.exists(video_fp):
                frame_ids = self.get_frame_ids(start_frame, end_frame, num_segments=self.video_params['num_frames'], jitter=(self.split == 'train'))
                print("finished loading frames...")
                imgs = self.video_loader_by_frames(video_fp, frame_ids)
                print("finished loading video...")
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
            print("finished transforms...")

        meta_arr = {'video_uid': sample[2], 'clip_uid': sample[3], 'narration_uid': sample[0], 'dataset': self.dataset_name}
        data = {'video': imgs, 'text': caption, 'meta': meta_arr}
        return data

if __name__ == "__main__":
    kwargs = dict(
        dataset_name="EpicKitchens_text_NG",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4, #TODO: Need to increase this after verifying pipeline
        "loading": "lax"
        },
        data_dir="/datasets01/EPIC-KITCHENS-100-VIDEOS-ht256px/060122/",
        meta_dir="/private/home/arjunrs1/epic-kitchens-100-annotations/",
        tsfms=init_video_transform_dict()['test'],
        reader='cv2_epic',
        split='train'
    )
    dataset = TextNarrationGrounding(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())
