import random
from glob import glob
from os import path, getcwd
from natsort import natsorted

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import List
from data_processing import data_utils as du
import torchaudio


def get_sample_video_frames_interval(paths_to_video_sample_frames: List[str],
                                     video_frame_transform,
                                     video_batch_transform):
    """ The function return an interval of frames from the sample's frames
    after making some process on it"""
    frames = [Image.open(p) for p in paths_to_video_sample_frames]
    processed_frames = [video_frame_transform(f) for f in frames]

    if processed_frames:
        processed_frames = torch.stack(processed_frames)
        processed_frames = video_batch_transform(processed_frames)
    return processed_frames


def get_sample_audio_frames_interval(paths_to_audio_sample_frames: List[str], num_frames: int, step_size: int,
                                     rand: float, audio_frame_transform, audio_batch_transform):
    """ The function return an interval of frames from the sample's frames
    after making some process on it"""
    du.pick_new_amplitude_gain()
    start_idx = int((len(paths_to_audio_sample_frames) - num_frames * step_size) * rand)
    # start_idx = 2
    path_to_sample_frames_interval = paths_to_audio_sample_frames[
                                     start_idx: start_idx + num_frames * step_size: step_size]
    frames = [torchaudio.load(p)[0] for p in path_to_sample_frames_interval]
    processed_frames = [audio_frame_transform(f) for f in frames]

    if processed_frames:
        processed_frames = torch.stack(processed_frames)
        processed_frames = audio_batch_transform(processed_frames)
    return processed_frames


class VideoDataset(Dataset):
    def __init__(self, ds_root_dir: str, path_to_labels: str, frame_transform,
                 video_transform, num_frames: int = 16, test: bool = False,
                 step_size: int = 1):
        self.test = test
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.ds_path = ds_root_dir
        self.samples = natsorted(glob(path.join(self.ds_path, 'sample*')))
        self.labels_map = du.read_metadata(path_to_labels)
        self.num_frames = num_frames
        self.step_size = step_size
        self.tmp_rand = -1

    def __len__(self):
        return len(self.samples)

    def get_label(self, idx):
        return self.labels_map.iloc[idx, 1]

    def is_available(self, idx):
        return self.labels_map.iloc[idx, 4] >= (self.step_size * self.num_frames)

    def choose_frames_from_interval(self, idx, num_intervals, tmp_rand):
        paths_to_frames = []
        first_interval = int((num_intervals - self.num_frames * self.step_size)
                             * tmp_rand)
        #first_interval = 2
        real_index = du.get_real_index_by_path(self.samples[idx])
        for i in range(self.num_frames):
            interval_frames = natsorted(glob(path.join(self.samples[idx], "video",
                                                       f"sample_{real_index}_v_{first_interval + i}_*")))
            paths_to_frames.append(random.choice(interval_frames))
            #paths_to_frames.append(interval_frames[len(interval_frames)//2])
        return paths_to_frames

    def __getitem__(self, idx):
        """assume is_available(self, idx) == True when called"""
        num_intervals = self.labels_map.iloc[idx, 4]
        tmp_rand = self.tmp_rand if self.tmp_rand != -1 else np.random.uniform()
        paths_to_frames = self.choose_frames_from_interval(idx, num_intervals, tmp_rand)
        processed_frames = get_sample_video_frames_interval(paths_to_frames,
                                                            self.frame_transform,
                                                            self.video_transform)
        self.tmp_rand = -1
        label = self.get_label(idx)
        return processed_frames, label


class AudioDataset(Dataset):
    def __init__(self, ds_root_dir: str, path_to_labels: str, frame_transform,
                 audio_transform, num_frames: int = 16, test: bool = False,
                 step_size: int = 1):
        self.test = test
        self.frame_transform = frame_transform
        self.audio_transform = audio_transform
        self.ds_path = ds_root_dir
        self.samples = natsorted(glob(path.join(self.ds_path, 'sample*')))
        self.labels_map = du.read_metadata(path_to_labels)
        self.num_frames = num_frames
        self.step_size = step_size
        self.tmp_rand = -1

    def __len__(self):
        return len(self.samples)

    def get_label(self, idx):
        return self.labels_map.iloc[idx, 1]

    def is_available(self, idx):
        return self.labels_map.iloc[idx, 4] >= (self.step_size * self.num_frames)

    def __getitem__(self, idx):
        """assume is_available(self, idx) == True when called"""
        num_intervals = self.labels_map.iloc[idx, 4]
        path_to_frames = natsorted(
            glob(path.join(self.samples[idx], "audio", "*.wav")))[:num_intervals]
        tmp_rand = self.tmp_rand if self.tmp_rand != -1 else np.random.uniform()
        processed_frames = get_sample_audio_frames_interval(path_to_frames, self.num_frames,
                                                            self.step_size, tmp_rand,
                                                            self.frame_transform,
                                                            self.audio_transform)
        self.tmp_rand = -1
        label = self.get_label(idx)
        return processed_frames, label


class CombinedDataset(Dataset):
    def __init__(self, ds_root_dir: str, path_to_labels: str, transforms: dict,
                 test: bool = False, num_frames: int = 16, step_size: int = 1):
        self.test = test
        self.audio_ds = AudioDataset(ds_root_dir, path_to_labels,
                                     transforms['a_frame_transform'],
                                     transforms['a_batch_transform'],
                                     num_frames, test, step_size)
        self.video_ds = VideoDataset(ds_root_dir, path_to_labels,
                                     transforms['v_frame_transform'],
                                     transforms['v_batch_transform'],
                                     num_frames, test, step_size)
        self.ds_path = ds_root_dir
        self.samples = natsorted(glob(path.join(self.ds_path, 'sample*')))
        self.labels_map = du.read_metadata(path_to_labels)
        self.samples = natsorted(glob(path.join(self.ds_path, 'sample*')))
        self.transforms = transforms
        self.num_frames = num_frames
        self.step_size = step_size

    def __len__(self):
        return len(self.samples)

    def get_label(self, idx):
        return self.labels_map.iloc[idx, 1]

    def __getitem__(self, idx):
        while not (self.audio_ds.is_available(idx) and
                   self.video_ds.is_available(idx)):
            idx = (idx + 1) % len(self)
        self.tmp_rand = np.random.uniform() if not self.test else 0
        self.audio_ds.tmp_rand = self.video_ds.tmp_rand = self.tmp_rand  # synchronization
        audio_data, audio_label = self.audio_ds[idx]
        video_data, video_label = self.video_ds[idx]
        main_label = self.get_label(idx)
        if not (audio_label == video_label == main_label):
            return self[(idx + 1) % len(self)]
        return video_data, audio_data, main_label
