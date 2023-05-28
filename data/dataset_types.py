from glob import glob
from os import path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def is_available_by_folder(path, folder, lower_limit):
    path_to_check = path.join(path, folder, "*.jpg")
    return len(glob(path_to_check)) >= lower_limit


def get_sample_frames_interval(paths_to_sample_frames: list[str], num_frames: int, step_size: int,
                               rand: float, sample_transform, batch_transform,
                               end_char: bool = True):
    """ The function return an interval of frames from the sample's frames
    after making some process on it"""
    start_idx = int((len(paths_to_sample_frames) - num_frames * step_size) * rand)
    path_to_sample_frames_interval = paths_to_sample_frames[
                                     start_idx: start_idx + num_frames * step_size: step_size]
    frames = [Image.open(p) for p in path_to_sample_frames_interval]
    if end_char:  # if True: add a black image at the end of every sequence
        frames.append(Image.new(mode="RGB", size=frames[0].size))
    processed_frames = [sample_transform(f) for f in frames]

    if processed_frames:
        processed_frames = torch.stack(processed_frames)
        processed_frames = batch_transform(processed_frames)
    return processed_frames


class VideoDataset(Dataset):
    def __init__(self, ds_root_dir: str, path_to_labels: str, frame_transform,
                 video_transform, num_frames: int = 16, test: bool = False,
                 step_size: int = 1):
        self.test = test
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.ds_path = ds_root_dir
        self.samples = sorted(glob(path.join(self.ds_path, '*')))
        self.labels_map = pd.read_csv(path_to_labels)
        self.num_frames = num_frames
        self.step_size = step_size
        self.tmp_rand = -1

    def __len__(self):
        return len(self.samples)

    def get_label(self, idx):
        return int(self.labels_map.iloc[idx, 1])

    def is_available(self, idx):
        return is_available_by_folder(self.samples[idx], "video",
                                      self.step_size * self.num_frames)

    def __getitem__(self, idx):
        """assume is_available(self, idx) == True when called"""
        path_to_frames = sorted(
            glob(path.join(self.samples[idx], "video", "*.jpg")))
        tmp_rand = self.tmp_rand if self.tmp_rand != -1 else np.random.uniform()
        processed_frames = get_sample_frames_interval(path_to_frames, self.num_frames,
                                                      self.step_size, tmp_rand,
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
        self.samples = sorted(glob(path.join(self.ds_path, '*')))
        self.labels_map = pd.read_csv(path_to_labels)
        self.num_frames = num_frames
        self.step_size = step_size
        self.tmp_rand = -1

    def __len__(self):
        return len(self.samples)

    def get_label(self, idx):
        return int(self.labels_map.iloc[idx, 1])

    def is_available(self, idx):
        return is_available_by_folder(self.samples[idx], "audio",
                                      self.step_size * self.num_frames)

    def __getitem__(self, idx):
        """assume is_available(self, idx) == True when called"""
        path_to_frames = sorted(
            glob(path.join(self.samples[idx], "video", "*.jpg")))
        tmp_rand = self.tmp_rand if self.tmp_rand != -1 else np.random.uniform()
        processed_frames = get_sample_frames_interval(path_to_frames, self.num_frames,
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
                                     transforms['a_batch _transform'],
                                     num_frames, test, step_size)
        self.video_ds = VideoDataset(ds_root_dir, path_to_labels,
                                     transforms['v_frame_transform'],
                                     transforms['v_batch_transform'],
                                     num_frames, test, step_size)
        self.ds_path = ds_root_dir
        self.labels_map = pd.read_csv(path_to_labels)
        self.samples = sorted(glob(path.join(self.ds_path, '*')))
        self.transforms = transforms
        self.num_frames = num_frames
        self.step_size = step_size

    def __len__(self):
        return len(self.samples)

    def get_label(self, idx):
        return int(self.labels_map.iloc[idx, 1])

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
