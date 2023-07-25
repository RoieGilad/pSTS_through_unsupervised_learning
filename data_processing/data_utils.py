import glob
import os
import shutil
from os import path, remove
import torchvision.transforms as v_transforms
from torchvision.transforms import functional as F
import torchaudio.transforms as a_transforms
from natsort import natsorted
import torch
import numpy as np
from PIL import Image

mode = 'bilinear'
align_corners = True

train_v_frame_transformer = v_transforms.Compose([
    v_transforms.Resize((224, 224)), v_transforms.ToTensor(),
    v_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_end_v_frame_transformer = v_transforms.Compose([
    v_transforms.Resize((224, 224)), v_transforms.ToTensor(),
    v_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_video_transformer = v_transforms.Compose([
    v_transforms.RandomHorizontalFlip(p=0.5),
    v_transforms.ColorJitter(),
    v_transforms.RandomCrop([224, 224])])

train_a_frame_transformer = v_transforms.Compose([
    a_transforms.Spectrogram(n_fft=256, hop_length=16),
    lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), size=(224, 224),
                                              mode=mode, align_corners=align_corners),
    lambda x: x.squeeze(dim=0),
    lambda x: x.expand(3, -1, -1),
    lambda x: F.normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_end_a_frame_transformer = v_transforms.Compose([
        v_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_audio_transformer = v_transforms.Compose([])

def audio_frame_transforms(waveform):
    transform = a_transforms.Spectrogram(n_fft=256, hop_length=16)
    mel_specgram = transform(waveform)
    return torch.squeeze(mel_specgram, dim=0)

def add_addition_to_path(input_path, addition):
    """ for a given path = dirname/base_name.type
    will return the new path: dirname/base_name_addition.type"""
    dirname, base_name = path.split(input_path)
    file_name, file_ext = path.splitext(base_name)
    base_name = "".join([file_name, "_", addition, file_ext])
    return path.join(dirname, base_name)


def folder_iterator_by_path(root_dir: str):
    """yield the folder path of the next sample, in order"""
    for p in natsorted(glob.glob(root_dir, recursive=False)):
        if not path.isfile(p):
            yield p


def file_iterator_by_path(root_dir: str):
    """yield the file path of the next sample, in order"""
    for p in natsorted(glob.glob(root_dir, recursive=False)):
        if path.isfile(p):
            yield p


def audio_folder_iterator(root_dir: str):
    """return all the file paths matching the following pattern
    samples_root_dir/*/audio, in-order"""
    for p in folder_iterator_by_path(path.join(root_dir, "*", "audio")):
        yield p


def video_folder_iterator(root_dir: str):
    """return all the file paths matching the following pattern
    samples_root_dir/*/video, in-order"""
    for p in folder_iterator_by_path(path.join(root_dir, "*", "video")):
        yield p


def file_iterator_by_type(root_dir: str, type: str):
    """return all path of the files in the root_dir from the type is given
     as input, in-order"""
    for p in natsorted(glob.glob(path.join(root_dir, "*." + type))):
        yield p


def get_sample_index(p: str):
    p = os.path.normpath(p)
    for dir_name in p.split(os.sep):
        if dir_name.startswith("sample"):
            return dir_name
    return ""


def get_num_sample_index(p: str):
    sample_index = get_sample_index(p)
    if sample_index:
        _, index = get_sample_index(p).split("_")
        return int(index)
    return -1


def is_mp4a(path_to_audio_file):
    return path_to_audio_file[-3:] == "m4a"


def get_md_by_sample_id_and_column(data_md, s_id, column):
    """return the frame rate of sample_id from the given DF"""
    if s_id > -1:
        row = data_md[data_md["sample_index"] == s_id]
        return row[column].values[0]
    return ""


def get_video_frame_rate(data_md, sample_id: int):
    """return the frame rate of sample_id from the given DF"""
    if sample_id > -1:
        return int(get_md_by_sample_id_and_column(data_md, sample_id,
                                                  "frame_rate"))
    return 0


def get_label_path(data_root_dir: str):
    dir = path.join(os.getcwd(), data_root_dir)
    os.makedirs(dir, exist_ok=True)
    return path.join(dir, "data_md.xlsx")


def get_num_frame(path):
    path = path[::-1]
    return int(path[path.find(".") + 1: path.find("_")][::-1])


def checks_same_videos_audios_data(video_data, audio_data):
    """checks if the video and audio data_processing matches.
    the function checks for one sample if it appears in both video
    & audio directories - by id/sample basenames"""
    video_dir = path.basename(video_data)
    audio_dir = path.basename(audio_data)
    video_dir = video_dir[:video_dir.find(".")]
    audio_dir = audio_dir[:audio_dir.find(".")]
    if audio_dir != video_dir:
        print(
            f'Video data_processing and audio data_processing doesnt match.\n' f'{video_dir}' '!=' f'{audio_dir}')
        return 0
    return 1


def checks_same_videos_audios_id_samples(video_id_directory,
                                         audio_id_directory):
    video_speaker_id = path.basename(video_id_directory)
    audio_speaker_id = path.basename(audio_id_directory)
    if audio_speaker_id != video_speaker_id:
        print("Video data_processing and audio data_processing doesnt match")
        return 0
    return 1


def create_sample_video_audio_dirs(destination_dir, video_id_sample,
                                   audio_id_sample, sample_num, delete_origin):
    """create the video and audio directories for one sample
    and copy the corresponding files from source directory"""
    # Create dirs
    destination_sample_path = path.join(destination_dir, f'sample_{sample_num}')
    destination_video_sample_path = path.join(destination_sample_path, 'video')
    destination_audio_sample_path = path.join(destination_sample_path, 'audio')
    os.makedirs(destination_video_sample_path, exist_ok=True)
    os.makedirs(destination_audio_sample_path, exist_ok=True)

    os.chmod(destination_video_sample_path, 0o0777)
    os.chmod(destination_audio_sample_path, 0o0777)
    os.chmod(destination_sample_path, 0o0777)

    # Copy files
    shutil.copy(video_id_sample, path.join(destination_video_sample_path,
                                           f'sample_{sample_num}.mp4'))
    shutil.copy(audio_id_sample, path.join(destination_audio_sample_path,
                                           f'sample_{sample_num}.m4a'))
    os.chmod(destination_video_sample_path, 0o0777)
    os.chmod(destination_audio_sample_path, 0o0777)
    if delete_origin:
        remove(video_id_sample)
        remove(audio_id_sample)


def get_real_index_by_path(sample_path):
    """expected path to be in format some_path/sample_{real_index} and the
    function will return real_index"""
    return sample_path[sample_path.rfind("_") + 1:]
