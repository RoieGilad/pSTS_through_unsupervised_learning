import glob
import os
import random
import shutil
from os import path, remove

import torchaudio
import torchvision.transforms as v_transforms
from torchvision.transforms import functional as F
import torchaudio.transforms as a_transforms
from natsort import natsorted
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

mode = 'nearest'
align_corners = None
amplitude_gain = 0

train_v_frame_transformer = v_transforms.Compose([
    v_transforms.Resize((256, 256)), v_transforms.ToTensor(),
    # v_transforms.Normalize([0.4595, 0.3483, 0.3344], [0.5337, 0.4163, 0.4099])])#mean and std of 160k sample and 100ms audio
    # v_transforms.Normalize([0.4642, 0.3595, 0.3521], [0.5421, 0.4302, 0.4297])])#mean and std of 10k sample and 500ms audio
    v_transforms.Normalize([0.4595, 0.3483, 0.3345], [0.5337, 0.4163, 0.4100])])#mean and std of 160k sample and 500ms audio



train_video_transformer = v_transforms.Compose([
    v_transforms.RandomHorizontalFlip(p=0.5),
    #v_transforms.ColorJitter(),
    v_transforms.RandomCrop([224, 224])])

train_a_frame_transformer = v_transforms.Compose([
    # lambda x: change_amplitude(x),
    # lambda x: add_gaussian_white_noise(x),
    a_transforms.Spectrogram(n_fft=256, hop_length=16),
    # lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), size=(224, 224),
    #                                           mode=mode,
    #                                           align_corners=align_corners),
    # lambda x: x.squeeze(dim=0),
    lambda x: x.expand(3, -1, -1),
    #v_transforms.Normalize([0.6147, 0.6147, 0.6147], [11.1462, 11.1462, 11.1462]) #mean and std of 160k sample and 100ms audio
    # v_transforms.Normalize([0.6911, 0.6911, 0.6911], [12.7088, 12.7088, 12.7088]) # mean and std of 10k and 1s audio
    # v_transforms.Normalize([0.6906, 0.6906, 0.6906], [12.6992, 12.6992, 12.6992]) # mean and std of 10k and 1s audio
    # v_transforms.Normalize([0.6157, 0.6157, 0.6157], [11.2199, 11.2199, 11.2199]) #mean and std of 160k sample and 500ms audio
    v_transforms.Normalize([0.6151, 0.6151, 0.6151], [11.1725, 11.1725, 11.1725]) #mean and std of 160k sample and 500ms audio without interpolation


])

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


def add_gaussian_white_noise(waveform, noise_level=0.01):
    """receives a waveform tensor and add to it a gaussian white noise
    according to the noise level"""
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise


def pick_new_amplitude_gain(low=0.05, high=3.5):
    global amplitude_gain
    amplitude_gain = random.uniform(low, high)


def change_amplitude(waveform):
    "change the amplitude of the waveform by the factor of amplitude_gain"
    amplitude_vol = a_transforms.Vol(gain=amplitude_gain, gain_type="amplitude")
    return amplitude_vol(waveform)


def get_mean_std_video(path_to_data):
    transform = v_transforms.Compose([
        v_transforms.Resize((224, 224)), v_transforms.ToTensor()])
    sum_channels, squared_sum_channels, total_elements = 0, 0, 0
    for vf in tqdm(video_folder_iterator(path_to_data),
                   desc="get_mean_std_video:"):
        batch = []
        for jpeg_path in file_iterator_by_type(vf, "jpg"):
            with Image.open(jpeg_path) as frame:
                batch.append(transform(frame))
        batch = torch.stack(batch, dim=0)
        total_elements += batch.size(0) * batch.size(2) * batch.size(3)
        sum_channels += torch.sum(batch, dim=(0, 2, 3))
        squared_sum_channels += torch.sum(batch ** 2, dim=(0, 2, 3))

    mean_channels = sum_channels / total_elements
    variance_channels = (squared_sum_channels - (
            mean_channels ** 2)) / total_elements
    std_channels = torch.sqrt(variance_channels)
    return mean_channels, std_channels


def get_mean_std_audio(path_to_data):
    transform = v_transforms.Compose([
        a_transforms.Spectrogram(n_fft=256, hop_length=16),
        #lambda x: torch.nn.functional.interpolate(x.unsqueeze(0),
         #                                         size=(224, 224),
          #                                        mode=mode,
           #                                       align_corners=align_corners),
        #lambda x: x.squeeze(dim=0),
        lambda x: x.expand(3, -1, -1),
    ])
    sum_channels, squared_sum_channels, total_elements = 0, 0, 0
    for af in tqdm(audio_folder_iterator(path_to_data),
                   desc="get_mean_std_audio:"):
        batch = [torchaudio.load(p)[0] for p in
                 file_iterator_by_type(af, "wav")]
        batch = [transform(f) for f in batch]
        batch = torch.stack(batch, dim=0)
        total_elements += batch.size(0) * batch.size(2) * batch.size(3)
        sum_channels += torch.sum(batch, dim=(0, 2, 3))
        squared_sum_channels += torch.sum(batch ** 2, dim=(0, 2, 3))

    mean_channels = sum_channels / total_elements
    variance_channels = (squared_sum_channels - (
            mean_channels ** 2)) / total_elements
    std_channels = torch.sqrt(variance_channels)
    return mean_channels, std_channels


def add_idx_to_path(path_to_save, idx):
    directory, filename = os.path.split(path_to_save)
    filename_without_ext, file_extension = os.path.splitext(filename)
    file_extension = file_extension[1:] if file_extension else ""
    if file_extension:
        return os.path.join(directory,
                            filename_without_ext + f'_{idx}.' + file_extension)
    return os.path.join(directory, filename_without_ext + f'_{idx}')


def split_and_save(df, path_to_save):
    max_size = 1000000
    split_dataframe_lst = []
    num_chunks = len(df) // max_size + 1
    for i in range(num_chunks):
        start_idx = i * max_size
        end_idx = (i + 1) * max_size
        split_dataframe_lst.append(df.iloc[start_idx:min(end_idx, len(df))])

    for i in range(len(split_dataframe_lst)):
        path_with_idx = add_idx_to_path(path_to_save, i)
        split_dataframe_lst[i].to_excel(path_with_idx, index=False)
        os.chmod(path_with_idx, 0o0777)
    return


def read_metadata(p):
    directory, filename = os.path.split(p)
    metadata_paths = [p for p in file_iterator_by_type(directory, "xlsx")]
    metadata_dataframes = [pd.read_excel(p) for p in metadata_paths]
    concatenated_df = pd.concat(metadata_dataframes)
    concatenated_df.reset_index(drop=True, inplace=True)
    return concatenated_df


def save_mean_and_std(file_path, numbers):
    with open(file_path, "w") as file:
        file.write(str(numbers[0][0]) + " " + str(numbers[0][1]) + " " + str(numbers[0][2]))
        file.write("\n")
        file.write(str(numbers[1][0]) + " " + str(numbers[1][1]) + " " + str(numbers[1][2]))