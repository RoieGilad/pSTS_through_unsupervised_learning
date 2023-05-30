import os

import cv2
import glob
from pydub import AudioSegment

import numpy as np
import openpyxl
import pandas as pd
import shutil
from PIL import Image
from os import makedirs, path, remove
import torchaudio
from tqdm import tqdm
import stat
import data.data_utils as du
from MTCNN.mtcnn_pytorch.src import detect_faces

windows = True  # TODO change for gpu


# TODO think if the file path is global
def create_metadata_file():
    md = pd.DataFrame()
    md['sample_index'] = []
    md['speaker_id'] = []
    return md


def data_flattening(source_dir, dest_dir, file_extension, data_type,
                    write_labels):
    """ Iterate source dir of voxceleb2 dataset and create samples directory
    with audio and video subdirectories. The function create xl file saving the
    sample label - speaker id and num of video's frames."""
    md_path = du.get_label_path(dest_dir)
    metadata_df = create_metadata_file()
    id_directories = sorted(glob.glob(path.join(source_dir, "id*")))
    sample_num = 0
    sample_indexes, speaker_ids = [], []
    row = 0

    for id_directory in tqdm(id_directories,
                             desc=f'"flattening for {data_type} I.D Directories:'):
        speaker_id = path.basename(id_directory)
        # Gets all audio/video sample of speaker_id
        id_samples = sorted(glob.glob(path.join(id_directory, '*',
                                                f'*{file_extension}')))
        for id_sample in id_samples:  # Iterate over samples
            destination_sample_path = path.join(dest_dir,
                                                f'sample_{sample_num}',
                                                str(data_type))
            makedirs(destination_sample_path, exist_ok=True)
            file_path = path.join(destination_sample_path,
                                  f'sample_{sample_num}{file_extension}')
            shutil.copy(id_sample, file_path)  # Copy samples file from source
            os.chmod(file_path, 0o0777)

            if write_labels:
                sample_indexes.append(sample_num)
                speaker_ids.append(speaker_id)

                row += 1
            sample_num += 1

    if write_labels:
        metadata_df['sample_index'] = np.asarray(sample_indexes)
        metadata_df['speaker_id'] = np.asarray(speaker_ids)
        metadata_df.to_excel(md_path, index=False)
    return sample_num


def data_flattening_env(audio_source_dir, video_source_dir, destination_dir):
    """Flattening for video and audio files."""
    num1 = data_flattening(audio_source_dir, destination_dir, '.m4a', 'audio',
                           True)
    num2 = data_flattening(video_source_dir, destination_dir, '.mp4', 'video',
                           False)
    if num1 != num2:
        print("Error in flattening, not the samue number of samples")


def center_face_by_path(path_to_image, override=True):
    try:
        img = Image.open(path_to_image)
        bounding_boxes, _ = detect_faces(img)
        if len(bounding_boxes) > 0:
            cropped_img = img.crop(bounding_boxes[0][:4])
            if not override:
                path_to_image = du.add_addition_to_path(path_to_image,
                                                        "centered")
            cropped_img.save(path_to_image)
    except:
        pass


def center_all_faces(root_dir: str, override=True):
    """given root dir, center all the images in the sub video folders
    when override is True and the output is saved under a new name"""
    for vf in tqdm(du.video_folder_iterator(root_dir), desc="Center Videos:"):
        for jpeg_path in du.file_iterator_by_type(vf, "jpg"):
            if "centered" not in jpeg_path:
                center_face_by_path(jpeg_path, override)


def iterate_over_frames(video, path_to_video_dir):
    """Iterate over video's frames and save each frame
    in the video root directory. the function return the frame rate of video"""
    frame_count = 0
    ret, frame = video.read()
    while ret:
        frame_path = path.join(path_to_video_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        os.chmod(frame_path, 0o0777)
        frame_count += 1
        ret, frame = video.read()
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    return frame_rate, frame_count


def split_video_to_frames(path_to_video_dir: str, delete_video: bool = False):
    """ The function extract the frames from the video and save
    them in the same directory. The function saves the number of
    frames as metadata for each sample video in the xl file"""
    path_to_video_file = \
        list(du.file_iterator_by_type(path_to_video_dir, "mp4"))[0]
    sample_index = du.get_num_sample_index(path.dirname(path_to_video_file))
    video = cv2.VideoCapture(path_to_video_file)  # Open the video file
    if not video.isOpened():
        print(f'Error in split the video file of sample_{sample_index}')
        return 0
    frame_rate, num_frames = iterate_over_frames(video, path_to_video_dir)
    video.release()
    if delete_video:
        remove(path_to_video_file)
    return sample_index, frame_rate, num_frames


def split_all_videos(path_to_data: str, delete_video: bool = False):
    """ The function iterate all video directories corresponding each
    sample and calls split_video_to_frames function. the function save
    the last metadata dataframe after saving all frames rates for each
    video"""
    md_data = du.get_label_path(path_to_data)
    metadata_df = pd.read_excel(md_data)
    index_to_fr, index_to_nf = dict(), dict()
    for path_to_video_dir in tqdm(du.video_folder_iterator(path_to_data),
                                  desc="Split Videos:"):
        index, fr, nf = split_video_to_frames(path_to_video_dir, delete_video)
        index_to_fr[index] = fr
        index_to_nf[index] = nf
    index_to_nf = np.asarray(
        [index_to_nf[i] for i in sorted(index_to_nf.keys())])
    index_to_fr = np.asarray(
        [index_to_fr[i] for i in sorted(index_to_fr.keys())])
    metadata_df.insert(2, "frame_rate", index_to_fr, False)
    metadata_df.insert(3, "numer_of_frames", index_to_nf, False)
    metadata_df.to_excel(md_data, index=False)
    os.chmod(md_data, 0o0777)


def convert_mp4a_to_wav(path_to_audio_file):
    wav_filename = path_to_audio_file[:-3] + "wav"
    track = AudioSegment.from_file(path_to_audio_file, format='m4a')
    file_handle = track.export(wav_filename, format='wav')
    return wav_filename

def concatenate_audio_files(input_files, output_file):
    output_audio = AudioSegment.empty()
    for file in input_files:
        audio = AudioSegment.from_file(file)
        output_audio += audio

    output_audio.export(output_file, format="mp3")

def concatenate_audio_by_dir(input_dir, output_path):
    files = glob.glob(path.join(input_dir, "*." + "wav"))
    files.sort(key=lambda x: du.get_num_frame(x))
    concatenate_audio_files(files, output_path)

def concatenate_all_audio(path_to_data):
    for path_to_audio_folder in tqdm(du.audio_folder_iterator(path_to_data),
                                    desc="Split Audio:"):
        concatenate_audio_by_dir(path_to_audio_folder,
                                 path.join(path_to_audio_folder,"cont.wav"))

def split_audio_by_frame_rate(path_to_audio_file: str, frame_rate: int,
                              window_size: int, delete_input: bool = False):
    """takes an audio file and split it a different audio files s.t. for each
    video frame there is an "audio frame" in size window_size and the video frame
    "taken" from the middle of the audio frame, if delete_input the input path
    will be deleted"""
    if du.is_mp4a(path_to_audio_file) and windows:
        path_to_audio_file = convert_mp4a_to_wav(path_to_audio_file)

    waveform, sample_rate = torchaudio.load(path_to_audio_file)
    gap_between_frames = int(sample_rate / frame_rate)
    num_slices = int(waveform.size(1) * (frame_rate / sample_rate))
    half_window_size = window_size // 2 if window_size else gap_between_frames // 2
    for i in range(num_slices):
        mid = gap_between_frames + gap_between_frames * i
        start = max(0.0, mid - half_window_size)
        end = min(waveform.size(1), mid + half_window_size)
        slice = waveform[:, start: end]
        output_path = du.add_addition_to_path(path_to_audio_file, f"a_{i}")
        torchaudio.save(output_path, slice, sample_rate)
    if delete_input:
        os.remove(path_to_audio_file)
    return num_slices


def split_all_audio(path_to_data: str, window_size: int, delete_input=False):
    path_to_md = du.get_label_path(path_to_data)
    data_md = pd.read_excel(path_to_md)
    index_to_num = dict()
    for path_to_audio_folder in tqdm(du.audio_folder_iterator(path_to_data),
                                     desc="Split Audio:"):
        sample_index = du.get_num_sample_index(path_to_audio_folder)
        video_frame_rate = du.get_video_frame_rate(data_md, sample_index)
        if video_frame_rate:
            for path_to_audio_file in du.file_iterator_by_type(
                    path_to_audio_folder, "m4a"):
                index_to_num[sample_index] = split_audio_by_frame_rate(
                    path_to_audio_file, video_frame_rate, window_size,
                    delete_input)
        else:
            index_to_num[sample_index] = 0
            print(f'Error in split the video file of {sample_index}, '
                  f'got frame_rate {video_frame_rate}')

    index_to_num = np.asarray(
        [index_to_num[i] for i in sorted(index_to_num.keys())])
    data_md.insert(4, "num_audio_slices", index_to_num, False)
    data_md.to_excel(path_to_md, index=False)
    os.chmod(path_to_md, 0o0777)
