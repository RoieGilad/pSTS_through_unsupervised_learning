import glob
import os
import shutil
import time
from os import path, remove
from natsort import natsorted

import cv2
import numpy as np
import pandas as pd
import torchaudio
import torchvision.transforms as T
from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm
from typing import List
import math

import data_processing.data_utils as du
import MTCNN.mtcnn_pytorch.src

windows = True  # TODO change for gpu
cuda = False  # TODO chnange for GPU


def create_metadata_file():
    md = pd.DataFrame()
    md['sample_index'] = []
    md['speaker_id'] = []
    md['id_part'] = []
    return md


def data_flattening(video_source_dir, audio_source_dir, destination_dir,
                    delete_origin=False, sample_limit=10000):
    """ iterate source dir of voxceleb2 dataset
    and create samples directory with audio and video subdirectories.
    the function create xl file saving the sample label - speaker id
    and num of video's frames."""
    md_path = du.get_label_path(destination_dir)
    metadata_df = create_metadata_file()
    sample_num, row = 0, 0
    sample_indexes, speaker_ids, id_part = [], [], []
    video_id_directories = natsorted(
        glob.glob(path.join(video_source_dir, "id*")))
    audio_id_directories = natsorted(
        glob.glob(path.join(audio_source_dir, "id*")))
    sample_num = 0
    row = 0

    # Iterate over id directories of audio & video data_processing
    for video_id_directory, audio_id_directory in zip(video_id_directories,
                                                      audio_id_directories):
        if not du.checks_same_videos_audios_data(video_id_directory,
                                                 audio_id_directory):
            return 0
        id = path.basename(video_id_directory)
        part = 0
        # Get the next-level subdirectories (number of video/audio)
        video_subdirectories = natsorted(
            glob.glob(path.join(video_id_directory, '*')))
        audio_subdirectories = natsorted(
            glob.glob(path.join(audio_id_directory, '*')))

        # Iterate over video and audio subdirectories
        for video_subdir, audio_subdir in zip(video_subdirectories,
                                              audio_subdirectories):
            video_id_samples = sorted(
                glob.glob(path.join(video_subdir, '*.mp4')))
            audio_id_samples = sorted(
                glob.glob(path.join(audio_subdir, '*.wav')))

            for video_id_sample, audio_id_sample in zip(video_id_samples,
                                                        audio_id_samples):

                if not du.checks_same_videos_audios_data(video_id_sample,
                                                         audio_id_sample):
                    return 0
                du.create_sample_video_audio_dirs(destination_dir,
                                                  video_id_sample,
                                                  audio_id_sample, sample_num,
                                                  delete_origin)
                sample_indexes.append(sample_num)
                id_part.append(part)
                speaker_ids.append(id)
                row += 1
                sample_num += 1

            if sample_num > sample_limit:
                break
            part += 1

        if sample_num > sample_limit:
            break

    metadata_df['sample_index'] = np.asarray(sample_indexes)
    metadata_df['speaker_id'] = np.asarray(speaker_ids)
    metadata_df['id_part'] = np.asarray(id_part)
    du.split_and_save(metadata_df, md_path)
    print("Data flattened successfully")
    return sample_num


def center_face_by_path(path_to_image, override=True):
    transform = T.RandomCrop((222, 222))
    img_to_run = img = Image.open(path_to_image)
    num_tries = 50
    for i in range(num_tries):
        try:
            bounding_boxes, _ = MTCNN.mtcnn_pytorch.src.detect_faces(img_to_run,
                                                                     cuda)
            if len(bounding_boxes) > 0:
                cropped_img = img_to_run.crop(bounding_boxes[0][:4])
                if not override:
                    path_to_image = du.add_addition_to_path(path_to_image,
                                                            "centered")
                cropped_img.save(path_to_image)
                return 0
        except Exception as e:
            # print("An exception occurred:", str(e))
            img_to_run = transform(img)
    # print("Failed!", path_to_image)
    return 1


def center_faces_by_folder(vf, override=True):
    cnt_failure = 0
    num_jpg = len(glob.glob(path.join(vf, "*." + "jpg")))
    for jpeg_path in du.file_iterator_by_type(vf, "jpg"):
        if "centered" not in jpeg_path:
            cnt_failure += center_face_by_path(jpeg_path, override)
        if cnt_failure / num_jpg > 0.15:
            index = du.get_sample_index(vf)
            return [index] if index else []
    return []


def delete_samples(root_dir: str, to_delete: List[str]):
    to_delete = set(to_delete)
    for sample_dir in du.folder_iterator_by_path(
            path.join(root_dir, "sample_*")):
        if du.get_sample_index(sample_dir) in to_delete:
            shutil.rmtree(sample_dir)
    md_path = du.get_label_path(root_dir)
    data_md = du.read_metadata(md_path)
    index_to_delete = []
    for i, row in data_md.iterrows():
        if 'sample_' + str(row['sample_index']) in to_delete:
            index_to_delete.append(i)
    data_md.drop(index_to_delete, inplace=True)
    data_md.reset_index(drop=True, inplace=True)
    du.split_and_save(data_md, md_path)


def center_all_faces(root_dir: str, override=True):
    """given root dir, center all the images in the sub video folders
    when override is True and the output is saved under a new name"""
    samples_to_delete = []
    for vf in tqdm(du.video_folder_iterator(root_dir),
                   desc="Center Videos:"):
        samples_to_delete.extend(center_faces_by_folder(vf, override))
    print("done centering, the following samples should be deleted:")
    print(samples_to_delete)
    if samples_to_delete:
        delete_samples(root_dir, samples_to_delete)


def iterate_over_frames(video, path_to_video_dir, frame_interval):
    """Iterate over video's/audio's frames and save each frame
    in the video/audio root directory. the function return the frame rate of video and count"""
    sample_index = du.get_sample_index(path_to_video_dir)
    frame_origin_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_origin_rate = video.get(cv2.CAP_PROP_FPS)
    duration_ms = (frame_origin_count / frame_origin_rate) * 1000
    duration_desire = (duration_ms // frame_interval) * frame_interval
    frame_count = 0
    interval = frame_interval
    interval_num = 0
    ret, frame = video.read()
    while ret:
        frame_time_stamp = video.get(cv2.CAP_PROP_POS_MSEC)
        if frame_time_stamp > duration_desire - 0.01:
            break
        if frame_time_stamp >= interval:
            interval += frame_interval
            interval_num += 1
        save_video_frame(frame, path_to_video_dir, sample_index, frame_count,
                         interval_num)
        frame_count += 1
        ret, frame = video.read()
    frame_rate = 1000 / frame_interval
    return frame_rate, frame_count, interval_num + 1


def save_video_frame(frame, path_to_video_dir, sample_index, frame_count,
                     interval_num):
    """Save a video frame to the specified directory"""
    frame_path = path.join(path_to_video_dir,
                           f"{sample_index}_v_{interval_num}_{frame_count}.jpg")
    cv2.imwrite(frame_path, frame)
    os.chmod(frame_path, 0o0777)


def split_video_to_frames(sample_index, path_to_video_dir: str,
                          frame_interval, delete_video: bool = False):
    """extract the frames from the video and save
    them in the same directory. The function saves the number of
    frames as metadata for each sample video in the xl file"""
    path_to_video_file = \
        list(du.file_iterator_by_type(path_to_video_dir, "mp4"))[0]
    video = cv2.VideoCapture(path_to_video_file)  # Open the video file
    if not video.isOpened():
        print(f'Error in split the video file of sample_{sample_index}')
        return -1, -1, -1
    frame_rate, num_frames, num_intervals = iterate_over_frames(video,
                                                                path_to_video_dir,
                                                                frame_interval)
    video.release()
    if delete_video:
        remove(path_to_video_file)
    return frame_rate, num_frames, num_intervals


def split_all_videos(path_to_data: str, frame_interval,
                     delete_video: bool = False):
    """ The function iterate all video directories corresponding each
    sample and calls split_video_to_frames function. the function save
    the last metadata dataframe after saving all frames rates for each
    video"""
    md_path = du.get_label_path(path_to_data)
    metadata_df = du.read_metadata(md_path)
    failed_to_split = []
    index_to_fr, index_to_nf, index_to_ni = dict(), dict(), dict()
    for path_to_video_dir in tqdm(du.video_folder_iterator(path_to_data),
                                  desc="Split Videos:"):
        sample_index = du.get_num_sample_index(path.dirname(path_to_video_dir))
        fr, nf, ni = split_video_to_frames(sample_index, path_to_video_dir,
                                           frame_interval, delete_video)
        if fr == nf == ni == -1:
            failed_to_split.append(sample_index)

        index_to_fr[sample_index] = fr
        index_to_nf[sample_index] = nf
        index_to_ni[sample_index] = ni

    index_to_nf = np.asarray(
        [index_to_nf[i] for i in sorted(index_to_nf.keys())])
    index_to_fr = np.asarray(
        [index_to_fr[i] for i in sorted(index_to_fr.keys())])
    index_to_ni = np.asarray(
        [index_to_ni[i] for i in sorted(index_to_ni.keys())])
    metadata_df.insert(3, "frame_rate", index_to_fr, False)
    metadata_df.insert(4, "numer_of_frames", index_to_nf, False)
    metadata_df.insert(5, "num_video_intervals", index_to_ni, False)
    du.split_and_save(metadata_df, md_path)
    print(f'these samples have been failed to video split: {failed_to_split}')


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
                                     desc="Concatenate Audio:"):
        concatenate_audio_by_dir(path_to_audio_folder,
                                 path.join(path_to_audio_folder, "cont.wav"))


def split_audio(path_to_audio_file: str,
                interval: int, delete_input: bool = False):
    """takes an audio file and split it a different audio files s.t. for each
        video frame there is an "audio frame" in size window_size and the video frame
        "taken" from the middle of the audio frame, if delete_input the input path
        will be deleted"""
    if du.is_mp4a(path_to_audio_file) and windows:
        path_to_audio_file = convert_mp4a_to_wav(path_to_audio_file)
    audio = AudioSegment.from_wav(path_to_audio_file)
    duration_ms = len(audio)
    duration_desire = (duration_ms // interval) * interval
    num_slices = -1
    for start in range(0, duration_desire, interval):
        num_slices += 1
        slice = audio[start:start + interval]
        output_path = du.add_addition_to_path(path_to_audio_file,
                                              f"a_{num_slices}")
        slice.export(output_path, format="wav")
    if delete_input:
        remove(path_to_audio_file)
    return num_slices + 1


def split_all_audio(path_to_data: str, interval: int, delete_input=False):
    path_to_md = du.get_label_path(path_to_data)
    data_md = du.read_metadata(path_to_md)
    index_to_num = dict()
    samples_to_delete = []
    for path_to_audio_folder in tqdm(du.audio_folder_iterator(path_to_data),
                                     desc="Split Audio:"):
        sample_index = du.get_num_sample_index(path_to_audio_folder)
        video_frame_rate = du.get_video_frame_rate(data_md, sample_index)
        if video_frame_rate > 0:
            for path_to_audio_file in du.file_iterator_by_type(
                    path_to_audio_folder, "wav"):
                index_to_num[sample_index] = split_audio(
                    path_to_audio_file, interval,
                    delete_input)

        else:
            index_to_num[sample_index] = 0
            samples_to_delete.append('sample_' + str(sample_index))
            print(f'Error in split the video file of {sample_index}, '
                  f'got frame_rate {video_frame_rate}')

    index_to_num = np.asarray(
        [index_to_num[i] for i in sorted(index_to_num.keys())])
    data_md.insert(5, "num_audio_intervals", index_to_num, False)
    du.split_and_save(data_md, path_to_md)
    print(f'these samples have been failed to audio split: {samples_to_delete}')
    if samples_to_delete:
        delete_samples(path_to_data, samples_to_delete)


def update_intervals_num(path_to_data):
    """ The function checks if the video and the audio files are already had been processed.
    Then, the function update the metadata file with the minimum intervals for each sample"""
    path_to_md = du.get_label_path(path_to_data)
    data_md = du.read_metadata(path_to_md)
    if 'num_audio_intervals' in data_md and 'num_video_intervals' in data_md:
        data_md['num_of_intervals'] = data_md[
            ['num_audio_intervals', 'num_video_intervals']].min(axis=1)
        data_md = data_md.drop(['num_audio_intervals', 'num_video_intervals'],
                               axis=1)
        du.split_and_save(data_md, path_to_md)


def get_mean_and_std(root_dir):
    print("start cal mean and std of audio")
    mean_std_audio = du.get_mean_std_audio(root_dir)
    du.save_mean_and_std("audio_mean_std.txt", mean_std_audio)
    print("audios: ", mean_std_audio)
    print("start cal mean and std video")
    mean_std_video = du.get_mean_std_video(root_dir)
    du.save_mean_and_std("video_mean_std.txt", mean_std_video)
    print("videos: ", mean_std_video)


def filter_dataset_by_label(root_dir, reference_dir):
    reference_md = du.read_metadata(du.get_label_path(reference_dir))
    unique_labels = set(reference_md['speaker_id'].unique())
    unique_labels = {int(l[2:]) for l in unique_labels}
    to_delete, keep = [], []
    test_md = du.read_metadata(du.get_label_path(root_dir))
    for index, row in test_md.iterrows():
        if int(row['speaker_id'][2:]) not in unique_labels:
            print(row['speaker_id'], row['speaker_id'] not in unique_labels, unique_labels)
            to_delete.append('sample_' + str(row['sample_index']))
        else:
            keep.append('sample_' + str(row['sample_index']))
    print('keep: ', keep)
    if to_delete:
        delete_samples(root_dir, to_delete)



