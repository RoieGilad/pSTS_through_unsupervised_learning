from os import path
from PIL import Image
import data_utils as du
from MTCNN.mtcnn_pytorch.src import detect_faces
from torch import torchaudio
import openpyxl
import glob
from os import makedirs
from os import remove
import shutil
import pandas as pd
import cv2

audio_source_dir = "audio_dir_sample_17_before_flattening"
video_source_dir = "video_dir_sample_17_before_flattening"
destination_dir = "sample_17_after_flattening"
metadata_file = "../demo_data/labels.xlsx"


# TODO think if the file path is global
def create_metadata_file():
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet['A1'] = 'sample_index'
    sheet['B1'] = 'speaker_id'
    sheet['C1'] = 'frame_rate'
    wb.save(metadata_file)


def data_flattening(source_dir, destination_dir, file_extension, type, write_labels):
    """ This function iterate source dir of voxceleb2 dataset
    and create samples directory with audio and video subdirectories.
    the function create xl file saving the sample label - speaker id
    and num of video's frames."""

    create_metadata_file()
    # Open xl file as dataframe
    metadata_df = pd.read_excel(metadata_file)

    id_directories = sorted(glob.glob(path.join(source_dir, "id*")))
    sample_num = 0
    row = 0

    # Iterate over id directories of audio/video data
    for id_directory in id_directories:
        speaker_id = path.basename(id_directory)

        # Gets all audio/video sample corresponding speaker_id
        id_samples = sorted(glob.glob(path.join(id_directory, '*', f'*{file_extension}')))
        # Iterate over samples
        for id_sample in id_samples:
            # Creating the sample directories
            destination_sample_path = path.join(destination_dir, f'sample_{sample_num}')
            destination_sample_path = path.join(destination_sample_path, f'{type}')
            makedirs(destination_sample_path, exist_ok=True)
            # Copy samples file from source
            shutil.copy(id_sample, path.join(destination_sample_path, f'sample_{sample_num}{file_extension}'))
            # Writing label for corresponding sample into xl file
            if write_labels:
                metadata_df[row, 'sample_index'] = sample_num
                metadata_df[row, 'speaker_id'] = speaker_id
                row += 1
            sample_num += 1

    if write_labels:
        metadata_df.to_excel(metadata_file, index=False)


def data_flatenning_env(audio_source_dir, video_source_dir, destination_dir):
    """ The function calls data flattening function for video and audio files."""
    data_flattening(audio_source_dir, destination_dir, '.m4a', 'audio', True)
    data_flattening(video_source_dir, destination_dir, '.mp4', 'video', False)


def center_face_by_path(path_to_image, override=True):
    img = Image.open(path_to_image)
    bounding_boxes, _ = detect_faces(img)
    cropped_img = img.crop(bounding_boxes)
    if not override:
        path_to_image = du.add_addition_to_path(path, "centered")
    cropped_img.save(path_to_image)


def center_all_faces(root_dir: str, override=True):
    """given root dir, center all the images in the sub video folders
    when override is True and the output is saved under a new name"""
    for video_folder in du.video_folder_iterator(root_dir):
        for jpeg_path in du.file_iterator_by_type(video_folder, "jpg"):
            center_face_by_path(jpeg_path, override)


def iterate_over_frames(video, path_to_video_dir):
    """ The function iterate over video's frames and save each frame
    in the video root directory. the function return the frame rate of video"""

    frame_count = 0
    # Read frames from the video until the end
    while True:
        ret, frame = video.read()
        # Break the loop if no frame is read
        if not ret:
            break
        frame_path = path.join(path_to_video_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    return frame_rate


def split_video_to_frames(path_to_video_file: str, metadata_df, delete_video: bool = False):
    """ The function extract the frames from the video and save
    them in the same directory. The function saves the number of
    frames as metadata for each sample video in the xl file"""

    path_to_video_dir = path.dirname(path_to_video_file)
    sample_index = du.get_sample_index(path_to_video_dir)
    # Open the video file
    video = cv2.VideoCapture(path_to_video_file)
    if not video.isOpened():
        print(f'Error in split the video file of sample_{sample_index}')
        return 0
    frame_rate = iterate_over_frames(video, path_to_video_dir)
    metadata_df[sample_index, 'frame_rate'] = frame_rate
    video.release()
    if delete_video:
        remove(path_to_video_file)
    return metadata_df

def split_all_videos(path_to_data: str, delete_video: bool = False):
    """ The function iterate all video directories corresponding each
    sample and calls split_video_to_frames function. the function save
    the last metadata dataframe after saving all frames rates for each
    video"""

    metadata_df = pd.read_excel(metadata_file)
    for path_to_video_file in video_folder_iterator(path_to_data):
        metadata_df = split_video_to_frames(path_to_video_file, metadata_df, delete_video)
    metadata_df.to_excel(metadata_file, index=False)



def split_audio_by_frame_rate(path_to_audio_file: str, frame_rate: int,
                              window_size: int, delete_input: bool = False):
    """takes an audio file and split it a different audio files s.t. for each
    video frme there is a "audio frame" in size window_size and the video frame
    "taken" from the middle of the audio frame, if delete_input the input path
    will be deleted"""

    waveform, sample_rate = torchaudio.load(path_to_audio_file)
    gap_between_frames = 1.0 / frame_rate
    num_slices = waveform.size(1) // gap_between_frames
    half_window_size = window_size/2
    for i in range(num_slices):
        mid = gap_between_frames + gap_between_frames*i
        start = max(0, mid - half_window_size)
        end = min(waveform.size(1), mid + half_window_size)
        slice = waveform[:, start,end]
        output_path = add_addition_to_path(path_to_audio_file, f"a_{i}")
        torchaudio.save(output_path, slice,  sample_rate)
    if delete_input:
        os.remove(path_to_audio_file)

def split_all_audio(path_to_data: str,path_to_md: str, window_size:int,
                    delete_input: bool = False):
    data_md = pd.read_csv(path_to_md)
    for path_to_audio_folder in audio_folder_iterator(path_to_data):
        sample_index = du.get_sample_index(path_to_audio_folder)
        video_frame_rate = du.get_video_frame_rate(data_md, sample_index)
        if video_frame_rate:
            for path_to_audio_file in file_iterator_by_path(path_to_audio_folder):
                split_audio_by_frame_rate(path_to_audio_file, video_frame_rate,
                                          window_size, delete_input)
        else:
            print(f'Error in split the video file of {sample_index}, '
                  f'got frame_rate {video_frame_rate}')




