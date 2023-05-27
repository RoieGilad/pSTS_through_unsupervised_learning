from os import path
from PIL import Image
import data_utils as du
from MTCNN.mtcnn_pytorch.src import detect_faces
from torch import torchaudio


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


def split_audio_by_frame_rate(path_to_audio: str, frame_rate: int,
                              window_size: int, delete_input: bool = False):
    """takes an audio file and split it a different audio files s.t. for each
    video frme there is a "audio frame" in size window_size and the video frame
    "taken" from the middle of the audio frame, if delete_input the input path
    will be deleted"""
    waveform, sample_rate = torchaudio.load(path_to_audio)
    gap_between_frames = 1.0 / frame_rate
    num_slices = waveform.size(1) // gap_between_frames
    half_window_size = window_size/2
    for i in range(num_slices):
        mid = gap_between_frames + gap_between_frames*i
        start = max(0, mid - half_window_size)
        end = min(waveform.size(1), mid + half_window_size)
        slice = waveform[:, start,end]
        output_path = add_addition_to_path(path_to_audio, f"a_{i}")
        torchaudio.save(output_path, slice,  sample_rate)
    if delete_input:
        os.remove(path_to_audio)

def split_all_audio(path_to_data: str,path_to_md: str, window_size:int,
                    delete_input: bool = False):
    data_md = pd.read_csv(path_to_md)
    for path_to_audio_folder in audio_folder_iterator(path_to_data):
        sample_id = du.get_sample_id(path_to_audio_folder)
        video_frame_rate = du.get_video_frame_rate(data_md, sample_id)
        if video_frame_rate:
            for path_to_audio in file_iterator_by_path(path_to_audio_folder):
                split_audio_by_frame_rate(path_to_audio, video_frame_rate,
                                          window_size, delete_input)
        else:
            print(f'Error in split the video file of {sample_id}, '
                  f'got frame_rate {video_frame_rate}')


