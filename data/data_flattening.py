import glob
from os import path
from os import makedirs
import shutil


audio_source_dir = "path/voxceleb2_audio"
video_source_dir = "path/voxceleb2_video"
destination_dir = "path/voxceleb2"

#TODO understand the speceific audio/video_source_dir

def data_flattening(source_dir, destination_dir, file_extension, type):
    """ This function iterate source dir of voxceleb2 dataset
    and reorder id's and scenes of the speaker, in order to merge
    audio and video data corresponding same speaker and scenes"""
    id_directories = glob.glob(path.join(source_dir, 'id*'))
    speaker_id = 0
    # Iterate over id directories of audio/video data
    for id_directory in id_directories:
        destination_speaker_path = path.join(destination_dir, f'id_{speaker_id}')
        makedirs(destination_speaker_path, exist_ok=True)

        # Gets scenes directories corresponding id_directory
        scene_directories = glob.glob(path.join(id_directory, '*'))
        # Iterate over scenes
        scene_num = 0
        for scene_directory in scene_directories:
            destination_scene_path = path.join(destination_speaker_path, f'scene_{scene_num}')
            makedirs(destination_scene_path, exist_ok=True)

            # Iterate all audio/video (=type) files in scene directory
            file_num = 0
            files = glob.glob(path.join(scene_directory, f'*{file_extension}'))
            for file in files:
                shutil.copy(file, path.join(destination_scene_path, f'{type}_{file_num}{file_extension}'))
                file_num += 1
            scene_num += 1
        speaker_id += 1


data_flattening(audio_source_dir, destination_dir, '.m4a', 'audio')
data_flattening(video_source_dir, destination_dir, '.mp4', 'video')





