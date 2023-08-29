import glob
import os
import shutil
import time
from os import path, remove
import random


def prepare_data_for_preprocessing(src_dir):
    video_dest = "stimuli_video_data"
    audio_dest = "stimuli_audio_data"
    os.makedirs(video_dest, exist_ok=True)
    os.makedirs(audio_dest, exist_ok=True)

    video_samples = sorted(
        glob.glob(path.join(src_dir, '*.mp4')))
    print(video_samples)
    audio_samples = sorted(
        glob.glob(path.join(src_dir, '*.wav')))
    for video_file, audio_file in zip(video_samples, audio_samples):
        video_file_name = video_file.split("\\")[1]
        audio_file_name = audio_file.split("\\")[1]
        id = "id_" + video_file_name.split("_")[0]
        sample_number = video_file_name.split("_")[1].split(".")[0]
        id_video_dir = os.path.join(video_dest, id)
        id_audio_dir = os.path.join(audio_dest, id)

        os.makedirs(id_video_dir, exist_ok=True)
        os.makedirs(id_audio_dir, exist_ok=True)

        # Create the sample directory inside the ID directory
        video_num_dir = os.path.join(id_video_dir, f"video_{sample_number}")
        audio_num_dir = os.path.join(id_audio_dir, f"audio_{sample_number}")
        print(video_num_dir)

        os.makedirs(video_num_dir, exist_ok=True)
        os.makedirs(audio_num_dir, exist_ok=True)

        # Move the file to the sample directory
        shutil.move(os.path.join(src_dir, video_file_name), os.path.join(video_num_dir, video_file_name))
        shutil.move(os.path.join(src_dir, audio_file_name), os.path.join(audio_num_dir, audio_file_name))

def create_vox_samples_dir_for_rsa(test_dir):
    destination_directory = 'vox_samples_rsa'
    num_samples_to_copy = 30

    # List all files in the source directory
    all_files = os.listdir(test_dir)

    # Randomly select num_samples_to_copy files
    selected_files = random.sample(all_files, num_samples_to_copy)

    # Copy the selected files to the destination directory
    for file_name in selected_files:
        source_path = os.path.join(test_dir, file_name)
        destination_path = os.path.join(destination_directory, file_name)
        shutil.copy(source_path, destination_path)
        print(f"Copying {file_name} to destination.")

    print("Copying completed.")


if __name__ == '__main__':
    #prepare_data_for_preprocessing("stimuli")
    create_vox_samples_dir_for_rsa("dataset/160k_train_500ms")
    
