import glob
from os import path
from os import makedirs
import shutil
import openpyxl


audio_source_dir = "audio_dir_sample_17_before_flattening"
video_source_dir = "video_dir_sample_17_before_flattening"
destination_dir = "sample_17_after_flattening"
labels_file = "../demo_data/labels.xlsx"


wb = openpyxl.Workbook()
sheet = wb.active

#TODO understand the speceific audio/video_source_dir

def data_flattening(source_dir, destination_dir, file_extension, type, write_labels):
    """ This function iterate source dir of voxceleb2 dataset
    and create samples directory with audio and video subdirectories.
    the function create xl file saving the sample label - speaker id."""
    id_directories = glob.glob(path.join(source_dir, "id*"))
    sample_num = 0
    row = 1
    # Iterate over id directories of audio/video data
    for id_directory in id_directories:
        speaker_id = path.basename(id_directory)

        # Gets all audio/video sample corresponding speaker_id
        id_samples = glob.glob(path.join(id_directory, '*', f'*{file_extension}'))
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
                sheet[f'A{row}'] = sample_num
                sheet[f'B{row}'] = speaker_id
                row += 1
            sample_num += 1

    if write_labels:
        wb.save(labels_file)

if __name__ == "__main__":
    data_flattening(audio_source_dir, destination_dir, '.m4a', 'audio', True)
    data_flattening(video_source_dir, destination_dir, '.mp4', 'video', False)





