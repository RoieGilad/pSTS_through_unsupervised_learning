import glob
from os import path
from os import makedirs
import shutil
import openpyxl


audio_source_dir = "path/voxceleb2_audio"
video_source_dir = "path/voxceleb2_video"
destination_dir = "path/voxceleb2"
labels_file = "path/labels.xlsx"


wb = openpyxl.Workbook()
sheet = wb.active
sheet['A1'] = 'Sample_index'
sheet['B1'] = 'Speaker_id'

#TODO understand the speceific audio/video_source_dir

def data_flattening(source_dir, destination_dir, file_extension, type, write_labels):
    """ This function iterate source dir of voxceleb2 dataset
    and create samples directory with audio and video subdirectories.
    the function create xl file saving the sample label - speaker id."""
    id_directories = glob.glob(path.join(source_dir, 'id*'))
    sample_num = 0
    row = 2
    # Iterate over id directories of audio/video data
    for id_directory in id_directories:
        speaker_id = path.basename(id_directory)

        # Gets all audio/video sample corresponding speaker_id
        id_samples = glob.glob(path.join(id_directory, '*', f'*{file_extension}'))
        # Iterate over samples
        for id_sample in id_samples:
            # Creating the sample directories
            destination_sample_path = path.join(destination_dir, f'sample_{sample_num}')
            makedirs(path.join(destination_sample_path, f'{type}'), exist_ok=True)
            # Copy samples file from source
            shutil.copy(id_sample, path.join(destination_sample_path, f'sample_{sample_num}{file_extension}'))
            # Writing label for corresponding sample into xl file
            if write_labels:
                sheet[f'A{row}'] = sample_num
                sheet[f'B{row}'] = speaker_id
                row += 1
            sample_num += 1


data_flattening(audio_source_dir, destination_dir, '.m4a', 'audio', True)
data_flattening(video_source_dir, destination_dir, '.mp4', 'video', False)





