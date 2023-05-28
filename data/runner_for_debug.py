import data_prep as dp
import data_utils as du
import dataset_types as dt

audio_source_dir = "../demo_data/audio_before_flattening"
video_source_dir = "../demo_data/video_before_flattening"
destination_dir = "../demo_data/demo_after_flattening"

if __name__ == '__main__':
    dp.data_flatenning_env(audio_source_dir, video_source_dir, destination_dir)
    # dp.split_all_videos(destination_dir, False)
    # dp.center_all_faces(destination_dir, True)
    # dp.split_all_audio(destination_dir, 1/25, False)
