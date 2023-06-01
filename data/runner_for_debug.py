import data.data_prep as dp
import data.data_utils as du
import data.dataset_types as dt
import os
import warnings
import cv2
import soundfile as sf

audio_source_dir = os.path.join("demo_data", "audio_before_flattening")
video_source_dir = os.path.join("demo_data", "video_before_flattening")
destination_dir = os.path.join("demo_data", "demo_after_flattening")

audio_source_dir_mini = os.path.join("demo_data", "audio_before_flattening_mini")
video_source_dir_mini = os.path.join("demo_data", "video_before_flattening_mini")
destination_dir_mini = os.path.join("demo_data", "demo_after_flattening_mini")

if __name__ == '__main__':
    dp.windows = True
    # dp.data_flattening(video_source_dir, audio_source_dir, destination_dir,
    #                    False)
    # dp.split_all_videos(destination_dir, True)
    # dp.center_all_faces(destination_dir, True)
    # dp.split_all_audio(destination_dir, 0, True)
    # dp.concatenate_all_audio(destination_dir)

    # dp.data_flattening(video_source_dir_mini, audio_source_dir_mini, destination_dir_mini,
    #                    False)
    # dp.split_all_videos(destination_dir_mini, True)
    # dp.center_all_faces(destination_dir_mini, True)
    dp.split_all_audio(destination_dir_mini, 0, True)

