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

if __name__ == '__main__':
    dp.data_flattening_env(audio_source_dir, video_source_dir, destination_dir)
    dp.split_all_videos(destination_dir, False)
    dp.split_all_audio(destination_dir, 0, False)
    dp.concatenate_all_audio(destination_dir)