from data_processing import data_prep as dp
from data_processing import data_utils as du
from data_processing import dataset_types as dt
import os
import warnings
import cv2
import soundfile as sf
import matplotlib.pyplot as plt

audio_source_dir = os.path.join("demo_data", "audio_before_flattening")
video_source_dir = os.path.join("demo_data", "video_before_flattening")
destination_dir = os.path.join("demo_data", "demo_after_flattening")

audio_source_dir_mini = os.path.join("demo_data", "audio_before_flattening_mini")
video_source_dir_mini = os.path.join("demo_data", "video_before_flattening_mini")
destination_dir_mini = os.path.join("demo_data", "demo_after_flattening_mini")


def plot_processed_frame(image_tensor):
    normalized_image = (image_tensor - image_tensor.min()) / \
                       (image_tensor.max() - image_tensor.min())
    image_array = normalized_image.numpy()
    image_array = image_array.transpose(1, 2, 0)
    plt.imshow(image_array)
    plt.show()


def plot_spectrogram(spectrogram):
    plt.figure()
    plt.imshow(spectrogram.log2(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Frame')
    plt.ylabel('Frequency bin')
    plt.title('Spectrogram')
    plt.show()


def check_data_set(index, type):
    data_object = None
    root_dir = r"demo_data/demo_after_flattening_mini"
    path_to_labels = r"demo_data/demo_after_flattening_mini/data_md.xlsx"
    if type == "video":
        data_object = dt.VideoDataset(root_dir, path_to_labels, du.train_v_frame_transformer,
                                      du.train_end_v_frame_transformer, du.train_video_transformer)
    elif type == "audio":
        data_object = dt.AudioDataset(root_dir, path_to_labels, du.train_a_frame_transformer,
                                      du.train_end_a_frame_transformer, du.train_audio_transformer)
    elif type == "combined":
        transforms_dict = {'a_frame_transform': du.train_a_frame_transformer,
                           'end_a_frame_transform': du.train_end_a_frame_transformer,
                           'a_batch_transform': du.train_audio_transformer,
                           'v_frame_transform': du.train_v_frame_transformer,
                           'end_v_frame_transform': du.train_end_v_frame_transformer,
                           'v_batch_transform': du.train_video_transformer, }
        data_object_combined = dt.CombinedDataset(root_dir, path_to_labels, transforms_dict)

    if type == "combined":
        print(f"{type} Object: ", data_object_combined)
        print("Num of samples: ", len(data_object_combined))
        print(f"Label of sample {index} is: ", data_object_combined.get_label(index))
        processed_video_frames, processed_audio_frames, label = data_object_combined[index]
        for video_frame, audio_frame in zip(processed_video_frames, processed_audio_frames):
            plot_processed_frame(video_frame)
            plot_spectrogram(audio_frame)
        return

    print(f"{type} Object: ", data_object)
    print("Num of samples: ", len(data_object))
    data_object.is_available(index)
    print(f"Label of sample {index} is: ", data_object.get_label(index))
    processed_frames, label = data_object[index]
    print(f"Number of frames in sample {index}: ", len(processed_frames))
    frame = processed_frames[5]
    if type == "video":
        plot_processed_frame(frame)
    # Spectrogram plot for debug
    else:
        plot_spectrogram(frame)


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
    # dp.split_all_audio(destination_dir_mini, 100, True)
    check_data_set(17, "combined")
