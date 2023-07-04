import data.data_prep as dp
import data.data_utils as du
import data.dataset_types as dt
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


def check_video_data_set(index, type):
    data_object = None
    root_dir = r"../demo_data/after_flattening"
    path_to_labels = r"../demo_data/after_flattening/data_md.xlsx"
    if type == "video":
        data_object = dt.VideoDataset(root_dir, path_to_labels, du.train_v_frame_transformer,
                                     du.train_end_v_frame_transformer, du.train_video_transformer)
    elif type == "audio":
        data_object = dt.AudioDataset(root_dir, path_to_labels, du.train_a_frame_transformer,
                                      du.train_end_a_frame_transformer, du.train_audio_transformer)

    print(f"{type} Object: ", data_object)
    print("Num of samples: ", len(data_object))
    data_object.is_available(index)
    print(f"Label of sample {index} is: ", data_object.get_label(index))
    processed_frames, label = data_object[index]
    print(f"Number of frames in sample {index}: ", len(processed_frames))
    image_tensor = processed_frames[5]
    normalized_image = (image_tensor - image_tensor.min()) / \
                       (image_tensor.max() - image_tensor.min())
    image_array = normalized_image.numpy()
    image_array = image_array.transpose(1, 2, 0)
    plt.imshow(image_array)
    plt.show()



if __name__ == '__main__':
    # dp.windows = True
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
    # dp.split_all_audio(destination_dir_mini, 0, True)
    check_video_data_set(4, "audio")
