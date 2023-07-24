from data_processing import data_prep as dp
from data_processing import dataset_types as dt
import matplotlib.pyplot as plt
from torchvision import transforms
from glob import glob
import soundfile as sf
from data_processing import data_utils as du
import torchaudio
import glob
import os
from os import path
from torchvision.transforms import functional as F
import torchaudio.transforms as a_transforms
from natsort import natsorted
import torch

audio_source_dir = os.path.join("demo_data", "audio_before_flattening")
video_source_dir = os.path.join("demo_data", "video_before_flattening")
destination_dir = os.path.join("demo_data", "demo_after_flattening")

audio_source_dir_mini = os.path.join("demo_data",
                                     "audio_before_flattening_mini")
video_source_dir_mini = os.path.join("demo_data",
                                     "video_before_flattening_mini")
destination_dir_mini = os.path.join("demo_data", "demo_after_flattening_mini")


def plot_processed_frame(image_tensor):
    normalized_image = (image_tensor - image_tensor.min()) / \
                       (image_tensor.max() - image_tensor.min())
    image_array = normalized_image.numpy()
    image_array = image_array.transpose(1, 2, 0)
    plt.imshow(image_array)
    plt.show()


def plot_spectrogram(spectrogram, title):
    spectrogram = spectrogram.squeeze(dim=0)
    plt.figure()
    plt.imshow(spectrogram.log2(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Frame')
    plt.ylabel('Frequency bin')
    plt.title('Spectrogram: '+title)
    plt.show()


def plot_two_spectrograms(spectrogram1, title1, spectrogram2, title2):
    spectrogram1 = spectrogram1.squeeze(dim=0)
    spectrogram2 = spectrogram2.squeeze(dim=0)

    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

    plt.subplot(1, 2, 1)
    plt.imshow(spectrogram1.log2(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Frame')
    plt.ylabel('Frequency bin')
    plt.title('Spectrogram: ' + title1)

    plt.subplot(1, 2, 2)
    plt.imshow(spectrogram2.log2(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Frame')
    plt.ylabel('Frequency bin')
    plt.title('Spectrogram: ' + title2)

    plt.tight_layout()  # Ensures the plots don't overlap
    plt.show()


def check_data_set(index, type):
    data_object = None
    root_dir = r"demo_data/demo_after_flattening_mini"
    path_to_labels = r"demo_data/demo_after_flattening_mini/data_md.xlsx"
    if type == "video":
        data_object = dt.VideoDataset(root_dir, path_to_labels,
                                      du.train_v_frame_transformer,
                                      du.train_end_v_frame_transformer,
                                      du.train_video_transformer)
    elif type == "audio":
        data_object = dt.AudioDataset(root_dir, path_to_labels,
                                      du.train_a_frame_transformer,
                                      du.train_end_a_frame_transformer,
                                      du.train_audio_transformer)
    elif type == "combined":
        transforms_dict = {'a_frame_transform': du.train_a_frame_transformer,
                           'end_a_frame_transform': du.train_end_a_frame_transformer,
                           'a_batch_transform': du.train_audio_transformer,
                           'v_frame_transform': du.train_v_frame_transformer,
                           'end_v_frame_transform': du.train_end_v_frame_transformer,
                           'v_batch_transform': du.train_video_transformer, }
        data_object_combined = dt.CombinedDataset(root_dir, path_to_labels,
                                                  transforms_dict)

    if type == "combined":
        print(f"{type} Object: ", data_object_combined)
        print("Num of samples: ", len(data_object_combined))
        print(f"Label of sample {index} is: ",
              data_object_combined.get_label(index))
        processed_video_frames, processed_audio_frames, label = \
            data_object_combined[index]
        for video_frame, audio_frame in zip(processed_video_frames,
                                            processed_audio_frames):
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
        print(frame.size())
    # Spectrogram plot for debug
    else:
        print(frame.size())
        print(frame)
        frame = F.to_pil_image(frame)
        print(frame)
        frame = transforms.ToTensor()(frame)
        print(frame)
        plot_processed_frame(frame)
        # plot_spectrogram(frame)


def checks_audio_after_transform(path_to_sample):
    modes = ['nearest', 'bilinear', 'bicubic', 'area',
             'nearest-exact']
    align_corners = [None, True, True, None, None]
    path_to_frames = natsorted(
        glob.glob(path.join(path_to_sample, "audio", "*.wav")))
    frames = [torchaudio.load(p) for p in path_to_frames]
    sample_rate = frames[0][1]

    spectrograms = [a_transforms.Spectrogram(n_fft=256, hop_length=16)(frame[0])
                    for frame in frames]
    for spectrogram in spectrograms:
        spectrogram = spectrograms[39]
        # plot_spectrogram(spectrogram, "before")
        target_size = (224, 224)
        for mode, ac in zip(modes, align_corners):
            input = spectrogram.unsqueeze(0) if mode != 'linear' else spectrogram
            new_spectrogram = torch.nn.functional.interpolate(
                input, size=target_size, mode=mode,
                align_corners=ac)
            new_spectrogram = new_spectrogram.squeeze(dim=0)
            plot_two_spectrograms(spectrogram, "before", new_spectrogram, mode)
            plot_spectrogram(spectrogram,mode)
            print(spectrogram.size())
            print(new_spectrogram)
            print(new_spectrogram.size())
        return

        # spectrogram_pil = F.to_pil_image(spectrogram)
        # spectrogram_tensor = transforms.ToTensor()(spectrogram_pil)
        # mask = spectrogram_tensor == 0
        #
        # Zero out the corresponding values in spectrogram using the mask
        # spectrogram_zeroed = spectrogram * (~mask)
        # spectrogram_zeroed = spectrogram_zeroed.to(torch.cdouble)
        #
        # waveform = a_transforms.InverseSpectrogram(n_fft=256, hop_length=16)(spectrogram_zeroed)
        # waveform = waveform.to(torch.float32)
        #
        # torchaudio.save(f'{path_to_sample}/audio/after_{i}.wav', waveform, sample_rate)
        # i += 1


def concatinate_wav_files(path_to_files):
    input_files = natsorted(
        glob.glob(path.join(path_to_files, "audio", "after*")))
    output_file = "output.wav"

    # Initialize an empty audio array to store the concatenated audio
    audio_data = []

    # Read each input WAV file and append its audio data to the array
    for file in input_files:
        data, sr = sf.read(file)
        audio_data.extend(data)

    output_path = os.path.join(path_to_files, output_file)

    # Write the concatenated audio data to the output WAV file
    sf.write(output_path, audio_data, sr)


def prepare_data():
    # dp.data_flattening(video_source_dir, audio_source_dir, destination_dir,
    #                    False)
    # dp.split_all_videos(destination_dir, True)
    dp.center_all_faces(destination_dir, True)
    # dp.split_all_audio(destination_dir, 100, True)
    dp.cuda = True


if __name__ == '__main__':
    dp.windows = True
    dp.cuda = False

    checks_audio_after_transform(
        "demo_data\demo_after_flattening\sample_0")
    # concatinate_wav_files("demo_data\demo_after_flattening_mini\sample_0")
    # dp.data_flattening(video_source_dir, audio_source_dir, destination_dir,
    #                    False)
    # dp.split_all_videos(destination_dir, True)
    # dp.center_all_faces(destination_dir, True)
    # dp.split_all_audio(destination_dir, 100, True)
    # check_data_set(4, "audio")
