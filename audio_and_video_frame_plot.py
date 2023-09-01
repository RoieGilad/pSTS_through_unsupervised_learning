from PIL import Image
from data_processing import data_utils as du
from preprocessing_script import plot_processed_frame, plot_spectrogram
import torchaudio


def create_audio_and_video_frame_plots(video_frame_path, audio_frame_path):
    # video
    video_frame = Image.open(video_frame_path)
    video_frame = du.val_v_frame_transformer(video_frame)
    plot_processed_frame(video_frame)

    # audio
    audio_frame = torchaudio.load(audio_frame_path)[0]
    audio_frame = du.val_a_frame_transformer(audio_frame)
    plot_spectrogram(audio_frame)


if __name__ == '__main__':
    create_audio_and_video_frame_plots("sample_18_v_0_12.jpg", "sample_18_a_2.wav")


