import os
import torch
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment


def process_audio(input_path, output_dir, name, audio_transform=None, spectrogram_transform=None):
    spec_transform = T.Spectrogram()
    inv_spec_transform = T.InverseSpectrogram()

    waveform, sample_rate = open_audio_file(input_path)
    if audio_transform:
        waveform = audio_transform(waveform)
    # spectrogram = spec_transform(waveform)
    # if spectrogram_transform:
    #     spectrogram = spectrogram_transform(spectrogram)
    # waveform = inv_spec_transform(spectrogram)

    output_path = os.path.join(output_dir, name + '.wav')
    torchaudio.save(output_path, waveform, sample_rate)
    return output_path


def add_gaussian_white_noise(waveform, noise_level=0.01):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise


def convert_mp4a_to_wav(path_to_audio_file):
    wav_filename = path_to_audio_file[:-3] + "wav"
    track = AudioSegment.from_file(path_to_audio_file, format='m4a')
    file_handle = track.export(wav_filename, format='wav')
    return wav_filename

def open_audio_file(path):
    path = convert_mp4a_to_wav(path)
    audio = AudioSegment.from_wav(path)
    audio.export(path, format="wav")
    return torchaudio.load(path)
if __name__ == "__main__":
    input_audio_path = r'demo_data/audio_before_flattening/id00017/01dfn2spqyE/00001.m4a'

    output_directory = r'output_audio_research'
    waveform, sample_rate = open_audio_file(input_audio_path)

    noise_transform = add_gaussian_white_noise
    amplitude_vol_high = T.Vol(gain=3, gain_type="amplitude")
    amplitude_vol_low = T.Vol(gain=0.1, gain_type="amplitude")

    process_audio(input_audio_path, output_directory, r'with_noise', noise_transform, None)

    process_audio(input_audio_path, output_directory, r'amplitude_vol_high', amplitude_vol_high, None)
    process_audio(input_audio_path, output_directory, r'amplitude_vol_low', amplitude_vol_low, None)



