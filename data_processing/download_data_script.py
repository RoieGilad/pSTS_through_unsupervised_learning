import requests
import os
from os import system
def download_videos(video_dir, base_url):
    for part in range(ord('a'), ord('i') + 1):
        current_url = f"{base_url}{chr(part)}"
        save_filename = os.path.join(video_dir, f"vox2_dev_mp4_parta{chr(part)}")
        download_file(current_url, save_filename)
        print(f"finish download part {part} video")


def download_audios(audio_dir, base_url):
    for part in range(ord('a'), ord('h') + 1):
        current_url = f"{base_url}{chr(part)}"
        save_filename = os.path.join(audio_dir, f"vox2_dev_aac_parta{chr(part)}")
        download_file(current_url, save_filename)
        print(f"finish download part {part} audio")

def download_test(test_dir, base_url):
    save_audio_filename = os.path.join(test_dir, "vox2_test_aac.zip")
    save_video_filename = os.path.join(test_dir, "vox2_test_mp4.zip")
    current_audio_test_url = f"{base_url}_aac.zip"
    current_video_test_url = f"{base_url}_mp4.zip"
    download_file(current_audio_test_url, save_audio_filename)
    download_file(current_video_test_url, save_video_filename)

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


if __name__ == "__main__":
    base_video_url = "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_parta"
    base_audio_url = "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_parta"
    base_test_url = "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test"
    dataset_dir = "/dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    os.chmod(dataset_dir, 0o0777)
    video_dataset_dir = "/dataset/video"
    os.makedirs(video_dataset_dir, exist_ok=True)
    os.chmod(video_dataset_dir, 0o0777)
    audio_dataset_dir = "/dataset/audio"
    os.makedirs(audio_dataset_dir, exist_ok=True)
    os.chmod(audio_dataset_dir, 0o0777)
    test_dataset_dir = "/dataset/test"
    os.makedirs(test_dataset_dir, exist_ok=True)
    os.chmod(test_dataset_dir, 0o0777)

    download_audios(audio_dataset_dir, base_audio_url)
    download_videos(video_dataset_dir, base_video_url)
    download_test(test_dataset_dir, base_test_url)

    #system('cat /dataset/video/vox2_dev_mp4* > vox2_mp4.zip')
    #system('cat /dataset/audio/vox2_dev_aac* > vox2_aac.zip')
    #system('unzip vox2_mp4.zip')
    #system('unzip vox2_aac.zip')


