import os
from glob import glob
from os import path

import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from natsort import natsorted
from torch import Tensor, zeros, device
from torchvision import transforms
from tqdm.notebook import tqdm


def get_video_embedding(video_imgs_dir: str, mtcnn, model):
    output = []
    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    for im_path in natsorted(glob(path.join(video_imgs_dir, '*.jpg'))):
        print(im_path)
        img = Image.open(im_path)
        if img.mode != 'RGB':  # PNG imgs are RGBA
            img = img.convert('RGB')
        img = mtcnn(img)
        if img is not None:
            img = img.cuda()
            img = img / 255.0
            img = normalize_imagenet(img).to(device('cuda:0')).unsqueeze(
                0).float()
            output.append(model(img).cpu())
        else:
            print(f'failed mtcnn in {im_path}')

    #add mean frame
    stacked_frames_embedding = torch.stack(output)
    output.append(torch.mean(stacked_frames_embedding, dim=0))
    return output


def get_all_video_embeddings(videos_dir:str, mtcnn, model):
    all_video_embeddings = []
    video_dirs = [subdir for subdir in glob(os.path.join(videos_dir, '*'))
                  if os.path.isdir(subdir)]
    for video_dir in tqdm(video_dirs):
        embeddings = get_video_embedding(video_dir, mtcnn, model)
        embeddings.append(os.path.basename(video_dir))
        all_video_embeddings.append(embeddings)
    return all_video_embeddings


def get_face_recognition_embeddings(videos_dir: str):
    mtcnn = MTCNN(image_size=160, post_process=False,
                  device='cuda:0')  # .cuda()
    model = InceptionResnetV1(pretrained='vggface2', device='cuda:0').eval()
    return get_all_video_embeddings(videos_dir, mtcnn, model)

if __name__ == '__main__':
    print(get_face_recognition_embeddings(r'demo_data/rsa_video'))



