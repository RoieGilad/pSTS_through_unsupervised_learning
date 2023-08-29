import albumentations as A
import cv2
import random
from matplotlib import pyplot as plt
import os

random.seed(42)


def visualize_aug(image, aug_image):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(image)
    ax[1].imshow(aug_image)
    plt.show()

def augment_image(image, transform):
    transformed = transform(image=image)
    return transformed['image']




if '__main__' == __name__:
    data_dir = r'C:\Users\shiri\Documents\School\Master\Research\Face_recognition\Data\VGG-Face2\MTCNN\train_mtcnn_aligned_2000\n001744'
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.3),
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0.0),
    ])

    for img in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir, img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        aug_image = augment_image(img, transform)
        visualize_aug(img, aug_image)
