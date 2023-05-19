from os import path
from PIL import Image
import data_utils as du
from MTCNN.mtcnn_pytorch.src import detect_faces


def center_face_by_path(path_to_image, override=True):
    img = Image.open(path_to_image)
    bounding_boxes, _ = detect_faces(img)
    cropped_img = img.crop(bounding_boxes)
    if not override:
        path_to_image = du.add_addition_to_path(path, "centered")
    cropped_img.save(path_to_image)


def center_all_faces(root_dir: str, override=True):
    """given root dir, center all the images in the sub video folders
    when override is True and the output is saved under a new name"""
    for video_folder in du.video_folder_iterator(root_dir):
        for jpeg_path in du.file_iterator_by_type(video_folder, "jpg"):
            center_face_by_path(jpeg_path, override)
