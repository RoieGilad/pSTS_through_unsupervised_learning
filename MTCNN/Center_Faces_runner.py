from PIL import Image
from mtcnn_pytorch.src import detect_faces
import os
import glob
from os import path
from tqdm import tqdm

destination_dir = os.path.join("/demo_data", "demo_after_flattening")
destination_dir = r'C:\Users\AVIV\Roie\Galit_Project\pSTS_through_unsupervised_learning\demo_data\demo_after_flattening'


def add_addition_to_path(input_path, addition):
    """ for a given path = dirname/base_name.type
    will return the new path: dirname/base_name_addition.type"""
    dirname, base_name = path.split(input_path)
    file_name, file_ext = path.splitext(base_name)
    base_name = "".join([file_name, "_", addition, file_ext])
    return path.join(dirname, base_name)


def video_folder_iterator(root_dir: str):
    """return all the file paths matching the following pattern
    samples_root_dir/*/video, in-order"""
    for p in folder_iterator_by_path(path.join(root_dir, "*", "video")):
        yield p


def folder_iterator_by_path(root_dir: str):
    """yield the folder path of the next sample, in order"""
    for p in sorted(glob.glob(root_dir, recursive=False)):
        if not path.isfile(p):
            yield p


def file_iterator_by_type(root_dir: str, type: str):
    """return all path of the files in the root_dir from the type is given
     as input, in-order"""
    for p in sorted(glob.glob(path.join(root_dir, "*." + type))):
        yield p


def center_face_by_path(path_to_image, override=True):
    errors = []
    # try:
    img = Image.open(path_to_image)
    errors.append(1)
    bounding_boxes, _ = detect_faces(img)
    errors.append(2)
    if len(bounding_boxes) > 0:
        cropped_img = img.crop(bounding_boxes[0][:4])
        errors.append(3)
        if not override:
            path_to_image = add_addition_to_path(path_to_image, "centered")
        cropped_img.save(path_to_image)
        errors.append(4)
    # except:
    #     print(errors)


def center_all_faces(root_dir: str, override=True):
    """given root dir, center all the images in the sub video folders
    when override is True and the output is saved under a new name"""
    for vf in tqdm(video_folder_iterator(root_dir), desc="Center Videos:"):
        for jpeg_path in file_iterator_by_type(vf, "jpg"):
            if "centered" not in jpeg_path:
                center_face_by_path(jpeg_path, override)


file_path = r'pSTS_through_unsupervised_learning\MTCNN\text'

def count_occurrences(text, substring):
    count = 0
    start = 0
    while True:
        start = text.find(substring, start) + 1
        if start > 0:
            count += 1
        else:
            break
    return count

if __name__ == '__main__':
    before = len(glob.glob(path.join(destination_dir, "**", "*." + "jpg"), recursive=True))
    print("before " + str(before))
    center_all_faces(destination_dir, False)
    after = len(glob.glob(path.join(destination_dir, "**", "*." + "jpg"), recursive=True))
    print("after " + str(after))
    print("diff " + str(2*before - after))

    # try:
    #     with open(file_path, 'r') as file:
    #         text = file.read()
    #         substring = "[1]"
    #         occurrences = count_occurrences(text, substring)
    #         print(f"Number of occurrences of '{substring}': {occurrences}")
    # except FileNotFoundError:
    #     print("File not found. Please check the file path and try again.")

