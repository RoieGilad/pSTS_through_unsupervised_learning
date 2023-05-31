from PIL import Image
from mtcnn_pytorch.src import detect_faces
import torchvision.transforms as T
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
    transform = T.RandomCrop((222, 222))
    img = Image.open(path_to_image)
    num_tries = 50
    img_to_run = img
    for i in range(num_tries):
        try:
            errors.append(1)
            bounding_boxes, _ = detect_faces(img_to_run)
            errors.append(2)
            if len(bounding_boxes) > 0:
                cropped_img = img_to_run.crop(bounding_boxes[0][:4])
                errors.append(3)
                if not override:
                    path_to_image = add_addition_to_path(path_to_image,
                                                         "centered")
                cropped_img.save(path_to_image)
                return 1
        except:
            if i != num_tries - 1:
                print(i, path_to_image, errors)
                errors = []
                img_to_run = transform(img)
            else:
                print("Failed!", path_to_image)
                return 0


def center_faces_by_folder(vf, override=True):
    cnt_all, cnt_success = 0, 0
    for jpeg_path in file_iterator_by_type(vf, "jpg"):
        if "centered" not in jpeg_path:
            cnt_all += 1
            cnt_success += center_face_by_path(jpeg_path, override)
    if cnt_all / cnt_success > 0.15:
        return [du.get_sample_index(vf)]
    return []


def delete_samples(root_dir:str, to_delete: list[str]):
    to_delete = set(to_delete)
    for sample_dir in folder_iterator_by_path(root_dir):
        if du.get_sample_index(sample_dir) in to_delete:
            remove(sample_dir)
    md_path = du.get_label_path(root_dir)
    data_md = pd.read_excel(md_path)
    to_drop = []
    for i, row in data_md.iterrows():
        if 'sample_' + row['sample_index'] in to_delete:
            to_drop.append(i)
    data_md.drop(to_drop.index[to_drop])


def center_all_faces(root_dir: str, override=True):
    """given root dir, center all the images in the sub video folders
    when override is True and the output is saved under a new name"""
    samples_to_delete = []
    for vf in tqdm(video_folder_iterator(root_dir), desc="Center Videos:"):
        samples_to_delete.extend(center_faces_by_folder(vf, override))
    print("done centering, the following samples should be deleted:")
    print(samples_to_delete)
    delete_samples(root_dir, samples_to_delete)




file_path = r'pSTS_through_unsupervised_learning\MTCNN\text'

if __name__ == '__main__':
    # center_faces_by_folder(
    #     r'C:\Users\AVIV\Roie\Galit_Project\pSTS_through_unsupervised_learning\demo_data\demo_after_flattening\sample_21\video',
    #     False)
    #

    before = len(glob.glob(path.join(destination_dir, "**", "*." + "jpg"),
                           recursive=True))
    print("before " + str(before))
    center_all_faces(destination_dir, False)
    after = len(glob.glob(path.join(destination_dir, "**", "*." + "jpg"),
                          recursive=True))
    print("after " + str(after))
    print("diff " + str(2 * before - after))
