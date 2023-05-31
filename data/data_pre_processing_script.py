import sys
from . import data_prep
from . import dataset_types
from . import data_utils
from os import path


def read_and_check_arguments():
    video_source_dir, audio_source_dir, destination_dir = get_dirs_paths()
    run_all = checks_and_convert_bool_arguments(
        input("Enter y/N in order to run_all/run_part of data prep:"))
    while checks_and_convert_bool_arguments(run_all) == 2:
        run_all = checks_and_convert_bool_arguments(
            input("Enter y/N in order to run_all/run_part of data prep:"))
    save_curr_data = checks_and_convert_bool_arguments(
        input("Enter y/N in order to"
              " keep/remove data before splitting:"))
    while checks_and_convert_bool_arguments(save_curr_data) == 2:
        save_curr_data = checks_and_convert_bool_arguments(
            input("Enter y/N in order to"
                  " keep/remove data before splitting:"))

    return video_source_dir, audio_source_dir, destination_dir, save_curr_data, run_all


def get_dirs_paths():
    video_source_dir = input("Enter video source directory:")
    while not path.exists(video_source_dir):
        print("Path not valid")
        video_source_dir = input("Enter video source directory:")
    audio_source_dir = input("Enter audio source directory:")
    while not path.exists(audio_source_dir):
        print("Path not valid")
        audio_source_dir = input("Enter audio source directory:")
    destination_dir = input("Enter destination directory:")
    while not path.exists(destination_dir):
        print("Path not valid")
        destination_dir = input("Enter destination directory:")
    return video_source_dir, audio_source_dir, destination_dir


def which_part_to_run():
    run_flattening = checks_and_convert_bool_arguments(
        input("Run flattening data? y/N:"))
    while checks_and_convert_bool_arguments(run_flattening) == 2:
        run_flattening = checks_and_convert_bool_arguments(
            input("Run flattening data? y/N:"))

    run_centering = checks_and_convert_bool_arguments(
        input("Run centering data? y/N:"))
    while checks_and_convert_bool_arguments(run_centering) == 2:
        run_centering = checks_and_convert_bool_arguments(
            input("Run centering data? y/N:"))

    run_split_video = checks_and_convert_bool_arguments(
        input("Run split video? y/N:"))
    while checks_and_convert_bool_arguments(run_split_video) == 2:
        run_split_video = checks_and_convert_bool_arguments(
            input("Run split video? y/N:"))

    run_split_audio = checks_and_convert_bool_arguments(
        input("Run split audio? y/N:"))
    while checks_and_convert_bool_arguments(run_split_audio) == 2:
        run_split_audio = checks_and_convert_bool_arguments(
            input("Run split audio? y/N:"))

    return run_flattening, run_centering, run_split_video, run_split_audio


def checks_and_convert_bool_arguments(arg):
    if arg == 'y':
        return True
    elif arg == 'N':
        return False
    else:
        print("Char invalid.")
        return 2


def main():
    """ Script for pre-processing the data into an adaptive samples
    for the model"""
    video_source_dir, audio_source_dir, destination_dir, save_curr_data, run_all = read_and_check_arguments()

    if not run_all:
        run_flattening, run_centering, run_split_video, run_split_audio = which_part_to_run()
    else:
        run_flattening = run_centering = run_split_video = run_split_audio = True

    # Data flattening
    if run_flattening:
        data_prep.data_flattening(video_source_dir, audio_source_dir,
                                  destination_dir)

    # Split videos into frames
    if run_split_video:
        data_prep.split_all_videos(destination_dir, save_curr_data)

    # Center faces after split
    if run_centering:
        data_prep.center_all_faces(destination_dir, save_curr_data)

    # split audio corresponding video's frames
    if run_split_audio:
        data_prep.split_all_audio(destination_dir, 0, save_curr_data)


if __name__ == '__main__':
    data_prep.windows = False    # TODO change for GPU or GC
    main()
