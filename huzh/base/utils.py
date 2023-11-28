import os
import cv2 as cv


def get_all_files(root_path):
    all_file_path = []
    for it in os.listdir(root_path):
        current_path = os.path.join(root_path, it)
        if os.path.isdir(current_path):
            all_file_path.extend(get_all_files(current_path))
        elif os.path.isfile(current_path):
            all_file_path.append(current_path)
    return all_file_path


def get_video_info(vd_reader):
    im_width = int(vd_reader.get(cv.CAP_PROP_FRAME_WIDTH))
    im_height = int(vd_reader.get(cv.CAP_PROP_FRAME_HEIGHT))
    count = int(vd_reader.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(vd_reader.get(cv.CAP_PROP_FPS))
    channels = int(vd_reader.get(cv.CAP_PROP_VIDEO_TOTAL_CHANNELS))
    return (im_width, im_height, count, fps, channels)


def get_specify_files(
    input_path,
    suffixes=None,
):
    return [
        it
        for it in get_all_files(input_path)
        if os.path.splitext(it)[1] in suffixes
    ]
