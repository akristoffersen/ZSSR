import GPUtil
import glob
import os
from utils import prepare_result_dir
import configs
from time import sleep
import sys
import run_ZSSR_single_input

import cv2
import argparse
import numpy as np


def main(conf_name, gpu):
    # will train on the first frame of the video, then run
    # inference on the rest of the frames.
    # optionally, we can also run zssr on the last image to
    # show the delta from the beginning to the end.

    if conf_name is None:
        conf = configs.Config()
    else:
        if conf_name == "X2_REAL_CONF_VIDEO":
            conf = configs.X2_REAL_CONF_VIDEO
        # elif conf_name == "X2_GRADUAL_IDEAL_CONF_VIDEO":
        #     conf = configs.X2_GRADUAL_IDEAL_CONF_VIDEO

    res_dir = prepare_result_dir(conf)
    local_dir = os.path.dirname(__file__)

    files = [file_path for file_path in glob.glob('%s/*.%s' % (conf.input_path, conf.input_file_ext))
             if not file_path[-7:-4] == '_gt']

    print("locations:", res_dir, local_dir)
    print("files:", files)

    for file_ind, input_file in enumerate(files):

        conf.name = input_file[:-4] + "_frame_1_2x2"

        vidcap = cv2.VideoCapture(input_file)

        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print("Number of frames: ", video_length)

        # frame 1:
        success, frame_one = vidcap.read()

        # train on frame one

        # TODO: move this conversion to run_ZSSR_single_input
        converted_frame_one = cv2.cvtColor(frame_one, cv2.COLOR_BGR2RGB)
        converted_frame_one = cv2.normalize(converted_frame_one, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        print(converted_frame_one)

        # TODO: not implementing ground truth at the moment.

        ground_truth_file = '0'
        image_size = frame_one.shape
        
        # MUST BE REVERSED FOR OPENCV STUFF
        new_image_size = (image_size[1] * 2, image_size[0] * 2)
        print("New Image Size:", new_image_size)

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("Video FPS:", fps)

        # TODO: not implementing kernels at the moment.

        # TODO: clustering for scene detection (?)
        # or use final_test() to check if you need to retrain the net.

        kernel_files_str = '0'
        net = run_ZSSR_single_input.main(converted_frame_one, ground_truth_file, kernel_files_str, gpu, conf, res_dir)

        video_name = input_file[:-4] + "_2x2" + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        new_vid = cv2.VideoWriter(video_name, fourcc, fps, new_image_size)

        count = 0
        image = None
        image_temp = frame_one

        while success:
            image = image_temp
            # convert to float32:
            image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # have to figure out how to do this
            # I think we have to use forward_pass() in ZSSR.py, but not sure.
            scaled_image = net.forward_pass(image)

            print('Inference on Frame: ', count, scaled_image.shape)

            if count % 100 == 0:
                print(scaled_image)

            # convert to something we can add to new_vid
            scaled_image = cv2.normalize(scaled_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR)

            # writing to new video
            new_vid.write(scaled_image)

            success, image_temp = vidcap.read()
            count += 1

        # train on the last frame
        new_vid.release()
        conf.name = input_file[:-4] + "_frame_last_2x2"
        run_ZSSR_single_input.main(image, ground_truth_file, kernel_files_str, gpu, conf, res_dir)


if __name__ == '__main__':
    conf_str = sys.argv[1] if len(sys.argv) > 1 else None
    gpu_str = sys.argv[2] if len(sys.argv) > 2 else None
    main(conf_str, gpu_str)
