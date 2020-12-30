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
        vidcap = cv2.VideoCapture(conf.input_path + input_file)

        vidcap = cv2.VideoCapture(conf.input_path)
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print("Number of frames: ", video_length)

        # frame 1:
        success, frame_one = vidcap.read()

        # train on frame one

        # TODO: not implementing ground truth at the moment.

        ground_truth_file = '0'
        image_size = frame_one.shape
        new_image_size = (image_size[0] * 2, image_size[1] * 2)

        # TODO: not implementing kernels at the moment.

        # Numeric kernel files need to be like the input file with serial number
        # kernel_files = ['%s_%d.mat;' % (input_file[:-4], ind) for ind in range(len(conf.scale_factors))]
        # kernel_files_str = ''.join(kernel_files)
        # for kernel_file in kernel_files:
        #     if not os.path.isfile(kernel_file[:-1]):
        #         kernel_files_str = '0'
        #         print('no kernel loaded')
        #         break
        kernel_files_str = '0'

        net = run_ZSSR_single_input.main(frame_one, ground_truth_file, kernel_files_str, gpu, conf, res_dir)

        video_name = input_file[:-4] + "_2x2." + "mp4"
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # not sure what 20.0 is.
        new_vid = cv2.VideoWriter(video_name, fourcc, 20.0, new_image_size)

        count = 0
        image = None

        while success:
            success, image = vidcap.read()
            print('Inference on Frame: ', count)

            # have to figure out how to do this
            # I think we have to use forward_pass() in ZSSR.py, but not sure.

            scaled_image = net.forward_pass(image)
            new_vid.write(scaled_image)
            count += 1

        # train on the last frame
        new_vid.release()

        run_ZSSR_single_input.main(image, ground_truth_file, kernel_files_str, gpu, conf, res_dir)


if __name__ == '__main__':
    conf_str = sys.argv[1] if len(sys.argv) > 1 else None
    gpu_str = sys.argv[2] if len(sys.argv) > 2 else None
    main(conf_str, gpu_str)
