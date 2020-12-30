import sys
import os
import configs
import ZSSR
import numpy as np
from binary_file_ops import read_binary_image


def main(input_img, ground_truth, kernels, gpu, conf, results_path):
    # Choose the wanted GPU
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = '%s' % gpu

    # 0 input for ground-truth or kernels means None
    ground_truth = None if ground_truth == '0' else ground_truth
    print('*****', kernels)
    kernels = None if kernels == '0' else kernels.split(';')[:-1]

    # this code doesn't seem to work in python 3 even though the command is supported
    # just passing in the actual conf object from the run_ZSSR file
    # Setup configuration and results directory
    # conf = configs.Config()
    # if conf_str is not None:
        # exec ('conf = configs.%s' % conf_str)
    conf.result_path = results_path

    if conf.input_image_type == 1:
        tmp_img = read_binary_image(input_img)
        tmp_img = np.true_divide(tmp_img.astype(np.float32),conf.image_scale)
        h, w = tmp_img.shape
        bin_img = np.empty((h, w, 3), dtype=np.float)
        # convert to a 3 channel image - tf and this code don't seem to work with a single channel image?????
        bin_img[:, :, 2] =  bin_img[:, :, 1] =  bin_img[:, :, 0] =  tmp_img

        net = ZSSR.ZSSR(bin_img, conf, ground_truth, kernels)
    else:
        # Run ZSSR on the image
        net = ZSSR.ZSSR(input_img, conf, ground_truth, kernels)
        
    net.run()
    return net


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
