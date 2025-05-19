import os.path

import numpy as np
import scipy.io as scio
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
from Data_for_magneton.DP import load_net_n_predict
from Data_for_magneton.DP import utils
# from DP_small import load_net_n_predict
# from DP_small import utils
from blur import BlurEst5x5 , BlurEst5x5_gauss

NETWORKS_DIR = "./networks"
PROPERTIES_DIR = "./properties"
# NETWORKS_DIR = "./networks_small"
# PROPERTIES_DIR = "./properties_small"

TH_VALUE = 1
INP_LOWER_BOUND = 1
INP_UPPER_BOUND = 2
DELTA = 0.001
BLURED_PATCH_SIZE = 5
BLUE_LOCATION = (10, 10)
ptchSz = 25
num_channels = 2
main_path_net = r"Data_for_magneton/DP"
whichNet = 'binary_crossentropy'
# main_path_net = r"DP_small/2L_no_bn"
# whichNet = 'binarySmall'
gamma = 0
title = 'test'
flag_save_point_score = False

save_path = "./Results"

if __name__ == '__main__':
    if not os.path.exists(NETWORKS_DIR):
        os.makedirs(NETWORKS_DIR)
    if not os.path.exists(PROPERTIES_DIR):
        os.makedirs(PROPERTIES_DIR)

    # Load Data
    x_data_ptchs_test_targets_file = scio.loadmat("Data_for_magneton/DP/patches_targets.mat")
    x_data_ptchs_test_targets = x_data_ptchs_test_targets_file['patches_targets']
    x_data_ptchs_test_far_file = scio.loadmat("Data_for_magneton/DP/patches_no_targets.mat")
    x_data_ptchs_test_far = x_data_ptchs_test_far_file['patches_no_targets']

    # # Normalize
    x_data_ptchs_targets , _, _  =  utils.normImg(x_data_ptchs_test_targets , kernelSize=12,flag_patches=1)
    x_data_ptchs_far, _, _ = utils.normImg(x_data_ptchs_test_far , kernelSize=12, flag_patches=1)

    orig_net = load_net_n_predict.load_net(x_data_ptchs_targets, x_data_ptchs_far, main_path_net,
                                           ptchSz=ptchSz,
                                           whichNet=whichNet,
                                           title=title,
                                           flag_save_point_score=flag_save_point_score,
                                           gamma=gamma,
                                           save_path=save_path,
                                           flag_batch_norm = False,
                                           flag_norm=0)

    output_for_th_value = orig_net.predict(x_data_ptchs_targets)

    blur = BlurEst5x5()
    blur_lower_bounds, blur_upper_bounds = blur.get_blur_bounds()
    blur_lower_bounds, blur_upper_bounds = blur_lower_bounds.flatten(), blur_upper_bounds.flatten()

    for i in range(len(x_data_ptchs_far)):
        print("creating network and vnnlib number " + str(i))
        img = x_data_ptchs_far[i]
        planting_img = x_data_ptchs_targets[i] - img

        planting_img[planting_img < 0.001] = 0

        # Build Marabou Network
        new_input = tf.keras.Input(shape=(BLURED_PATCH_SIZE * BLURED_PATCH_SIZE,))

        new_dense = tf.keras.layers.Dense(ptchSz * ptchSz * num_channels)(new_input)

        input_to_orig = tf.keras.layers.Reshape((ptchSz, ptchSz, num_channels))(new_dense)
        new_output = orig_net(input_to_orig)
        marabou_network = tf.keras.Model(inputs=new_input, outputs=new_output)

        # Set Weights:
        orig_weights = orig_net.get_weights()
        new_dense_weights = np.zeros((BLURED_PATCH_SIZE * BLURED_PATCH_SIZE, ptchSz * ptchSz * num_channels))
        for j in range(BLURED_PATCH_SIZE * BLURED_PATCH_SIZE):
            new_dense_weights[j, 2 * ((10 + j // 5) * 25 + 10 + j % 5)] = 1
            new_dense_weights[j, 2 * ((10 + j // 5) * 25 + 10 + j % 5) + 1] = 1

        new_weights = [new_dense_weights, img.flatten()] + orig_weights
        marabou_network.set_weights(new_weights)

        spec = [tf.TensorSpec((None, 25), tf.float32, name="input")]
        onnx_network, _ = tf2onnx.convert.from_keras(marabou_network, input_signature=spec)
        onnx.save(onnx_network, os.path.join(NETWORKS_DIR, f"dp-net_{i}.onnx"))

        # Create VNNLIB property file
        with open(os.path.join(PROPERTIES_DIR, f"dp-net_{i}.vnnlib"), "w") as f:
            for j in range(BLURED_PATCH_SIZE * BLURED_PATCH_SIZE):
                f.write(f"(declare-const X_{j} Real)\n")
            f.write("(declare-const Y_0 Real)\n")
            f.write("\n")
            for j in range(BLURED_PATCH_SIZE * BLURED_PATCH_SIZE):
                f.write(f"(assert (>= X_{j} {blur_lower_bounds[j] * INP_LOWER_BOUND}))\n")
                f.write(f"(assert (<= X_{j} {blur_upper_bounds[j] * INP_UPPER_BOUND}))\n")
            f.write(f"(assert (<= Y_0 {output_for_th_value[i, 0] - DELTA}))\n")