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

NETWORKS_DIR = "./networks"
PROPERTIES_DIR = "./properties"
# NETWORKS_DIR = "./networks_small"
# PROPERTIES_DIR = "./properties_small"

TH_VALUE = 1
INP_LOWER_BOUND = 1
INP_UPPER_BOUND = 2
DELTA = 0.001

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

    for i in range(len(x_data_ptchs_far)):
        print("creating network and vnnlib number " + str(i))
        img = x_data_ptchs_far[i]
        planting_img = x_data_ptchs_targets[i] - img

        planting_img[planting_img < 0.001] = 0

        # Build Marabou Network
        new_input = tf.keras.Input(shape=(1,))
        new_dense = tf.keras.layers.Dense(ptchSz * ptchSz * num_channels)(new_input)

        input_to_orig = tf.keras.layers.Reshape((ptchSz, ptchSz, num_channels))(new_dense)
        new_output = orig_net(input_to_orig)
        marabou_network = tf.keras.Model(inputs=new_input, outputs=new_output)

        # Set Weights:
        orig_weights = orig_net.get_weights()
        new_weights = [planting_img.reshape(1, ptchSz * ptchSz * num_channels), img.flatten()] + orig_weights
        marabou_network.set_weights(new_weights)

        # marabou_network.save(f"networks_tf/dp-net_" + str(i))

        onnx_network, _ = tf2onnx.convert.from_keras(marabou_network)
        onnx.save(onnx_network, os.path.join(NETWORKS_DIR, f"dp-net_{i}.onnx"))

        # run onnx network on threshold value:
        session = ort.InferenceSession(os.path.join(NETWORKS_DIR, f"dp-net_{i}.onnx"))
        input_name = session.get_inputs()[0].name
        input_data = np.array([[1]], dtype=np.float32)
        output_for_th_value = session.run(None, {input_name: input_data})[0][0, 0]

        # Create VNNLIB property file
        with open(os.path.join(PROPERTIES_DIR, f"dp-net_{i}.vnnlib"), "w") as f:
            f.write("(declare-const X_0 Real)\n")
            f.write("(declare-const Y_0 Real)\n")
            f.write("\n")
            f.write(f"(assert (>= X_0 {INP_LOWER_BOUND}))\n")
            f.write(f"(assert (<= X_0 {INP_UPPER_BOUND}))\n")
            f.write(f"(assert (<= Y_0 {output_for_th_value - DELTA}))\n")