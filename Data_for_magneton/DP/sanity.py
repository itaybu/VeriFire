import os
import scipy.io as scio
import utils
import numpy as np
# import Data_for_magneton.DP.load_net_n_predict
import model_PointNet2


# #Where to save results
# save_path = r"\Results"
# if not os.path.isdir(save_path):
#     os.makedirs(save_path, exist_ok=True)

# Load Data
# x_data_ptchs_test_targets_file = scio.loadmat(r"patches_targets.mat")
# x_data_ptchs_test_targets = x_data_ptchs_test_targets_file['patches_targets']
# x_data_ptchs_test_far_file = scio.loadmat(r"patches_no_targets.mat")
# x_data_ptchs_test_far = x_data_ptchs_test_far_file['patches_no_targets']
#
# # Normalize
# x_data_ptchs_targets , _, _  =  utils.normImg(x_data_ptchs_test_targets , kernelSize=12,flag_patches=1)
# x_data_ptchs_far, _, _ = utils.normImg(x_data_ptchs_test_far , kernelSize=12, flag_patches=1)

ptchSz = 25
main_path_net = r""
whichNet = 'binary_crossentropy'
gamma = 0

title = 'test'
flag_save_point_score = False
# net = load_net_n_predict.load_net(x_data_ptchs_targets, x_data_ptchs_far, main_path_net,
#                                       ptchSz=ptchSz,
#                                       whichNet=whichNet,
#                                       title=title,
#                                       flag_save_point_score=flag_save_point_score,
#                                       gamma=gamma,
#                                       save_path=save_path,
#                                       flag_batch_norm = False,
#                                       flag_norm=0)

net = model_PointNet2.build_net_13X13(patch_sz_row=ptchSz, patch_sz_col=ptchSz, flag_batch_norm=False, num_filters=4)
img = np.ones((1,25,25,1)) # TODO : change to input image

print(net.predict(np.ones((1,25,25,1))))

net.save("dp-net")