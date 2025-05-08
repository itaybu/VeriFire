import os
import glob
import numpy as np
## my modules
import scipy.io as scio
from Data_for_magneton.DP import model_PointNet_ver2
from Data_for_magneton.DP import utils


def load_net(x_data_targets, x_data_fars, main_path_net, ptchSz = 25, whichNet = 'binary_crossentropy', title='',
                       flag_save_point_score = False,flag_batch_norm = False,gamma = 0, save_path = '', flag_norm = 0):


    batch_size = 1024  # 256*4
    # batch_size = 13*13  # 256*4
    num_filters = 64
    # num_filters = 2

    # if x_data_targets and x_data_fars is the file name - load it
    if isinstance(x_data_targets, str):
        x_data_targets = scio.loadmat(x_data_targets)
        x_data_targets = x_data_targets['img']
        x_data_fars = scio.loadmat(x_data_fars)
        x_data_fars = x_data_fars['img']

    currPtchSz = x_data_targets.shape[1]

    # norm
    if flag_norm==True:
        x_data_targets_unNorm = np.copy(x_data_targets)
        x_data_fars_unNorm = np.copy(x_data_fars)
        x_data_targets, _, _ = utils.normImg(x_data_targets, flag_patches=1)
        x_data_fars, _, _ = utils.normImg(x_data_fars, flag_patches=1)

    # # ############ REMOVE ##########
    idx2remove = (currPtchSz - ptchSz) // 2
    if ptchSz != 25 and idx2remove!=0:
        x_data_targets = x_data_targets[:, idx2remove:-idx2remove, idx2remove:-idx2remove, :]
        x_data_fars = x_data_fars[:, idx2remove:-idx2remove, idx2remove:-idx2remove, :]
    # # ############

    if whichNet=='binary_crossentropy' or whichNet=='mobile':
        net_path = glob.glob(os.path.join(main_path_net,'*_ptchsz{}_{}'.format(ptchSz,whichNet)))[0]
    model_name = os.path.split(net_path)[1]
    model_file_name = os.path.join(main_path_net, model_name, model_name)
    model_dir = os.path.join(main_path_net, model_name)

    # Sizes
    trgt_data_size, patch_sz_row, patch_sz_col, num_channels = x_data_targets.shape
    far_data_size = x_data_fars.shape[0]
    total_data_size = trgt_data_size + far_data_size

    print('Train Data Size:{}\n'.format(trgt_data_size))
    print('Val Data Size:{}\n'.format(far_data_size))

    # Build NET
    if whichNet=='binary_crossentropy' or whichNet=='HingeLoss' or whichNet=='SquaredHingeLoss' or \
            whichNet == 'BCE_retina_loss':
        net = model_PointNet_ver2.build_net(ptchSz,
                      num_channels = num_channels,
                      flag_batch_norm = flag_batch_norm,
                      padding = 'valid',
                      num_filters = num_filters,
                      which_loss = whichNet,
                      gamma = gamma)
    net.summary()

    # load best weights
    net.load_weights(model_file_name+'_bestWeights.hdf5')

    # plot prediction
    # predictions_trgt = net.predict(x_data_targets, batch_size=batch_size)
    # predictions_far = net.predict(x_data_fars, batch_size=batch_size)
    #
    # dict = {'predictions_trgt': predictions_trgt, 'predictions_far': predictions_far}
    #
    #
    #
    # if save_path=='':
    #     scio.savemat(model_file_name + 'PredictionRes{}.mat'.format(title), mdict=dict)
    # else:
    #     scio.savemat(os.path.join(save_path,'PredictionRes{}.mat'.format(title)), mdict=dict)
    #
    # return predictions_trgt, predictions_far, model_name, model_dir
    return net
    # ############################ END OF CODE #################################

