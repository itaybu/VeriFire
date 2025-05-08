import matplotlib.pyplot as plt
import scipy.io as scio
import utils


data = scio.loadmat(r"patches_targets.mat")['patches_targets']
dara , _, _  =  utils.normImg(data , kernelSize=12,flag_patches=1)

plt.imshow(data[0][:,:,0])

plt.show()
pass