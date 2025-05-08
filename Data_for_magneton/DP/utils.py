import numpy as np
from scipy import ndimage

def normImg(x_data, kernelSize=12, flag_patches=0, flagConstStd=0):
    img_to_norm = x_data[:, :, :, 0]

    if flag_patches:
        mean_img = np.nanmean(img_to_norm, axis=(1, 2))
        std_img = np.nanstd(img_to_norm, axis=(1, 2))
    else:
        mean_img = calc_mean_image(img_to_norm, kernelSize)
        std_img = calc_std_image(img_to_norm, kernelSize)

    if flagConstStd:
        std_img = flagConstStd * np.ones(shape=std_img.shape)

    # Expand dims accordingly:
    if len(x_data.shape) > len(mean_img.shape):
        for axis in range(len(x_data.shape) - len(mean_img.shape)):
            mean_img = np.expand_dims(mean_img, axis=len(mean_img.shape))
            std_img = np.expand_dims(std_img, axis=len(std_img.shape))

    x_data = (x_data - mean_img) / (std_img + 1E-6)

    return x_data, mean_img, std_img

def calc_mean_image(img, kernel_size=12):
    kernel = np.ones(shape=(2 * kernel_size + 1, 2 * kernel_size + 1))
    kernel = kernel / np.sum(kernel, axis=(0, 1))
    kernel = np.expand_dims(kernel, axis=0)

    # Changed to astropy conv, in order to deal with nans
    # mean_image = astropy.convolution.convolve(img, kernel)
    mean_image = ndimage.filters.convolve(img, kernel, mode='constant')

    return mean_image

def calc_std_image(img, kernel_size=12):
    kernel = np.ones(shape=(2 * kernel_size + 1, 2 * kernel_size + 1))
    kernel = kernel / np.sum(kernel, axis=(0, 1))
    kernel = np.expand_dims(kernel, axis=0)
    c1 = ndimage.convolve(img, kernel, mode='reflect')
    c2 = ndimage.convolve(img * img, kernel, mode='reflect')

    return np.sqrt(c2 - c1 * c1)
