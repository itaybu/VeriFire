from blur import BlurEst5x5 , BlurEst5x5_gauss
import numpy as np
import matplotlib.pyplot as plt


blur = BlurEst5x5()
get_blur_accurate = blur.get_blur

get_blur_gauss = BlurEst5x5_gauss

x_vec = np.arange(-0.4,0.4,0.2)
y_vec = np.arange(-0.4,0.4,0.2)

fig_accurate,ax_accurate = plt.subplots(len(x_vec),len(y_vec))
fig_gauss,ax_gauss = plt.subplots(len(x_vec),len(y_vec))
fig_diff,ax_diff = plt.subplots(len(x_vec),len(y_vec))

blur_lower_bounds, blur_upper_bounds = blur.get_blur_bounds()


for ix,x in enumerate(x_vec):
    for iy,y in enumerate(y_vec):

        d_accurate = get_blur_accurate(x, y)
        d_gauss = get_blur_gauss(x, y, sigmaCol=0.5, sigmaLine=0.5)

        for ii,ax in enumerate([ax_accurate,ax_gauss,ax_diff]):

            if ii==0:
                img = d_accurate
            elif ii==1:
                img = d_gauss
            elif ii==2:
                img = d_accurate-d_gauss


            ax[ix,iy].imshow(img)
            if (ix==0):
                ax[ix,iy].set_title(f"y={y:.1f}")
            if (iy == 0):
                ax[ix, iy].set_ylabel(f"x={x:0.1f}")



fig_accurate.suptitle("Accurate")
fig_gauss.suptitle("gauss")
fig_diff.suptitle("diff")



plt.show()
# plt.figure(fig_accurate)
# plt.show()
# plt.figure(fig_gauss)
# plt.show()
# plt.figure(fig_diff)
# plt.show()



