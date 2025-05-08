import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.io as spio
from scipy import signal


# def get_blue_bounds(self):
#     Blur4 = self._Blur4
#     Sb = np.array(Blur4.shape)
#     Cen = 1 + (Sb - 1) / 2
#
#     row_col_pos = np.array([-2, -1, 0, 1, 2])
#
#     PosLineMin = Cen[1] + (-12 + 30 * row_col_pos) / self._Dh
#     PosColMin = Cen[0] + (-12 + 30 * row_col_pos) / self._Dh
#
#     PosLineMax = Cen[1] + (-12 + 30 * row_col_pos) / self._Dh
#     PosColMax = Cen[0] + (-12 + 30 * row_col_pos) / self._Dh
#
#     blue_lower_bounds, blue_upper_bounds = np.zeros((5, 5)), np.zeros((5, 5))
#
#     for i in range(5):
#         for j in range(5):
#             for k in range(PosLineMin[i], PosLineMax[i]):
#
#     return blue_lower_bounds, blue_upper_bounds
#

import numpy as np



class BlurEst5x5:
    def __init__(self , blurType='Blur11um_70'):
        self.blurType = blurType
        self.blurStructDir = os.path.join('blurs', blurType) + '.mat'
        self.blurStruct = spio.loadmat(self.blurStructDir)
        self._Dh = self.blurStruct['Dh'][0][0]
        self._Blur4 = self.blurStruct['Blur4']
        self.M5x5_func = self.M5x5()
        self.blurTypeGlobal = None

    def M5x5(self):

        [X_before, Y_before] = np.meshgrid(range(1, np.size(self._Blur4, 1) + 1), range(1, np.size(self._Blur4, 0) + 1))

        X_before = np.matrix.flatten(X_before)
        Y_before = np.matrix.flatten(Y_before)
        Blur4_vec = np.matrix.flatten(self._Blur4)
        M5x5_func = interpolate.NearestNDInterpolator((X_before, Y_before), Blur4_vec)

        return M5x5_func


    def get_blur (self,Column, Line ):

        x = Column
        y = Line

        # !!!!! Warning This might be pixel length dependant!!!!
        x = np.fmod(x, 1) * 30  # translate dimentionless position to [um]
        y = np.fmod(y, 1) * 30  # translate dimentionless position to [um]

        # Dh - distance in [um] between adjacent sampling points.
        # Blur4 -

        Blur4 = self._Blur4

        Sb = np.array(Blur4.shape)
        Cen = 1 + (Sb - 1) / 2  # medium-resolusion PSF array (Blur4) center position .

        # !!!!! Warning This might be pixel length dependant!!!!
        row_col_pos = np.array([-2, -1, 0, 1, 2])
        PosLine = Cen[1] + (-y + 30 * row_col_pos) / self._Dh  # Exact(non-discretized) Blur4 Row position
        PosCol = Cen[0] + (-x + 30 * row_col_pos) / self._Dh  # Exact(non-discretized) Blur4 Column position

        #
        # Interpolate medium-resolusion continouse PSF function to enhance accuracy -
        #

        [X_after, Y_after] = np.meshgrid(PosCol, PosLine)

        if True:
            [X_before, Y_before] = np.meshgrid(range(1, np.size(Blur4, 1) + 1), range(1, np.size(Blur4, 0) + 1))

            X_before = np.matrix.flatten(X_before)
            Y_before = np.matrix.flatten(Y_before)
            Blur4_vec = np.matrix.flatten(Blur4)

            M5x5_func = interpolate.NearestNDInterpolator((X_before, Y_before), Blur4_vec)
            #blurTypeGlobal = blurType

        M5x5 = M5x5_func((X_after, Y_after))

        M5x5[np.isnan(M5x5)] = 0

        return M5x5

    def get_blur_bounds(self):
        BlurMatrix = self._Blur4
        n_rows, n_cols = BlurMatrix.shape

        min_pos, max_pos = np.fmod(-0.4, 1) * 30, np.fmod(0.2, 1) * 30

        Sb = np.array([n_rows, n_cols])
        Cen = 1 + (Sb - 1) / 2.0

        row_col_pos = np.array([-2, -1, 0, 1, 2])

        PosLineMin = Cen[1] + (min_pos + 30 * row_col_pos) / self._Dh
        PosColMin = Cen[0] + (min_pos + 30 * row_col_pos) / self._Dh

        PosLineMax = Cen[1] + (max_pos + 30 * row_col_pos) / self._Dh
        PosColMax = Cen[0] + (max_pos + 30 * row_col_pos) / self._Dh

        blur_lower_bounds, blur_upper_bounds = np.zeros((5, 5)), np.zeros((5, 5))

        for i in range(5):
            for j in range(5):
                r0 = int(np.floor(PosLineMin[i])) - 1
                r1 = int(np.ceil(PosLineMax[i])) - 1
                c0 = int(np.floor(PosColMin[j])) - 1
                c1 = int(np.ceil(PosColMax[j])) - 1

                r0 = max(0, min(r0, n_rows - 1))
                r1 = max(0, min(r1, n_rows - 1))
                c0 = max(0, min(c0, n_cols - 1))
                c1 = max(0, min(c1, n_cols - 1))

                valid_ranges = BlurMatrix[r0:r1 + 1, c0:c1 + 1]

                blur_lower_bounds[i, j] = valid_ranges.min()
                blur_upper_bounds[i, j] = valid_ranges.max()

        return blur_lower_bounds, blur_upper_bounds


def BlurEst5x5_gauss(line, col, sigmaLine, sigmaCol, theta=0, intensityRatio=1):
    # for BoloMWS/PAWSU
    my_eps_line = 0.000001 * np.sign(line)
    my_eps_col = 0.000001 * np.sign(col)
    line = line - np.round(line + my_eps_line)
    col = col - np.round(col + my_eps_col)

    theta = theta * np.pi / 180

    # sigma line/col must be a scalar
    tempMat = np.array([[sigmaLine * sigmaLine, 0], [0, sigmaCol * sigmaCol]])

    rotMat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    covMat = rotMat * tempMat

    invCovMat = np.linalg.inv(covMat)

    M5x5 = samplePSF(invCovMat, line, col) * intensityRatio

    return M5x5


def samplePSF(iCov, subLine, subCol, radius=2):
    x_grid = np.arange(-radius, radius + 1) - subCol
    y_grid = np.arange(-radius, radius + 1) - subLine

    [X, Y] = np.meshgrid(x_grid, y_grid)

    X = np.ndarray.flatten(X, order='F')
    Y = np.ndarray.flatten(Y, order='F')

    temp = -0.5 * np.dot(iCov, np.array([X, Y]))

    result = np.exp(np.sum(np.transpose(temp) * np.transpose(np.array([X, Y])), axis=1))
    result = result / np.sum(result)
    result = np.reshape(result, (2 * radius + 1, 2 * radius + 1))

    return result



