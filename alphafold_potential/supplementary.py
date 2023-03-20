
import pickle as pkl
from scipy.interpolate import CubicSpline, interp2d
import numpy as np

def read_pickle(path):
    output = open(path, 'rb')
    pred_pickle = pkl.load(output, encoding='latin1')
    output.close()
    return pred_pickle

def interpolate_hist_spline3(bar_heights):
    fit_data = bar_heights[:SPLINE_INTERPOLATION_IDX_LIMIT]
    spline = CubicSpline((np.arange(len(fit_data)) * DIST_STEP) + 2, fit_data)
    big_dist_vals = bar_heights[SPLINE_INTERPOLATION_IDX_LIMIT:]
    const_val = spline(2 + (len(fit_data) * DIST_STEP))
    return spline, float(const_val)

def convert_discrete_potential_to_spline(matrix):
    res = [interpolate_hist_spline3(row) for row in matrix.reshape(-1, 64)]
    return np.array(res).reshape((matrix.shape[0], matrix.shape[1], 2))

DIST_STEP = (22 - 2) / 64
SPLINE_INTERPOLATION_LIMIT_DIST = 18
SPLINE_INTERPOLATION_IDX_LIMIT = np.ceil((SPLINE_INTERPOLATION_LIMIT_DIST - 2) / DIST_STEP).astype(int)
