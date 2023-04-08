import numpy as np


__all__ = ['rms_angle_error']


def _angle_error(n_estim, n_gt, orient=True):
    """Return the angle error(s) in degrees between the estimated and
    ground truth normal vector.
    
    Parameters
    ----------
    n_estim : numpy.ndsarray
        Estimated unit normals of shape (N, 3), where N is the
        number of points in the point cloud.
    n_gt : numpy.ndarray
        Ground truth normals of the corresponding shape.
    orient : bool, optional
        If it is set to True, orientation of the normals is taken
        into account. Otherwise, orientation does not matter.
    
    Returns
    -------
    float or numpy.ndarray
        Angle error(s) in degrees.
    """
    N = n_gt.shape[0]
    if orient:
        dot = np.sum(n_estim * n_gt, axis=1)
    else:
        dot = np.sum(np.abs(n_estim) * np.abs(n_gt), axis=1)
    dot = np.where(dot > 1, 1, dot)
    dot = np.where(dot < -1, -1, dot)
    if np.sum(dot) / N == 1:  # all normals matched -> no error
        return 0
    return np.arccos(dot) * 180 / np.pi
    

def rms_angle_error(n_estim, n_gt, orient=True):
    """Return the root mean square angle error between estimated and
    ground truth normal vectors.
    
    Parameters
    ----------
    n_estim : numpy.ndsarray
        Estimated unit normals of shape (N, 3), where N is the
        number of points in the point cloud.
    n_gt : numpy.ndarray
        Ground truth normals of the corresponding shape.
    orient : bool, optional
        If it is set to True, orientation of the normals is taken
        into account. Otherwise, orientation does not matter.
    
    Returns
    -------
    float
        Root mean square angle error in degrees.
    """
    err = _angle_error(n_estim, n_gt, orient)
    return np.sqrt(np.mean(err ** 2))
