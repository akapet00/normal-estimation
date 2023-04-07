import numpy as np


__all__ = ['weight', 'rms_angle_error']


def weight(p, nbhd, kernel='rbf', sigma=None):
    """Return scaling weights given distance between a targeted point
    and its surrounding local neighborhood.
    
    p : numpy_ndarray
        Targeted point of shape (3, ).
    nbhd : numpy.ndarray
        An array of shape (N, 3) representing the local neighborhood.
    kernel : str, optional
        The weighting function to use for MLS fitting. Possible options
        include 'rbf', the Gaussian kernel, 'cosine', the cosine
        similarity computed as the L2-normalized dot product,
        'truncated', the truncated quadratic kernel, 'linear', the
        linear kernel for weighting, 'inverse', the inverse kernel. If
        not set, all weights will be set to 1.
    sigma : float, optional
        A scaling factor for the weighting function. If not given, it
        is set to 1 / N where N is the total number of points in the
        local neighborhood.
    """
    dist = np.linalg.norm(nbhd - p, axis=1)  # squared Euclidian distance
    if sigma is None:
        sigma = 1.
    if kernel == 'rbf':  # Gaussian
        w = np.exp(-dist ** 2 / (2 * sigma ** 2))
    elif kernel == 'cosine':
        w = (nbhd @ p) / np.linalg.norm(nbhd * p, axis=1)
    elif kernel == 'linear':
        w = np.maximum(1 - sigma * dist, 0)
    elif kernel == 'inverse':
        w = 1 / dist
    elif kernel == 'truncated':
        w = np.maximum(1 - sigma * dist ** 2, 0)
    else:
        w = np.ones_like(dist)
    return w


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
    N = n_gt.shape[0]
    if orient:
        rel = np.sum(n_estim * n_gt, axis=1)
    else:
        rel = np.sum(np.abs(n_estim) * np.abs(n_gt), axis=1)
    rel = np.where(rel > 1, 1, rel)
    rel = np.where(rel < -1, -1, rel)
    if np.sum(rel) / N == 1:  # all normals matched -> no error
        return 0
    theta = np.arccos(rel) * 180 / np.pi
    return np.sqrt(np.mean(theta ** 2))
