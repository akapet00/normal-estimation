import numpy as np
from scipy import interpolate
from scipy import spatial
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from src.utils import apply_weight


class NormalEstimator(BaseEstimator):
    """Surface normal estimator.

    Parameters
    ----------
    k : float, optional
        The number of nearest neighbors of a local neighborhood around
        a current query point. Default is 30
    deg : str, optional
        Degree of the bivariate spline. Default is 3
    s : float, optional
        Positive smoothing factor for the bivariate spline
    kernel : str, optional
        The weighting function. By default, all weights are set to 1
    gamma : float, optional
        A scaling factor for the weighting function. If not given, it
        is set to 1
    """
    def __init__(self, k=30, deg=3, s=None, kernel=None, gamma=None):
        self.k = k
        self.deg = deg
        self.s = s
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : numpy.ndarray
            The input point cloud of shape (n_points, 3)
        y : None
            Ignored

        Returns
        -------
        self : object
            Fitted normal estimator
        """
        X = check_array(X, accept_sparse=True)
        # create a kd-tree for quick nearest-neighbor lookup
        tree = spatial.KDTree(X)
        self.n_estim = np.empty_like(X)
        # iterate over all points
        for i, p in enumerate(X):
            # local neighbourhood
            _, idx = tree.query([p], k=self.k, workers=-1)
            nbhd = X[idx.flatten()]
            # change of basis
            nbhd_c = nbhd - nbhd.mean(axis=0)
            C = (nbhd_c.T @ nbhd_c) / (nbhd.shape[0] - 1)
            U, _, _ = np.linalg.svd(C)
            nbhd_t = nbhd_c @ U
            # add weights
            if self.kernel:
                w = apply_weight(p, nbhd, self.kernel, self.gamma)
            else:
                w = np.ones((nbhd.shape[0], ))
            # interpolate
            h = interpolate.SmoothBivariateSpline(*nbhd_t.T,
                                                  w=w,
                                                  kx=self.deg,
                                                  ky=self.deg,
                                                  s=self.s)
            self.interpolant_ = h
            # compute normals as partial derivatives of the "height" function
            ni = np.array([-h(*nbhd_t[0, :2], dx=1).item(),
                           -h(*nbhd_t[0, :2], dy=1).item(),
                           1])
            # convert normal coordinates into the original coordinate frame
            ni = U @ ni
            # normalization
            self.n_estim[i, :] = ni / np.linalg.norm(ni)
        self.is_fitted_ = True
        return self

    def predict(self, X=None):
        """Return estimated normals.

        Parameters
        ----------
        X : None
            Ignored

        Returns
        -------
        y : numpy.ndarray
            Estimated surface normals of shape (n_points, 3)
        """
        check_is_fitted(self, 'is_fitted_')
        return self.n_estim
