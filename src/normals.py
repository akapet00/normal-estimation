import numpy as np
from scipy import interpolate
from scipy import spatial
from sklearn.decomposition import KernelPCA

from .utils import apply_weight, polyfit2d


__all__ = ['estimate_normals_pca',
           'estimate_normals_kpca',
           'estimate_normals_spline',
           'estimate_normals_poly']


def estimate_normals_pca(xyz, k, kernel=None, **kwargs):
    """Return the unit normals by fitting local tangent plane at each
    point in the point cloud.
    
    Ref: Hoppe et al., in proceedings of SIGGRAPH 1992, pp. 71-78,
         doi: 10.1145/133994.134011
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    k : float
        The number of nearest neighbors of a local neighborhood around
        a current query point.
    kernel : string, optional
        Kernel for computing distance-based weights.
    kwargs : dict, optional
        Additional keyword arguments for computing weights. For details
        see `apply_weight` function.
    
    Returns
    -------
    numpy.ndarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    """
    # create a kd-tree for quick nearest-neighbor lookup
    tree = spatial.KDTree(xyz)
    n = np.empty_like(xyz)
    for i, p in enumerate(xyz):
        # extract the local neighborhood
        _, idx = tree.query([p], k=k, eps=0.1, workers=-1)
        nbhd = xyz[idx.flatten()]
        
        # compute the kernel function and create the weights matrix
        if kernel:
            w = apply_weight(p, nbhd, kernel, **kwargs)
        else:
            w = np.ones((nbhd.shape[0], ))
        W = np.diag(w)
        
        # extract an eigenvector with smallest associeted eigenvalue
        X = nbhd.copy()
        X = X - np.average(X, weights=w, axis=0)
        C = (X.T @ (W @ X)) / np.sum(w)
        U, S, VT = np.linalg.svd(C)
        n[i, :] =  U[:, np.argmin(S)]
    return n


def estimate_normals_kpca(xyz, k, kernel=None, **kwargs):
    """Return the unit normals by fitting local tangent plane at each
    point in the point cloud by using kernel PCA approach to better
    handle non-linear patterns.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    k : float
        The number of nearest neighbors of a local neighborhood around
        a current query point.
    kernel : string, optional
        Kernel for computing distance-based weights.
    kwargs : dict, optional
        Additional keyword arguments for the initialization of the
        `sklearn.decomposition.KernelPCA` class.
    
    Returns
    -------
    numpy.ndarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    """
    # create a kd-tree for quick nearest-neighbor lookup
    tree = spatial.KDTree(xyz)
    n = np.empty_like(xyz)
    for i, p in enumerate(xyz):
        # extract the local neighborhood
        _, idx = tree.query([p], k=k, eps=0.1, workers=-1)
        nbhd = xyz[idx.flatten()]
        
        # normalize the data
        X = nbhd.copy()
        X = X - X.mean(axis=0)
        
        # create an instance of the kernel PCA class
        kpca = KernelPCA(n_components=3, kernel=kernel, **kwargs)
        kpca.fit(X)
        U = X.T @ kpca.eigenvectors_
        ni = U[:, np.argmin(kpca.eigenvalues_)]
        n[i, :] = ni / np.linalg.norm(ni)
    return n


def estimate_normals_spline(xyz,
                            k,
                            deg=3,
                            s=None,
                            unit=True,
                            kernel=None, 
                            **kwargs):
    """Return the (unit) normals by constructing smooth bivariate
    B-spline at each point in the point cloud considering its local
    neighborhood.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    k : float
        The number of nearest neighbors of a local neighborhood around
        a current query point.
    deg : float, optional
        Degrees of the bivariate spline.
    s : float, optional
        Positive smoothing factor defined for smooth bivariate spline
        approximation.
    unit : float, optional
        If true, normals are normalized. Otherwise, surface normals are
        returned.
    kernel : string, optional
        Kernel for computing distance-based weights.
    kwargs : dict, optional
        Additional keyword arguments for computing weights. For details
        see `apply_weight` function.
    
    Returns
    -------
    numpy.ndarray
        The (unit) normals of shape (N, 3), where N is the number of
        points in the point cloud.
    """
    # create a kd-tree for quick nearest-neighbor lookup
    tree = spatial.KDTree(xyz)
    n = np.empty_like(xyz)
    for i, p in enumerate(xyz):
        _, idx = tree.query([p], k=k, eps=0.1, workers=-1)
        nbhd = xyz[idx.flatten()]
        
        # change the basis of the local neighborhood
        X = nbhd.copy()
        X = X - X.mean(axis=0)
        C = (X.T @ X) / (nbhd.shape[0] - 1)
        U, _, _ = np.linalg.svd(C)
        Xt = X @ U
        
        # compute weights given specific distance function
        if kernel:
            w = apply_weight(p, nbhd, kernel, **kwargs)
        else:
            w = np.ones((nbhd.shape[0], ))
            
        # create a smooth B-Spline representation of the "height" function
        h = interpolate.SmoothBivariateSpline(*Xt.T, w=w, kx=deg, ky=deg, s=s)
        
        # compute normals as partial derivatives of the "height" function
        ni = np.array([-h(*Xt[0, :2], dx=1).item(),
                       -h(*Xt[0, :2], dy=1).item(),
                       1])
        
        # convert normal coordinates into the original coordinate frame
        ni = U @ ni
        
        # normalize normals by considering the magnitude of each
        if unit:
            ni = ni / np.linalg.norm(ni, 2)
        n[i, :] = ni
    return n


def estimate_normals_poly(xyz,
                          k,
                          deg=1,
                          unit=True,
                          kernel=None,
                          **kwargs):
    """Return the (unit) normals by fitting 2-D polynomial at each
    point in the point cloud considering its local neighborhood.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    k : float
        The number of nearest neighbors of a local neighborhood around
        a current query point.
    deg : float, optional
        Degrees of the polynomial.
    unit : float, optional
        If true, normals are normalized. Otherwise, surface normals are
        returned.
    kernel : string, optional
        Kernel for computing distance-based weights.
    kwargs : dict, optional
        Additional keyword arguments for computing weights. For details
        see `apply_weight` function.
    
    Returns
    -------
    numpy.ndarray
        The (unit) normals of shape (N, 3), where N is the number of
        points in the point cloud.
    """
    # create a kd-tree for quick nearest-neighbor lookup
    n = np.empty_like(xyz)
    tree = spatial.KDTree(xyz)
    for i, p in enumerate(xyz):
        _, idx = tree.query([p], k=k, eps=0.1, workers=-1)
        nbhd = xyz[idx.flatten()]
        
        # change the basis of the local neighborhood
        X = nbhd.copy()
        X = X - X.mean(axis=0)
        C = (X.T @ X) / (nbhd.shape[0] - 1)
        U, _, _ = np.linalg.svd(C)
        X_t = X @ U
        
        # compute weights given specific distance function
        if kernel:
            w = apply_weight(p, nbhd, kernel, **kwargs)
        else:
            w = np.ones((nbhd.shape[0], ))
            
        # fit parametric surface by usign a (weighted) 2-D polynomial
        X_t_w = X_t * w[:, np.newaxis]
        c = polyfit2d(*X_t_w.T, deg=deg)
        
        # compute normals as partial derivatives of the "height" function
        cu = np.polynomial.polynomial.polyder(c, axis=0)
        cv = np.polynomial.polynomial.polyder(c, axis=1)
        ni = np.array([-np.polynomial.polynomial.polyval2d(*X_t_w[0, :2], cu),
                       -np.polynomial.polynomial.polyval2d(*X_t_w[0, :2], cv),
                       1])
        
        # convert normal coordinates into the original coordinate frame
        ni = U @ ni
        
        # normalize normals by considering the magnitude of each
        if unit:
            ni = ni / np.linalg.norm(ni, 2)
        n[i, :] = ni
    return n
