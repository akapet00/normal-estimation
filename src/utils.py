import numpy as np


__all__ = ['polyfit2d']


def polyfit2d(x, y, z, deg=1, rcond=None, full_output=False):
    r"""Return the coefficients of a 2-D polynomial of a given degree.
    
    Note. The fitting assumes that the variable `z` corresponds the
    values of the "height" function such as z = f(x, y). The fitting is
    then performed on the polynomial given as:
    
    .. math:: f(x, y) = \sum_{i, j} c_{i, j} x^i y^j
    
    where `i + j Y= n` and `n` is the degree of a polynomial.
    
    Parameters
    ----------
    x, y : array_like, shape (N,)
        x- and y-oordinates of the M data points `(x[i], y[i])`.
    z : array_like, shape (M,)
        z-coordinates of the M data points.
    deg : int, optional
        Degree of the polynomial to be fit.
    rcond : float, optional
        Condition of the fit. See `scipy.linalg.lstsq` for details. All
        singular values less than `rcond` will be ignored.
    full_output : bool, optional
        Full diagnostic information from the SVD is returned if True,
        otherwise only the fitted coefficients are returned.
        
    Returns
    -------
    numpy.ndarray
        Array of coefficients of shape (deg+1, deg+1). If `full_output`
        is set to true, sum of the squared residuals of the fit, the
        effective rank of the design matrix, its singular values, and
        the specified value of `rcond` are also returned.
    """
    deg = int(deg)
    if deg < 1:
        raise ValueError('Degree must be at least 1.')
    # set up the Vandermode (design) matrix and the intercept vector
    A = np.polynomial.polynomial.polyvander2d(x, y, [deg, deg])
    b = z.flatten()
    # set up relative condition of the fit
    if rcond is None:
        rcond = x.size * np.finfo(x.dtype).eps
    # solve the least square
    coef, res, rank, s = np.linalg.lstsq(A, b, rcond=rcond)
    if full_output:
        return coef.reshape(deg+1, deg+1), res, rank, s
    return coef.reshape(deg+1, deg+1)