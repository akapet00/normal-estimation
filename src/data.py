import os

import numpy as np


__all__ = ['load_point_cloud']


def load_point_cloud(fname, unit=True):
    """Orient the normals in the outward direction.
    
    Parameters
    ----------
    fname : string
        Name of the desired point cloud model.
    unit : bool, optional
        If true, normals will be normalized by using the norm.
    
    Returns
    -------
    tuple
        Coordinates and corresponding unit normals.
    """
    fname = fname.lower()
    path = os.path.join('data', f'{fname}100k')
    try:
        xyz = np.genfromtxt(f'{path}.xyz', delimiter=' ')
        n = np.genfromtxt(f'{path}.normals', delimiter=' ')
        if unit:
            n = n / np.linalg.norm(n, axis=1)[:, np.newaxis]
        return xyz, n
    except IOError as e:
        print(e)
        return '', ''
