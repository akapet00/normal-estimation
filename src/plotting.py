import itertools

import matplotlib.pyplot as plt
import numpy as np


__all__ = ['show_point_cloud', 'draw_unit_cube']


def _set_axes_equal(ax):
    """Return adjusted axes ratios of a 3-D subplot.
    
    Note. See https://stackoverflow.com/a/31364297/15005103 for
    implementation details.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3-D subplot with unequal ratio.
    
    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        Subplot with adjusted ratios of axes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return ax


def _encode_normals_rgb(n):
    """Map vectors into corresponding RGB colors considering the RGB cube.
    
    Note. See
    https://www.mathworks.com/matlabcentral/fileexchange/71178-normal-vector-to-rgb
    for implementation details.
    
    Parameters
    ----------
    n : numpy.ndsarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.

    Returns
    -------
    numpy.ndarray
        Array of RGB color values (N, 3) for each normal.
    """
    n = np.divide(n, np.tile(np.expand_dims(
        np.sqrt(np.sum(np.square(n), axis=1)), axis=1), [1, 3]))
    rgb = 127.5 + 127.5 * n
    return rgb / 255.0


def show_point_cloud(xyz, n=None, elev=0, azim=0, **kwargs):
    """Show and return 3-D subplot of a given point cloud. Colorization
    is performed by converting normals to corresponding RGB values
    by mapping them considering the RGB cube.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    n : numpy.ndarray, optional
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud. If not given, point cloud will be
        colorized considering the values of the x-axis.
    elev : float, optional
        The elevation angle in degrees rotates the camera above the
        plane of the vertical axis, with a positive angle corresponding
        to a location above that plane.
    azim : float, optional
        The azimuthal angle in degrees rotates the camera about the
        vertical axis, with a positive angle corresponding to a
        right-handed rotation.
    kwargs : dict, optional
        Additional keyword arguments for the scatter plotting function.

    Returns
    -------
    tuple
        Figure and a 3-D subplot within.
    """
    if n is not None:
        c = _encode_normals_rgb(n)
    else:
        c = xyz[:, 0].flatten()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection ='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.25, c=c, **kwargs)
    ax.set_box_aspect([1, 1, 1])
    ax = _set_axes_equal(ax)
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
    return fig, ax


def draw_unit_cube(elev, azim):
    """Return the RGB unit cube.

    Parameters
    ----------
    elev : float, optional
        The elevation angle in degrees rotates the camera above the
        plane of the vertical axis, with a positive angle corresponding
        to a location above that plane.
    azim : float, optional
        The azimuthal angle in degrees rotates the camera about the
        vertical axis, with a positive angle corresponding to a
        right-handed rotation.

    Returns
    -------
    tuple
        Figure and a 3-D subplot within.
    """
    # create the skeleton of the cube
    r = [0, 1]
    X, Y = np.meshgrid(r, r)
    ones = np.ones(4).reshape(2, 2)
    zeros = np.zeros_like(ones)
    fig = plt.figure(figsize=(2, 2))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, zeros, lw=2, color='None', edgecolor='k')
    ax.plot_surface(X, Y, zeros, lw=2, color='None', edgecolor='k')
    ax.plot_surface(X, zeros, Y, lw=2, color='None', edgecolor='k')
    ax.plot_surface(X, ones, Y, lw=2, color='None', edgecolor='k')
    ax.plot_surface(ones, X, Y, lw=2, color='None', edgecolor='k')
    ax.plot_surface(zeros, X, Y, lw=2, color='None', edgecolor='k')
    # add colorized points
    pts = np.array(list(itertools.product([0, 1], repeat=3)))
    ax.scatter(*pts.T, c=pts, edgecolor='k', depthshade=False, s=450)
    ax.set_box_aspect([1, 1, 1])
    ax = _set_axes_equal(ax)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
    return fig, ax
