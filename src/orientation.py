from queue import LifoQueue

import numpy as np
from scipy import spatial
from scipy import sparse
from sklearn import neighbors


__all__ = ['orient_normals_cvx', 'orient_normals']


def orient_normals_cvx(xyz, n):
    """Orient the normals in the outward direction.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    n : numpy.ndarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    
    Returns
    -------
    numpy.ndarray
        Oriented normals.
    """
    center = np.mean(xyz, axis=0)
    for i in range(xyz.shape[0]):
        pi = xyz[i, :] - center
        ni = n[i]
        angle = np.arccos(np.clip(np.dot(ni, pi), -1.0, 1.0))
        if (angle > np.pi/2) or (angle < -np.pi/2):
            n[i] = -ni
    return n


def _compute_emst(xyz):
    """Compute the symmetric Euclidean minimum spanning graph.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        The symmetric Euclidian minimum spanning graph.
    """
    tri = spatial.Delaunay(xyz)
    edges = ((0, 1), (1, 2), (0, 2))
    delaunay_edges = None
    for edge in edges:
        if delaunay_edges is None:
            delaunay_edges = tri.simplices[:, edge]
        else:
            delaunay_edges = np.vstack((delaunay_edges,
                                        tri.simplices[:, edge]))
    euclidean_weights = np.linalg.norm((xyz[delaunay_edges[:, 0], :]
                                        - xyz[delaunay_edges[:, 1]]),
                                       axis=1)
    delaunay_euclidean_graph = sparse.csr_matrix((euclidean_weights,
                                                  delaunay_edges.T),
                                                 shape=(xyz.shape[0],
                                                        xyz.shape[0]))
    emst = sparse.csgraph.minimum_spanning_tree(delaunay_euclidean_graph,
                                                overwrite=True)
    return emst + emst.T


def _compute_kgraph(xyz, k):
    """Compute a graph whose edge (i, j) is nonzero iff j is in the
    k nearest neighborhood of i or i is in the k-neighborhood of j.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    k : int
        Number of k nearest neighbors used in constructing the
        Riemannian graph used to propagate normal orientation.
    
    Returns
    -------
    scipy.sparse.coo_matrix
        Symmetric graph.
    """
    kgraph = neighbors.kneighbors_graph(xyz, k).tocoo()
    return kgraph + kgraph.T


def _compute_rmst(xyz, n, k, eps=1e-4):
    """Compute the Riemannian minimum spanning tree.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    n : numpy.ndarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    k : int
        Number of k nearest neighbors used in constructing the
        Riemannian graph used to propagate normal orientation.
    eps : float, optional
        Value added to the weight of every edge of the Riemannian
        minimum spanning tree.
        
    Returns
    -------
    sparse.csgraph.minimum_spanning_tree
        The Riemannian minimum spanning tree.
    """
    symmetric_emst = _compute_emst(xyz)
    symmetric_kgraph = _compute_kgraph(xyz, k)
    enriched = (symmetric_emst + symmetric_kgraph).tocoo()

    conn_l = enriched.row
    conn_r = enriched.col
    riemannian_weights = [1 + eps - np.abs(np.dot(n[conn_l[k],:],
                                                  n[conn_r[k], :]))
                          for k in range(len(conn_l))]
    riemannian_graph = sparse.csr_matrix((riemannian_weights,
                                          (conn_l, conn_r)),
                                         shape=(xyz.shape[0],
                                                xyz.shape[0]))
    rmst = sparse.csgraph.minimum_spanning_tree(riemannian_graph,
                                                overwrite=True)
    return rmst + rmst.T


def _acyclic_graph_iterator(graph, seed):
    """Compute the iterator (depth-first) for an unoriented acyclic
    graph.
    
    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        The unoriented acyclic graph.
    seed : int
        The root of the graph.
    
    Returns
    -------
    graph_iterator : iterator
    """
    graph = sparse.csr_matrix(graph)
    stack = LifoQueue()
    stack.put((None, seed))
    while not stack.empty():
        parent, child = stack.get()
        connected_to_child = graph[child, :].nonzero()[1]
        for second_order_child in connected_to_child:
            if second_order_child != parent:
                stack.put((child, second_order_child))
        yield parent, child
        

def orient_normals(xyz, n, k):
    """Orient the normals with respect to consistent tangent planes.
    
    Ref: Hoppe et al., in proceedings of SIGGRAPH 1992, pp. 71-78,
         doi: 10.1145/133994.134011
         
    Note: The code is adjusted version of the implementation of
    an algorithm for consistent propagation of normals in unorganized
    set of points in
    https://github.com/PFMassiani/consistent_normals_orientation.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    n : numpy.ndarray
        The unit normals of shape (N, 3), where N is the number of
        points in the point cloud.
    k : int
        Number of k nearest neighbors used in constructing the
        Riemannian graph used to propagate normal orientation.
    
    Returns
    -------
    numpy.ndarray
        Oriented normals.
    """
    no = n.copy()
    # compute the Riemannian minimum spanning tree
    symmetric_rmst = _compute_rmst(xyz, n, k)
    # set the seed and define the orientation
    seed_idx = np.argmax(xyz[:, 2])
    ez = np.array([0, 0, 1])
    if no[seed_idx, :].T @ ez < 0:
        no[seed_idx, :] *= -1
    # traverse the MST (depth first order) to assign a consistent orientation
    for parent_idx, point_idx in _acyclic_graph_iterator(symmetric_rmst,
                                                         seed_idx):
        if parent_idx is None:
            parent_normal = no[seed_idx, :]
        else:
            parent_normal = no[parent_idx, :]

        if no[point_idx, :] @ parent_normal < 0:
            no[point_idx, :] *= -1
    return no
