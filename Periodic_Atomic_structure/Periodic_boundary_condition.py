import numpy as np

def pbc(displacement, cell):
    """
    This function applies periodic boundary conditions to a Cartesian displacement vector
    and finds the shortest equivalent displacement under periodicity, where

    displacement : array_like, shape (3,) Cartesian displacement vector between two atoms.
    cell : ndarray, shape (3, 3) Cell vectors stored as rows.

    Returns
    -------
    ndarray, shape (3,)
        Minimum-image displacement vector in Cartesian coordinates.
    """
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)

    return fractional @ cell