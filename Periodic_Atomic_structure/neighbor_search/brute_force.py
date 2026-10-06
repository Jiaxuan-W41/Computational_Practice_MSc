import numpy as np
from Periodic_boundary_condition import pbc

def find_neighbors_brute_force(positions, center_idx, cutoff, cell):
    """
    Find all neighbours of a central particle using a brute-force search(enumeration), where

    positions : ndarray, shape (N, 3), Cartesian coordinates of all particles.
    center_idx : int, Index of the central particle.
    cutoff : float, Neighbour cutoff distance in angstroms.
    cell : ndarray, shape (3, 3), Simulation-cell matrix with lattice vectors stored as rows.

    Returns
    neighbors : list, List of (neighbor_idx, displacement) tuples.
    """

    neighbors = []
    center = positions[center_idx]

    for idx, position in enumerate(positions):

        if idx == center_idx:
            continue

        displacement = position - center
        displacement = pbc(displacement, cell)

        distance = np.linalg.norm(displacement)

        if distance < cutoff:
            neighbors.append((idx, displacement))

    return neighbors