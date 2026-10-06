import numpy as np
from Periodic_boundary_condition import pbc


def find_neighbors_vectorized(positions, center_idx, cutoff, cell):
    """
    Find all neighbours of a central particle using vectorized NumPy operations.
    """

    center = positions[center_idx]

    # Displacements from the central particle to all particles
    displacements = positions - center

    # Apply periodic boundary conditions
    displacements = pbc(displacements, cell)

    # Distances from the central particle
    distances = np.linalg.norm(displacements, axis=1)

    # Select particles inside the cutoff and exclude the central particle
    mask = (
        (distances < cutoff)
        & (np.arange(len(positions)) != center_idx)
    )

    neighbor_indices = np.where(mask)[0]
    neighbor_displacements = displacements[mask]

    return list(zip(neighbor_indices, neighbor_displacements))