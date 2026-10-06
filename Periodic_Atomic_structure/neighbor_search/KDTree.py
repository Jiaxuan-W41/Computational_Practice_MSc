import numpy as np
from scipy.spatial import KDTree
from Periodic_boundary_condition import pbc


def build_kdtree(positions, cell):
    """
    Build a KD-tree using periodic images of the particle positions.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian coordinates of all particles.
    cell : ndarray, shape (3, 3)
        Simulation-cell matrix with lattice vectors stored as rows.

    Returns
    -------
    kdtree : scipy.spatial.KDTree
        KD-tree constructed from the extended periodic positions.
    extended_positions : ndarray
        Particle positions including neighbouring periodic images.
    original_indices : ndarray
        Original particle index corresponding to each extended position.
    """

    extended_positions = []
    original_indices = []

    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):

                translation = (
                    i * cell[0]
                    + j * cell[1]
                    + k * cell[2]
                )

                translated_positions = positions + translation

                extended_positions.append(translated_positions)
                original_indices.extend(range(len(positions)))

    extended_positions = np.vstack(extended_positions)
    original_indices = np.array(original_indices)

    kdtree = KDTree(extended_positions)

    return kdtree, extended_positions, original_indices

def find_neighbors_kdtree(
    positions,
    center_idx,
    cutoff,
    cell,
    kdtree,
    extended_positions,
    original_indices
):
    """
    Find neighbours of a central particle using a pre-built KD-tree.

    Returns
    -------
    neighbors : list
        List of (neighbor_idx, displacement) tuples.
    """

    center = positions[center_idx]

    candidate_indices = kdtree.query_ball_point(center, cutoff)

    neighbors = {}

    for extended_idx in candidate_indices:

        original_idx = original_indices[extended_idx]

        if original_idx == center_idx:
            continue

        displacement = extended_positions[extended_idx] - center
        displacement = pbc(displacement, cell)

        distance = np.linalg.norm(displacement)

        if distance < cutoff:
            if (
                original_idx not in neighbors
                or distance < neighbors[original_idx][0]
            ):
                neighbors[original_idx] = (
                    distance,
                    displacement
                )

    return [
        (neighbor_idx, data[1])
        for neighbor_idx, data in neighbors.items()
    ]