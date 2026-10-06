import numpy as np
from Periodic_boundary_condition import pbc


def build_cell_list(positions, cutoff, cell):
    """
    Build a cell-linked list for neighbour searching.

    This implementation assumes an orthogonal simulation cell.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian coordinates of all particles.
    cutoff : float
        Neighbour cutoff distance in angstroms.
    cell : ndarray, shape (3, 3)
        Simulation-cell matrix with cell vectors stored as rows.

    Returns
    -------
    cells : dict
        Dictionary mapping each linked-cell index to the particle
        indices contained in that cell.
    num_cells : ndarray, shape (3,)
        Number of linked cells along each direction.
    cell_size : ndarray, shape (3,)
        Actual size of each linked cell.
    """

    # Length of the simulation box along each direction
    box_lengths = np.linalg.norm(cell, axis=1)

    # Choose linked-cell size to be at least the cutoff
    num_cells = np.maximum(
        1,
        np.floor(box_lengths / cutoff).astype(int)
    )

    cell_size = box_lengths / num_cells

    cells = {}

    for idx, position in enumerate(positions):

        cell_idx = tuple(
            (
                np.floor(position / cell_size).astype(int)
            )
            % num_cells
        )

        if cell_idx not in cells:
            cells[cell_idx] = []

        cells[cell_idx].append(idx)

    return cells, num_cells, cell_size

def find_neighbors_cell_list(
    positions,
    center_idx,
    cutoff,
    cell,
    cells,
    num_cells,
    cell_size
):
    """
    Find neighbours of a central particle using a pre-built cell list.

    Returns
    -------
    neighbors : list
        List of (neighbor_idx, displacement) tuples.
    """

    center = positions[center_idx]

    center_cell_idx = tuple(
        (
            np.floor(center / cell_size).astype(int)
        )
        % num_cells
    )

    # All neighbouring linked cells, including the central cell
    neighbor_cells = set()

    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):

                offset = np.array([i, j, k])

                neighbor_cell_idx = tuple(
                    (
                        np.array(center_cell_idx) + offset
                    )
                    % num_cells
                )

                neighbor_cells.add(neighbor_cell_idx)

    neighbors = []

    for neighbor_cell_idx in neighbor_cells:

        for neighbor_idx in cells.get(neighbor_cell_idx, []):

            if neighbor_idx == center_idx:
                continue

            displacement = positions[neighbor_idx] - center
            displacement = pbc(displacement, cell)

            distance = np.linalg.norm(displacement)

            if distance < cutoff:
                neighbors.append((neighbor_idx, displacement))

    return neighbors