import numpy as np
import matplotlib.pyplot as plt

def make_cell(a, b, c):
    """
    Construct a 3D periodic simulation cell from three linearly independent cell vectors(lattice vector).
    Returns cell : ndarray, shape (3, 3)
    Cell vectors stored as rows(row vectors).
    """
    cell = np.array([a, b, c], dtype=float)

    if cell.shape != (3, 3):
        raise ValueError("A 3D cell must contain three 3D vectors.")

    if np.isclose(np.linalg.det(cell), 0.0):
        raise ValueError("Cell vectors must be linearly independent.")

    return cell

def apply_pbc(vector, box_length):
    """
    Apply periodic boundary conditions (PBC).
    :param vector: displacement vector between atoms
    :param box_length: length of the mock cubic lattice cell (unit：Å) 
    :return: corrected vector after applying PBC
    """
    return vector - box_length * np.round(vector / box_length)

#for an center atom A, use find_neighbors to find all its neighbors within cutoff distance
def find_neighbors(positions, center_idx, cutoff, box_length):
    """
    Find neighbors for the center atom.
    :param positions: coordinates of all atoms
    :param center_idx: Index of the center atom
    :param cutoff: Distance threshold for neighbor detection (unit：Å)
    :param box_length: length of the mock cubic lattice cell(unit：Å)
    :return: List of neighbors (indices and relative displacements)
    """
    neighbors = []
    center = positions[center_idx]
    for i, pos in enumerate(positions):
        if i == center_idx:
            continue
        displacement = apply_pbc(pos - center, box_length)
        distance = np.linalg.norm(displacement)
        if distance < cutoff:
            neighbors.append((i, displacement))
    return neighbors

#for an center atom A, use find_neighbors to find all its neighbors within cutoff distance
def find_neighbors_vectorized(positions, center_idx, cutoff, box_length):
    """
    Optimized neighbor search using vectorized linear algebra operations.
    :param positions: Array of particle positions (N x 3).
    :param center_idx: Index of the central particle.
    :param cutoff: Cutoff distance for neighbor searching.
    :param box_length: Length of the cubic simulation box.
    :return: List of tuples (neighbor_idx, displacement).
    """
    # Get the position of the central particle
    center = positions[center_idx]

    # Calculate all displacement vectors (vectorized)
    displacements = positions - center  # N x 3 array

    # Apply periodic boundary conditions to all displacements
    displacements = apply_pbc(displacements, box_length)  # Vectorized PBC

    # Calculate distances for all particles
    distances = np.linalg.norm(displacements, axis=1)  # Compute norms along axis 1

    # Create a mask for neighbors within the cutoff, excluding the central particle itself
    mask = (distances < cutoff) & (np.arange(len(positions)) != center_idx)

    # Extract indices and displacements of neighbors
    neighbor_indices = np.where(mask)[0]
    neighbor_displacements = displacements[mask]

    # Return a list of (neighbor_idx, displacement) tuples
    return list(zip(neighbor_indices, neighbor_displacements))

def find_neighbors_kdtree(positions, center_idx, cutoff, box_length):
    """
    Optimized neighbor search using scipy.spatial.KDTree.
    :param positions: Array of particle positions (N x 3).
    :param center_idx: Index of the central particle.
    :param cutoff: Cutoff distance for neighbor searching.
    :param box_length: Length of the cubic simulation box.
    :return: List of tuples (neighbor_idx, displacement).
    """
    # Apply periodic boundary conditions to handle the infinite box
    extended_positions = extend_positions_with_pbc(positions, box_length)

    # Create a KDTree for fast neighbor search
    kdtree = KDTree(extended_positions)

    # Query all neighbors within the cutoff distance
    center_pos = positions[center_idx]
    neighbor_indices = kdtree.query_ball_point(center_pos, cutoff)

    # Filter out neighbors outside the original simulation box
    neighbors = []
    for idx in neighbor_indices:
        original_idx = idx % len(positions)  # Map back to original index
        if original_idx == center_idx:
            continue
        displacement = apply_pbc(extended_positions[idx] - center_pos, box_length)
        neighbors.append((original_idx, displacement))

    return neighbors

def find_neighbors(positions, center_idx, cutoff, box_length):
    """
    Optimized neighbor search using the cell-linked list method.
    :param positions: Coordinates of all atoms.
    :param center_idx: Index of the center atom.
    :param cutoff: Distance threshold for neighbor detection (unit: Å).
    :param box_length: Length of the cubic simulation box (unit: Å).
    :return: List of neighbors (indices and relative displacements).
    """
    # Define cell size and number of cells
    cell_size = cutoff
    num_cells = int(np.floor(box_length / cell_size))
    cell_size = box_length / num_cells

    # Initialize cell-linked lists
    cells = {}
    for idx, pos in enumerate(positions):
        cell_idx = tuple((pos // cell_size).astype(int) % num_cells)
        if cell_idx not in cells:
            cells[cell_idx] = []
        cells[cell_idx].append(idx)

    # Identify the center atom's cell
    center_pos = positions[center_idx]
    center_cell_idx = tuple((center_pos // cell_size).astype(int) % num_cells)

    # Define neighbor cell offsets (27 neighbor cells in 3D)
    neighbor_offsets = [
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    ]

    # Collect neighbors
    neighbors = []
    for offset in neighbor_offsets:
        neighbor_cell_idx = tuple((np.array(center_cell_idx) + np.array(offset)) % num_cells)
        if neighbor_cell_idx in cells:
            for neighbor_idx in cells[neighbor_cell_idx]:
                if neighbor_idx == center_idx:
                    continue
                displacement = apply_pbc(positions[neighbor_idx] - center_pos, box_length)
                distance = np.linalg.norm(displacement)
                if distance < cutoff:
                    neighbors.append((neighbor_idx, displacement))

    return neighbors


#Calculation of the angle between two vectors v1 and v2.
def calculate_angle(v1, v2):
    """
    Calculate the bond angle between two vectors (unit:degrees).
    :param v1: The first vector
    :param v2: The second vector
    :return: angle(degrees)
    """
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

     # Handle numerical errors explicitly for parallel cases
    if np.isclose(cos_theta, 1.0, atol=1e-8):  # Theta = 0 degrees
        return 0.0
    elif np.isclose(cos_theta, -1.0, atol=1e-8):  # Theta = 180 degrees
        return 180.0

    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def bond_angle_distribution(positions, box_length, cutoff):
    """
    Calculate the bond angle distribution in a Mock Lattice.
    :param positions: Particle coordinations (N, 3)
    :param box_length:  length of the mock cubic lattice cell(unit：Å) 
    :param cutoff: Distance threshold (unit：Å)
    :return: bond_angle_distributions
    """
    angles = []
    for center_idx in range(len(positions)):
        neighbors = find_neighbors(positions, center_idx, cutoff, box_length)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                _, v1 = neighbors[i]
                _, v2 = neighbors[j]
                angles.append(calculate_angle(v1, v2))
    return angles


def plot_bond_angle_distribution(angles):
    """
    Plot the bond angle distribution
    :param angles: List of bond angles
    """
    # Calculate and normalzie Probability density P, and create histogram of theta vs. P
    bins = np.linspace(0, 180, 50)
    hist, bin_edges = np.histogram(angles, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    plt.plot(bin_centers, hist, label="P(θ)", color='blue')
    plt.xlabel("Bond Angle (degrees)")
    plt.ylabel("Probability Density")
    plt.title("Bond Angle Distribution in Mock Lattice")
    plt.grid()
    plt.legend()
    plt.show()