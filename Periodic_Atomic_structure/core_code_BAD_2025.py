import numpy as np
import matplotlib.pyplot as plt

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