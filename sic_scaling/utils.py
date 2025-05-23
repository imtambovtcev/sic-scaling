import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ase.atoms import Atoms
from ase.units import Bohr
from gpaw import restart


def compute_gradient(density: np.ndarray, grid_spacings: List[float]) -> np.ndarray:
    """
    Compute the gradient of the density using central finite differences.

    Parameters:
    density (ndarray): 3D array of density values.
    grid_spacings (list of floats): Grid spacing in Bohr for each axis (dx, dy, dz).

    Returns:
    grad_rho (ndarray): Magnitude of the gradient of the density.
    """
    grad_rho = np.zeros_like(density)
    for axis in range(3):
        grad_rho += np.gradient(density, grid_spacings[axis], axis=axis)**2
    return np.sqrt(grad_rho)


def normalize_density(rho_i: np.ndarray, dv: float, normalize: bool = True) -> np.ndarray:
    """
    Normalize the electron density over the grid so that its integral equals 1.
    Computes the normalization integral even if there is no request for normalization for debugging purposes.

    Parameters:
    rho_i (ndarray): Orbital electron density to normalize.
    dv (float): Volume element in Bohr^3 for integration.
    normalize (bool): Whether to perform normalization. Default is True.

    Returns:
    rho_i_norm (ndarray): Normalized electron density.
    """
    if not np.all(rho_i >= 0):
        print(
            f"Warning: Negative orbital density values detected. The lowest value is: {rho_i.min()}. Setting negative values to zero.")
        rho_i = np.maximum(rho_i, 0.0)

    integral = np.sum(rho_i) * dv
    print(f"Normalization integral: {integral:.6f}" +
          (' (Not in use)' if not normalize else ''))

    assert integral > sys.float_info.epsilon, f"Integral of the density is {integral}. Cannot normalize."
    if normalize:
        return rho_i / integral
    else:
        return rho_i


def calculate_density_center_gpw(calc: Optional[Any] = None, gpw_file: Optional[str] = None) -> np.ndarray:
    """
    Calculate the center of density from either all-electron density or pseudo-density.

    Parameters:
    calc (GPAW calculator object, optional): Preloaded GPAW calculator object.
    gpw_file (str, optional): Path to the GPAW gpw file.

    Returns:
    center_of_density (numpy array): The coordinates of the center of the electron density.
    """

    # Load GPAW calculation if a gpw_file is provided
    if gpw_file is not None:
        atoms, calc = restart(gpw_file, txt=None)
    elif calc is None:
        raise ValueError("Either 'calc' or 'gpw_file' must be provided.")

    # Retrieve the density grid
    rho = calc.get_all_electron_density(gridrefinement=1)

    if not np.all(rho >= 0):
        print(
            f"Warning: Negative density values detected. The lowest value is: {rho.min()}. Setting negative values to zero.")
        rho = np.maximum(rho, 0)

    # Get grid spacings and shape
    grid_spacings = np.array(
        [spacing * Bohr for spacing in calc.wfs.gd.get_grid_spacings()])
    grid_shape = rho.shape

    # Create a grid of coordinates for each axis
    x = np.arange(grid_shape[0]) * grid_spacings[0]
    y = np.arange(grid_shape[1]) * grid_spacings[1]
    z = np.arange(grid_shape[2]) * grid_spacings[2]

    # Compute meshgrid for the entire 3D grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Compute the center of density using the density as weights
    total_density = np.sum(rho)

    center_x = np.sum(X * rho) / total_density
    center_y = np.sum(Y * rho) / total_density
    center_z = np.sum(Z * rho) / total_density

    center_of_density = np.array([center_x, center_y, center_z])

    return center_of_density


def calculate_density_center_molecule(calc: Optional[Any] = None, gpw_file: Optional[str] = None) -> np.ndarray:
    """
    Calculate the center of density from either all-electron density or pseudo-density.

    Parameters:
    calc (GPAW calculator object, optional): Preloaded GPAW calculator object.
    gpw_file (str, optional): Path to the GPAW gpw file.

    Returns:
    center_of_density (numpy array): The coordinates of the center of the electron density.
    """

    # Load GPAW calculation if a gpw_file is provided
    if gpw_file is not None:
        atoms, calc = restart(gpw_file, txt=None)
    elif calc is None:
        raise ValueError("Either 'calc' or 'gpw_file' must be provided.")

    # Retrieve the density grid
    rho = calc.get_all_electron_density(gridrefinement=1)

    if not np.all(rho >= 0):
        print(
            f"Warning: Negative density values detected. The lowest value is: {rho.min()}. Setting negative values to zero.")
        rho = np.maximum(rho, 0)

    # Get grid spacings and shape
    grid_spacings = np.array(
        [spacing * Bohr for spacing in calc.wfs.gd.get_grid_spacings()])
    grid_shape = rho.shape

    # Create a grid of coordinates for each axis
    x = np.arange(grid_shape[0]) * grid_spacings[0]
    y = np.arange(grid_shape[1]) * grid_spacings[1]
    z = np.arange(grid_shape[2]) * grid_spacings[2]

    # Compute meshgrid for the entire 3D grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Compute the center of density using the density as weights
    total_density = np.sum(rho)

    center_x = np.sum(X * rho) / total_density
    center_y = np.sum(Y * rho) / total_density
    center_z = np.sum(Z * rho) / total_density

    center_of_density = np.array([center_x, center_y, center_z])

    return center_of_density


def density_from_orbitals(orbital: np.ndarray, dv: float, prenormalize: bool) -> np.ndarray:
    """
    Calculate density from orbital wavefunction.

    Parameters:
    orbital (ndarray): Orbital wavefunction.
    dv (float): Volume element in Bohr^3 for integration.
    prenormalize (bool): Whether to normalize the resulting density.

    Returns:
    rho_i (ndarray): Orbital electron density.
    """
    rho_i = orbital.conj() * orbital
    return normalize_density(rho_i, dv=dv, normalize=prenormalize)


def get_consistent_densities_and_occupations(
    orbital_densities: List[List[np.ndarray]],
    f_n_s: Tuple[np.ndarray, np.ndarray],
    orbital_index: int,
    dv: float,
    spin: int = 0,
    rho: Optional[np.ndarray] = None,
    density_type: str = 'sum_rho_i',
    occupation_method: str = 'keep',
    prenormalize: bool = False
) -> Tuple[np.ndarray, List[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Utility function to compute the electron density based on the specified method.

    Parameters:
    orbital_densities (list of lists of ndarray): Orbital densities for all spin channels.
    f_n_s (tuple of ndarray): Occupation numbers for all spin channels.
    orbital_index (int): The index of the orbital to compute the density for.
    dv (float): Volume element in Bohr^3 for integration.
    spin (int, optional): Spin channel index (0 for alpha, 1 for beta). Default is 0.
    rho (ndarray, optional): Pre-computed electron density. Required if density_type is 'given'.
    density_type (str, optional): The type of density to compute. Options are:
                               - 'given' : uses the given density.
                               - 'sum_rho_i' : sums densities of occupied orbitals.
                               Default is 'sum_rho_i'.
    occupation_method (str, optional): The method to determine the occupation numbers. Options are:
                                    - 'keep' : keeps the current occupation numbers.
                                    - 'imitate_excitation' : imitates excitation of the orbital.
                                    - 'all' : sets all occupations to 1.
                                    Default is 'keep'.
    prenormalize (bool, optional): Whether to prenormalize the orbital densities. Default is False.

    Returns:
    tuple containing:
        - rho (ndarray): The computed electron density on the grid.
        - orbital_densities (list of lists of ndarray): The electron densities of all orbitals.
        - f_n_s (tuple of ndarray): The occupation numbers of the orbitals.
    """
    print(f"Computing {occupation_method} occupation numbers...")
    f_n_s = np.array(f_n_s).copy()
    print(f'{f_n_s.shape=}')
    print(f"Occupation numbers: {f_n_s}")
    number_of_bands = len(f_n_s)
    number_of_electrons = np.sum(f_n_s)

    # Cast number_of_electrons // 2 to integer
    homo = int(number_of_electrons // 2)-1

    print(f"Number of bands: {number_of_bands}")
    print(f"Number of electrons: {number_of_electrons}")
    print(f'HOMO: {homo}')

    if occupation_method == 'keep':
        assert f_n_s[spin, orbital_index] > 0, f"Orbital {spin,
                                                          orbital_index} is unoccupied but the calculation is performed for this orbital."
    elif occupation_method == 'imitate_excitation':
        if f_n_s[spin, orbital_index] == 0:
            highest_occupied_in_channel = np.max(
                np.where(f_n_s[spin, :] >= 1)[0])
            f_n_s[spin, highest_occupied_in_channel] -= 1
            f_n_s[spin, orbital_index] += 1
            if orbital_index <= homo:
                print(f'Orbital {orbital_index} is in ground occupation')
        else:
            print(f'Orbital {spin, orbital_index} was already occupied')
    elif occupation_method == 'all':
        f_n_s[:] = 1
    else:
        raise ValueError(f'occupation_method is unknown: {occupation_method}')

    print(f'Final occupations: {f_n_s}')

    highest_occupied_orbital = np.max(
        np.where(np.sum(f_n_s, axis=0) > 0))  # Number of occupied orbitals
    print(f"Highest occupied orbital: {highest_occupied_orbital}")

    print(f"Computing {density_type} density...")

    if density_type == 'given':
        if rho is None:
            raise ValueError(
                "Density type is 'given' but no density 'rho' is provided.")
        total_density = rho

    elif density_type == 'sum_rho_i':
        total_density = np.zeros_like(orbital_densities[0][0])
        occupied = np.where(f_n_s > 0)
        for k, i in zip(list(occupied[0]), list(occupied[1])):
            rho_i = orbital_densities[k][i]
            total_density += f_n_s[k, i] * rho_i

    else:
        raise ValueError(
            "Invalid density_type. Choose 'all_electron', 'pseudo', or 'sum_rho_i'.")

    if not np.all(total_density >= 0):
        print(f"Warning: Negative electron density values detected. The lowest value is : {
              total_density.min()}. Setting negative values to zero.")
        total_density = np.maximum(total_density, 0.0)

    number_of_electrons_from_density = np.sum(total_density) * dv
    print(f"Number of electrons from density: {
          number_of_electrons_from_density}")

    return total_density, orbital_densities, f_n_s


def write_cube(file_obj: Union[str, Any], atoms: Atoms, data: Optional[np.ndarray] = None, origin: Optional[List[float]] = None, comment: Optional[str] = None) -> None:
    """Function to write a cube file.

    Parameters:
    file_obj (str or file object): File to which output is written.
    atoms (Atoms): The Atoms object specifying the atomic configuration.
    data (3-dim numpy array, optional): Array containing volumetric data as e.g. electronic density.
                                       Default is None, which will create a unity array.
    origin (list of 3 floats, optional): Origin of the volumetric data (units: Angstrom).
                                       Default is None, which will use [0, 0, 0].
    comment (str, optional): Comment for the first line of the cube file.
                           Default is None, which will generate a timestamp comment.
    """

    if data is None:
        data = np.ones((2, 2, 2))
    data = np.asarray(data)

    if data.dtype == complex:
        data = np.abs(data)

    if comment is None:
        comment = "Cube file from ASE, written on " + time.strftime("%c")
    else:
        comment = comment.strip()
    file_obj.write(comment)

    file_obj.write("\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")

    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin) / Bohr

    file_obj.write(
        "{:5}{:12.6f}{:12.6f}{:12.6f}\n".format(
            len(atoms), *origin))

    for i in range(3):
        n = data.shape[i]
        d = atoms.cell[i] / n / Bohr
        file_obj.write("{:5}{:12.6f}{:12.6f}{:12.6f}\n".format(n, *d))

    positions = atoms.positions / Bohr
    numbers = atoms.numbers
    for Z, (x, y, z) in zip(numbers, positions):
        file_obj.write(
            "{:5}{:12.6f}{:12.6f}{:12.6f}{:12.6f}\n".format(
                Z, 0.0, x, y, z)
        )

    # Increase precision to 10 significant digits in scientific notation
    data.tofile(file_obj, sep="\n", format="%.16e")
