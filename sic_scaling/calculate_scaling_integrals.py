import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Dict, Set

import ase.io.cube
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.io import write
from ase.units import Bohr
from gpaw import restart

from .utils import compute_gradient, density_from_orbitals, get_consistent_densities_and_occupations

DENSITY_THRESHOLD = 1e-10


def compute_scaled_gradient(rho_i: np.ndarray, grad_rho_i: np.ndarray) -> np.ndarray:
    """
    Compute the scaled gradient s_i.

    Parameters:
        rho_i (ndarray): Orbital density.
        grad_rho_i (ndarray): Gradient magnitude of the orbital density.

    Returns:
        s_i (ndarray): Scaled gradient.
    """
    factor = 2 * (3 * np.pi)**(1/3)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(rho_i > DENSITY_THRESHOLD, grad_rho_i / (factor * rho_i**(4/3)), 0.0)


def compute_density_ratio(rho_i: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Compute the ratio of the orbital density to the total electron density.

    Parameters:
        rho_i (ndarray): Orbital density.
        rho (ndarray): Total electron density.

    Returns:
        ratio (ndarray): Ratio of the orbital density to the total electron density.
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(rho > DENSITY_THRESHOLD, rho_i / rho, 0.0)


def compute_f_scaling_function(rho_i: np.ndarray, rho: np.ndarray, a: float, s_i: np.ndarray) -> np.ndarray:
    """
    Compute the scaling function f(rho_i, rho).

    Parameters:
        rho_i (ndarray): Orbital density.
        rho (ndarray): Total electron density (calculated on the same grid).
        a (float): Scaling factor parameter.
        s_i (ndarray): Scaled gradient.

    Returns:
        f (ndarray): Scaling function.
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = compute_density_ratio(rho_i, rho)
        if np.any(ratio > 1):
            print(
                f"Warning: Ratio of rho_i to rho exceeds 1. Maximum value: {ratio.max()}")

        return (1 - ratio) / (1 + a * s_i**2)
        # return np.where(rho > DENSITY_THRESHOLD, (1 - ratio)*rho_i / (1 + a * s_i**2), 0.0)


def compute_g_scaling_function(f: np.ndarray, rho_i: np.ndarray) -> np.ndarray:
    """
    Compute the scaling function g(f, rho_i).

    Parameters:
        f (ndarray): The scaling function values.
        rho_i (ndarray): Orbital density.

    Returns:
        g (ndarray): The g scaling function values.
    """
    return f * rho_i


def integrate_scaling_function(f: np.ndarray, dv: float) -> float:
    """
    Integrate the scaling function over the grid.

    Parameters:
        f (ndarray): The scaling function values on the grid.
        dv (float): Volume element in Bohr^3 for integration.

    Returns:
        integral_value (float): The integral of the scaling function over the grid.
    """
    return np.nansum(f) * dv


def analyze_scaling_function(f: np.ndarray, rho: np.ndarray, rho_i: np.ndarray, grad_rho_i: np.ndarray, grid_spacings: List[float]) -> None:
    """
    Analyze the scaling function to check for values above 1, print the most important points,
    and plot distributions of rho, rho_i, and grad_rho_i for points where f > 1, with full distributions for comparison.
    Also adds plots for f distribution and rho_i / rho, and prints the number and fraction of failed points.

    Parameters:
        f (ndarray): The scaling function values on the grid.
        rho (ndarray): Total electron density values.
        rho_i (ndarray): Orbital density values.
        grad_rho_i (ndarray): Gradient of the orbital density.
        grid_spacings (list): List of grid spacings for each axis.

    Returns:
        None
    """
    max_f = np.max(f)
    num_above_one = np.sum(f > 1)
    total_points = f.size
    fraction_above_one = num_above_one / total_points

    # Print the number of points where f > 1 and the fraction of total points
    print(f"Warning: {num_above_one} points where f > 1 detected.")
    print(f"Fraction of points where f > 1: {fraction_above_one:.6%}")
    print(f"Maximum value of f: {max_f:.6f}")

    if num_above_one > 0:
        # Find indices where f > 1
        indices_above_one = np.where(f > 1)

        # Extract corresponding rho, rho_i, and grad_rho_i values
        f_above_one = f[indices_above_one]
        rho_above_one = rho[indices_above_one]
        rho_i_above_one = rho_i[indices_above_one]
        grad_rho_i_above_one = grad_rho_i[indices_above_one]

        # Find the indices of the most important points among f > 1
        max_f_idx = np.argmax(f_above_one)
        max_rho_idx = np.argmax(rho_above_one)
        max_rho_i_idx = np.argmax(rho_i_above_one)
        max_grad_rho_i_idx = np.argmax(grad_rho_i_above_one)

        # Print the most important points
        print("\nMost important points among those where f > 1:")
        print(f"  Max f: f={f_above_one[max_f_idx]: .6e}, rho={rho_above_one[max_f_idx]: .6e}, rho_i={rho_i_above_one[max_f_idx]: .6e}, "
              f"grad_rho_i={grad_rho_i_above_one[max_f_idx]: .6e}")
        print(f"  Max rho: f={f_above_one[max_rho_idx]: .6e}, rho={rho_above_one[max_rho_idx]: .6e}, rho_i={rho_i_above_one[max_rho_idx]: .6e}, "
              f"grad_rho_i={grad_rho_i_above_one[max_rho_idx]: .6e}")
        print(f"  Max rho_i: f={f_above_one[max_rho_i_idx]: .6e}, rho={rho_above_one[max_rho_i_idx]: .6e}, rho_i={rho_i_above_one[max_rho_i_idx]: .6e}, "
              f"grad_rho_i={grad_rho_i_above_one[max_rho_i_idx]: .6e}")
        print(f"  Max grad_rho_i: f={f_above_one[max_grad_rho_i_idx]: .6e}, rho={rho_above_one[max_grad_rho_i_idx]: .6e}, rho_i={rho_i_above_one[max_grad_rho_i_idx]: .6e}, "
              f"grad_rho_i={grad_rho_i_above_one[max_grad_rho_i_idx]: .6e}")

        # Define log-scaled bins
        min_rho, max_rho = np.min(rho[rho > 0]), np.max(rho)
        min_rho_i, max_rho_i = np.min(rho_i[rho_i > 0]), np.max(rho_i)
        min_grad_rho_i, max_grad_rho_i = np.min(
            grad_rho_i[grad_rho_i > 0]), np.max(grad_rho_i)

        bins_rho = np.logspace(np.log10(min_rho), np.log10(max_rho), 100)
        bins_rho_i = np.logspace(np.log10(min_rho_i), np.log10(max_rho_i), 100)
        bins_grad_rho_i = np.logspace(
            np.log10(min_grad_rho_i), np.log10(max_grad_rho_i), 100)
        bins_f = np.logspace(
            np.log10(np.min(f[f > 0])), np.log10(np.max(f)), 100)

        # Compute rho_i / rho and define bins for it
        ratio_rho_i_rho = compute_density_ratio(rho_i, rho)
        print(f'{ratio_rho_i_rho.min()=} {ratio_rho_i_rho.max()=}')
    else:
        print("No points where f > 1 detected.")


def calculate_scaling_integral(
    atoms: Atoms,
    orbital_index: int,
    grid_spacings: List[float],
    rho: np.ndarray,
    orbital_densities: List[List[np.ndarray]],
    f_n_s: Tuple[np.ndarray, np.ndarray],
    dv: float,
    spin: int = 0,
    a: float = 0.5,
    prenormalize: bool = False,
    savedir: Optional[str] = None
) -> Tuple[float, float]:
    """
    Calculate the scaling integral for a single orbital.

    Parameters:
        atoms (Atoms): ASE atoms object representing the molecule.
        orbital_index (int): Index of the orbital to compute scaling integral for.
        grid_spacings (list): List of grid spacings for each axis.
        rho (ndarray): Total electron density.
        orbital_densities (list of lists of ndarray): Orbital densities for all spin channels.
        f_n_s (tuple of ndarray): Occupation numbers for all spin channels.
        dv (float): Volume element in Bohr^3 for integration.
        spin (int, optional): Spin channel (0 for alpha, 1 for beta). Default is 0.
        a (float, optional): Scaling parameter. Default is 0.5.
        prenormalize (bool, optional): Whether to prenormalize the orbital densities. Default is False.
        savedir (str or None, optional): Directory to save output files. Default is None.

    Returns:
        Tuple[float, float]: A tuple containing:
            - integral_value (float): The integral of the scaling function.
            - coefficient_value (float): The computed coefficient value (1 - integral_value).
    """

    print(f'{len(orbital_densities[0])=}')
    print(f'{orbital_densities[0][0].shape=}')
    print(f'{grid_spacings=}')
    print(f'{rho.shape=}')
    print(f'{f_n_s=}')
    print(f'{dv=}')

    # Calculate the total volume of the system
    grid_shape = rho.shape
    system_volume = dv * np.prod(grid_shape)
    print(f"System volume: {system_volume:.6f} Bohr^3")

    rho_i = orbital_densities[spin][orbital_index]

    # Compute the gradient of the orbital density
    grad_rho_i = compute_gradient(rho_i, grid_spacings)

    # Compute the scaled gradient
    s_i = compute_scaled_gradient(rho_i, grad_rho_i)

    # Compute the scaling function
    f = compute_f_scaling_function(
        rho_i=rho_i, rho=rho, a=a, s_i=s_i)

    # Analyze the scaling function, passing additional data
    analyze_scaling_function(f, rho, rho_i, grad_rho_i, grid_spacings)

    g = compute_g_scaling_function(f, rho_i)

    integral_value = integrate_scaling_function(g, dv)

    # Compute the normalized integral (integral divided by system volume)
    coefficient_value = 1-integral_value

    return integral_value, coefficient_value


def calculate_scaling_integrals(
    orbitals: List[int],
    spin: int = 0,
    gpw_file: Optional[str] = None,
    atoms: Optional[Atoms] = None,
    calc: Optional[Any] = None,
    density_type: str = 'sum_rho_i',
    occupation_method: str = 'keep',
    a: float = 0.5,
    prenormalize: bool = False,
    savedir: Optional[str] = None,
    uks: bool = True
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """
    Calculate the integral of the scaling function for a given list of orbitals.

    Parameters:
        orbitals (list): List of orbital indices to compute scaling integrals for.
        spin (int, optional): Spin channel (0 for alpha, 1 for beta). Default is 0.
        gpw_file (str, optional): Path to the GPAW gpw file. Default is None.
        atoms (Atoms, optional): ASE atoms object. Default is None.
        calc (GPAW calculator, optional): GPAW calculator object. Default is None.
        density_type (str, optional): The type of density to use ('all_electron', 'pseudo', 'sum_rho_i'). Default is 'sum_rho_i'.
        occupation_method (str, optional): Method to handle orbital occupations ('keep', 'imitate_excitation', 'all'). Default is 'keep'.
        a (float, optional): Scaling parameter. Default is 0.5.
        prenormalize (bool, optional): Whether to prenormalize the orbital densities. Default is False.
        savedir (str or None, optional): Directory to save output files. Default is None.
        uks (bool, optional): Whether unrestricted Kohn-Sham is used. Default is True.

    Returns:
        Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
            - integral_values: Dictionary with (spin, orbital_index) tuples as keys and integral values as values.
            - coefficient_values: Dictionary with (spin, orbital_index) tuples as keys and coefficient values as values.
    """
    assert gpw_file is not None or (
        atoms is not None and calc is not None), "Either 'gpw_file' or 'atoms' and 'calc' must be provided."

    if atoms is None or calc is None:
        atoms, calc = restart(gpw_file, txt=None)

    grid_spacings = [
        spacing * Bohr for spacing in calc.wfs.gd.get_grid_spacings()]
    # Calculate volume element (dv) for integration
    dv = np.prod(grid_spacings)

    integral_values = {}
    coefficient_values = {}
    if uks:
        kpts = calc.wfs.kpt_u[0], calc.wfs.kpt_u[1]
    else:
        kpts = calc.wfs.kpt_u[0], calc.wfs.kpt_u[0]

    original_f_n_s = kpts[0].f_n.copy(), kpts[1].f_n.copy()
    assert len(original_f_n_s[0]) == len(original_f_n_s[1])
    number_of_bands = len(original_f_n_s[0])
    ks = [0, 1] if uks else [0, 0]
    original_orbital_densities = [[density_from_orbitals(calc.get_pseudo_wave_function(band=i, spin=k), dv=dv, prenormalize=prenormalize)
                                   for i in range(number_of_bands)]
                                  for k in ks]

    for orbital_index in orbitals:
        print(f"=====\nProcessing orbital {orbital_index}...\n=====")

        # output_dir = Path(savedir)
        # output_dir.mkdir(exist_ok=True, parents=True)
        # cube_filename = output_dir / f"density_{orbital_index}.cube"
        # write(cube_filename, atoms,
        #       data=original_orbital_densities[orbital_index])

        rho, orbital_densities, f_n_s = get_consistent_densities_and_occupations(
            orbital_densities=original_orbital_densities,
            f_n_s=original_f_n_s,
            orbital_index=orbital_index,
            spin=spin,
            density_type=density_type,
            occupation_method=occupation_method,
            dv=dv,
            prenormalize=prenormalize)

        integral_value, coefficient_value = calculate_scaling_integral(
            atoms=atoms,
            orbital_index=orbital_index,
            grid_spacings=grid_spacings,
            rho=rho,
            orbital_densities=orbital_densities,
            f_n_s=f_n_s,
            dv=dv,
            spin=spin,
            a=a,
            prenormalize=prenormalize,
            savedir=savedir)

        print(f"""Completed orbital {orbital_index} with integral
            {integral_value:.6f} and coefficient {coefficient_value:.6f}.""")

        integral_values[(spin, orbital_index)] = integral_value
        coefficient_values[(spin, orbital_index)] = coefficient_value

    return integral_values, coefficient_values


if __name__ == '__main__':
    # filename = 'alec_redo/NH3/excited_4_triplet/triplet.gpw'
    # atoms, calc = restart(filename, txt=None)
    # orbitals = [0, 1, 2, 3, 5]
    # uks = True
    # spin = 1
    # occupation_method='keep'

    filename = 'alec_redo/NH3/excited_4_triplet/triplet.gpw'
    atoms, calc = restart(filename, txt=None)
    orbitals = [0, 1, 2, 3, 5]
    uks = True
    spin = 1
    occupation_method = 'imitate_excitation'

    integral_values, coefficient_values = calculate_scaling_integrals(orbitals, gpw_file=filename, atoms=atoms, calc=calc, density_type='sum_rho_i',
                                                                      occupation_method=occupation_method, spin=spin, uks=uks, a=0.2, prenormalize=True)
    print(f'{coefficient_values=}')
