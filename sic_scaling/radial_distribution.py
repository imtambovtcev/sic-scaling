from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from ase.units import Bohr
from gpaw import restart

from chem_utils import Molecule

from .utils import calculate_density_center_gpw, normalize_density

# from load_full_molecule import load_full_molecule



class RadialDistribution(ABC):

    @abstractmethod
    def get_electron_density(self) -> np.ndarray:
        """
        Get the electron density for the system.

        Returns:
            np.ndarray: The electron density array.
        """
        pass

    @abstractmethod
    def get_grid_spacings(self) -> List[float]:
        """
        Get the grid spacings for the calculation.

        Returns:
            List[float]: Grid spacings in x, y, and z directions.
        """
        pass

    @abstractmethod
    def get_orbital(self, orbital_index: int, spin: int = 0) -> np.ndarray:
        """
        Get the orbital wave function for a given index.

        Parameters:
            orbital_index (int): Index of the orbital to retrieve.
            spin (int, optional): Spin channel (0 for alpha, 1 for beta). Default is 0.

        Returns:
            np.ndarray: Orbital wave function.
        """
        pass

    def setup_grid(self) -> None:
        """
        Set up the grid and volume element for the calculations.
        """
        rho = self.get_electron_density()

        grid_spacings = self.get_grid_spacings()
        self.dv = np.prod(grid_spacings)

        # Create a grid of distances from the center (distance from the nucleus or center of density)
        grid_shape = rho.shape
        x, y, z = np.indices(grid_shape)

        if self.center_of_density is not None:
            center_x = self.center_of_density[0] / grid_spacings[0]
            center_y = self.center_of_density[1] / grid_spacings[1]
            center_z = self.center_of_density[2] / grid_spacings[2]
        else:
            center_x, center_y, center_z = np.array(grid_shape) // 2

        self.grid = np.sqrt(
            ((x - center_x) * grid_spacings[0]) ** 2 +
            ((y - center_y) * grid_spacings[1]) ** 2 +
            ((z - center_z) * grid_spacings[2]) ** 2
        )

    def compute_radial_distribution(self, orbital_index: int, spin: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the radial distribution for a given orbital.

        Parameters:
            orbital_index (int): The index of the orbital to compute.
            spin (int, optional): Spin channel (0 for alpha, 1 for beta). Default is 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - r_bins (ndarray): Radial distance bins.
                - radial_distribution (ndarray): Radial distribution values.
        """
        orbital = self.get_orbital(orbital_index, spin=spin)
        rho_i = orbital.conj() * orbital

        if self.prenormalize:
            rho_i = normalize_density(rho_i, self.dv)

        r_max = self.grid.max()
        r_bins = np.linspace(0, r_max, 100)
        radial_distribution = np.zeros_like(r_bins)

        for i in range(len(r_bins) - 1):
            shell_mask = (self.grid >= r_bins[i]) & (self.grid < r_bins[i + 1])
            radial_distribution[i] = np.sum(
                rho_i[shell_mask]) * r_bins[i] ** 2 * self.dv

        return r_bins, radial_distribution

    def calculate_moment(self, r_bins: np.ndarray, radial_distribution: np.ndarray, moment_order: int) -> float:
        """
        Calculate the nth moment of the radial distribution.

        Parameters:
            r_bins (ndarray): Radial distance bins.
            radial_distribution (ndarray): Radial distribution values.
            moment_order (int): The order of the moment to calculate.

        Returns:
            float: The nth moment of the distribution.
        """
        moment = np.sum(r_bins ** moment_order * radial_distribution)
        return moment

    def shift_center_of_density(self) -> None:
        """
        Shift the grid to the center of density if it was not set initially.
        """
        if not self.use_center_of_density:
            self.center_of_density = calculate_density_center_gpw(
                calc=self.calc)
            print(f"Center of density shifted to: {self.center_of_density}")
            self.setup_grid()

    def plot_radial_distribution(self, r_bins: np.ndarray, radial_distribution: np.ndarray, orbital_index: int) -> None:
        """
        Plot the radial distribution for the given orbital.

        Parameters:
            r_bins (ndarray): Radial distance bins.
            radial_distribution (ndarray): Radial distribution values.
            orbital_index (int): Orbital index for labeling the plot.
        """
        plt.figure()
        plt.plot(r_bins, radial_distribution, label=f'Orbital {orbital_index}')
        plt.xlabel('Radial Distance (Å)')
        plt.ylabel('Radial Distribution D(r)')
        plt.title(f'Radial Distribution for Orbital {orbital_index}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_radial_distribution(self, r_bins: np.ndarray, radial_distribution: np.ndarray, orbital_index: int, output_dir: Path) -> None:
        """
        Save the radial distribution data to a text file.

        Parameters:
            r_bins (ndarray): Radial distance bins.
            radial_distribution (ndarray): Radial distribution values.
            orbital_index (int): Orbital index for labeling.
            output_dir (Path): Directory to save the file.
        """
        output_dir.mkdir(exist_ok=True)
        np.savetxt(output_dir / f'radial_distribution_orbital_{orbital_index}.txt',
                   np.column_stack((r_bins, radial_distribution)))
        print(
            f'Radial distribution for orbital {orbital_index} saved to {output_dir}')

    def compute_spatial_extent(self, orbital_index: int, spin: int = 0) -> float:
        """
        Compute the spatial extent ⟨r⟩ for a given orbital.

        Parameters:
            orbital_index (int): The index of the orbital to compute.
            spin (int, optional): Spin channel (0 for alpha, 1 for beta). Default is 0.

        Returns:
            float: The spatial extent ⟨r⟩ of the orbital.
        """
        orbital = self.get_orbital(orbital_index, spin=spin)
        rho_i = orbital.conj() * orbital

        if self.prenormalize:
            rho_i = normalize_density(rho_i, self.dv)

        # Calculate the spatial extent ⟨r⟩
        spatial_extent = np.sum(rho_i * self.grid) * self.dv
        return spatial_extent

    def save_spatial_extent(self, spatial_extents: Dict[int, float], output_dir: Path) -> None:
        """
        Save the spatial extent data to a text file.

        Parameters:
            spatial_extents (dict): Dictionary of spatial extents for each orbital.
            output_dir (Path): Directory to save the file.
        """
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'spatial_extent.txt', 'w') as f_out:
            for orbital_index, spatial_extent in spatial_extents.items():
                f_out.write(
                    f'Orbital {orbital_index}: Spatial Extent ⟨r⟩ = {spatial_extent:.6f} Bohr\n')
        print(f'Spatial extent results saved in {output_dir}')


class RadialDistributionMolecule(RadialDistribution):
    def __init__(self, molecule: Molecule, prenormalize: bool = False, use_center_of_density: bool = False):
        """
        Initialize the RadialDistributionMolecule object.

        Parameters:
            molecule (Molecule): Molecule instance with orbitals as ScalarField instances.
            prenormalize (bool, optional): Whether to prenormalize the orbital densities. Default is False.
            use_center_of_density (bool, optional): Whether to center on the center of density. Default is False.
        """
        self.molecule = molecule
        self.prenormalize = prenormalize
        self.use_center_of_density = use_center_of_density
        self.center_of_density = None
        reference_field = self.molecule.scalar_fields["All electron density"]
        self.points = reference_field.points

        if self.use_center_of_density:
            self.center_of_density = self.calculate_center_of_density()
            print(f"Center of density: {self.center_of_density}")

        self.dv = reference_field.volume_element

        self.setup_grid()

    def calculate_center_of_density(self) -> np.ndarray:
        """
        Calculate the center of density for the All electron density field.

        Returns:
            np.ndarray: Center of density in the same units as the molecule coordinates.
        """
        density = self.get_electron_density()

        coords = self.points.reshape(-1, 3)
        density_flat = density.ravel()

        return np.sum(coords * density_flat[:, None], axis=0) / np.sum(density_flat)

    def get_electron_density(self) -> np.ndarray:
        """
        Get the electron density field for the calculation.

        Returns:
            np.ndarray: Electron density field.
        """
        return self.molecule.scalar_fields["All electron density"].scalar_field

    def get_grid_spacings(self) -> List[float]:
        """
        Get the grid spacings for the calculation.

        Returns:
            List[float]: Grid spacings in x, y, and z directions.
        """
        return self.dv

    def get_orbital(self, orbital_index: int, spin: int = 0) -> np.ndarray:
        """
        Get the orbital wave function for a given index.

        Parameters:
            orbital_index (int): Index of the orbital to retrieve.
            spin (int, optional): Spin channel (0 for alpha, 1 for beta). Default is 0.

        Returns:
            np.ndarray: Orbital wave function.
        """
        return self.molecule.scalar_fields[f'Orbital {orbital_index}'].scalar_field


class RadialDistributionGpw(RadialDistribution):
    def __init__(self, gpw_file: Optional[str] = None, atoms: Optional[Atoms] = None, calc: Optional[Any] = None, prenormalize: bool = False, use_center_of_density: bool = False):
        """
        Initialize the RadialDistribution object.

        Parameters:
            gpw_file (str, optional): Path to the GPAW gpw file. Default is None.
            atoms (Atoms, optional): ASE atoms object. Required if gpw_file is None. Default is None.
            calc (GPAW calculator, optional): GPAW calculator object. Required if gpw_file is None. Default is None.
            prenormalize (bool, optional): Whether to prenormalize the orbital densities. Default is False.
            use_center_of_density (bool, optional): Whether to center on the center of density. Default is False.
        """

        self.prenormalize = prenormalize
        self.use_center_of_density = use_center_of_density
        if gpw_file is not None:
            self.atoms, self.calc = restart(gpw_file, txt=None)
        else:
            assert atoms is not None and calc is not None, "Either gpw_file or atoms and calc must be provided."
            self.atoms, self.calc = atoms, calc

        self.grid, self.dv = None, None
        self.center_of_density = None

        if self.use_center_of_density:
            self.center_of_density = calculate_density_center_gpw(
                calc=self.calc)
            print(f"Center of density: {self.center_of_density}")

        self.setup_grid()

    def get_electron_density(self) -> np.ndarray:
        """
        Get the electron density field for the calculation.

        Returns:
            np.ndarray: Electron density field.
        """
        return self.calc.get_all_electron_density(gridrefinement=1)

    def get_grid_spacings(self) -> List[float]:
        """
        Get the grid spacings for the calculation.

        Returns:
            List[float]: Grid spacings in x, y, and z directions.
        """
        return [spacing * Bohr for spacing in self.calc.wfs.gd.get_grid_spacings()]

    def get_orbital(self, orbital_index: int, spin: int = 0) -> np.ndarray:
        """
        Get the orbital wave function for a given index.

        Parameters:
            orbital_index (int): Index of the orbital to retrieve.
            spin (int, optional): Spin channel (0 for alpha, 1 for beta). Default is 0.

        Returns:
            np.ndarray: Orbital wave function.
        """
        return self.calc.get_pseudo_wave_function(band=orbital_index, spin=spin)


# Example usage:
if __name__ == '__main__':
    gpw_file = 'molecules/CO2/orbitals/gpaw.gpw'
    orbital_index = 7
    prenormalize = True
    use_center_of_density = True

    # radial_dist = RadialDistributionGpw(
    #     gpw_file, prenormalize=prenormalize, use_center_of_density=use_center_of_density)

    # molecule = Molecule.load_from_gpaw(gpw_file)
    molecule = load_full_molecule('CO2', mode='GPAW')
    radial_dist = RadialDistributionMolecule(
        molecule, prenormalize=prenormalize, use_center_of_density=use_center_of_density)

    output_dir_radial = Path('radial_distributions')
    output_dir_extent = Path('spatial_extent')

    # Compute and save radial distribution and spatial extent for each orbital
    spatial_extents = {}
    r_bins, radial_distribution = radial_dist.compute_radial_distribution(
        orbital_index)
    radial_dist.plot_radial_distribution(
        r_bins, radial_distribution, orbital_index)
    radial_dist.save_radial_distribution(
        r_bins, radial_distribution, orbital_index, output_dir_radial)

    moment_order = 3  # Example: 2nd moment
    moment = radial_dist.calculate_moment(
        r_bins, radial_distribution, moment_order)
    print(f'{moment_order}th moment for orbital {orbital_index}: {moment/(4*np.pi):.6f}')

    # Calculate and save spatial extent
    spatial_extent = radial_dist.compute_spatial_extent(orbital_index)
    spatial_extents[orbital_index] = spatial_extent
    print(
        f'Spatial extent ⟨r⟩ for orbital {orbital_index}: {spatial_extent:.6f} Å')
