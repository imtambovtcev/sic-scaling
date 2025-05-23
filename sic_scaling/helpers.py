
import numpy as np

from .calculate_scaling_integrals import calculate_scaling_integrals
from .radial_distribution import RadialDistributionGpw


def get_coefficints_and_extents(atoms, calc, a=0.2, density_type="sum_rho_i", occupation_method="keep", prenormalize=True):
    orbital_ids_dict = {}
    coefficients_dict = {}
    spatial_extent_dict = {}
    radial_dist = RadialDistributionGpw(
        atoms=atoms, calc=calc, prenormalize=True, use_center_of_density=True
    )
    for k in [0, 1]:
        f_sn = calc.get_occupation_numbers(spin=k)
        orbital_ids = np.array(np.where(f_sn)[0], dtype=int)

        # Calculate scaling integrals
        integral_values, coefficient_values = calculate_scaling_integrals(
            orbital_ids,
            atoms=atoms,
            calc=calc,
            density_type=density_type,
            occupation_method=occupation_method,
            a=a,
            prenormalize=prenormalize,
            spin=k,
            uks=True,
        )
        spatial_extent_dict[k] = [
            radial_dist.compute_spatial_extent(orbital_index=orbital_index, spin=k)
            for orbital_index in orbital_ids
        ]
        orbital_ids_dict[k] = list(orbital_ids)
        coefficients_dict[k] = list(coefficient_values.values())
    return orbital_ids_dict, coefficients_dict, spatial_extent_dict
