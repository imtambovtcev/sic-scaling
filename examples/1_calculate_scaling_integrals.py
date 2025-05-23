
import os

import numpy as np
from gpaw import GPAW, restart
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.tools import excite
from gpaw.mom import prepare_mom_calculation
from sic_scaling.calculate_scaling_integrals import calculate_scaling_integrals
from sic_scaling.radial_distribution import RadialDistributionGpw

# Restart GPAW calculation
atoms, calc = restart("ground_state/h2o.gpw", txt=None)

# Initialize wave functions from restart file
calc.wfs.initialize_wave_functions_from_restart_file()

e = atoms.get_potential_energy()

print(f"Ground state energy: {e} eV")
radial_dist = RadialDistributionGpw(
        atoms=atoms, calc=calc, prenormalize=True, use_center_of_density=True
    )

k=0 # Spin index, 0 for alpha, 1 for beta

f_sn = calc.get_occupation_numbers(spin=k)
orbital_ids = np.array(np.where(f_sn)[0], dtype=int)

integral_values, coefficient_values = calculate_scaling_integrals(
            orbital_ids,
            atoms=atoms,
            calc=calc,
            density_type="sum_rho_i",
            occupation_method="keep",
            a=0.2,
            prenormalize=True,
            spin=k,
            uks=True,
        )

print(f"Orbital IDs: {orbital_ids}")
print(f"Coefficients: {coefficient_values}")
print(f"Radial distribution: {radial_dist.compute_spatial_extent(orbital_index=orbital_ids[0], spin=k)}")
print(f"Integral values: {integral_values}")