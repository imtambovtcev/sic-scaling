import os

import numpy as np
from gpaw import GPAW, restart
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.tools import excite
from gpaw.mom import prepare_mom_calculation
from sic_scaling.helpers import get_coefficints_and_extents

# Create directory for output
output_dir = 'excited_state_sic'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Restart GPAW calculation
atoms, calc = restart("excited_state/excited.gpw", txt=None)

# Initialize wave functions from restart file
calc.wfs.initialize_wave_functions_from_restart_file()

e = atoms.get_potential_energy()

print(f"Ground state energy: {e} eV")

orbital_ids_before, coefficients_before, radial_dist_before = get_coefficints_and_extents(atoms=atoms, calc=calc, a=0.2, density_type="sum_rho_i", occupation_method="keep", prenormalize=True)

scales = [np.array(coefficients_before[0], dtype=float) , np.array(coefficients_before[1], dtype=float) ]

print(f"Orbital IDs before: {orbital_ids_before}")
print(f"Coefficients before: {coefficients_before}")
print(f"Radial distribution before: {radial_dist_before}")

# Set scaling factor for the calculation
calc.set(
    eigensolver=FDPWETDM(
        excited_state=True, functional={"name": "PZ-SIC", "scaling_factor": (scales, scales)}
    ),
    txt=f"{output_dir}/excited.txt",
)

f_sn = [calc.get_occupation_numbers(spin=k) for k in [0, 1]]
# Prepare MOM calculation
prepare_mom_calculation(calc, atoms, f_sn)

# Get potential energy and save calculation
e = atoms.get_potential_energy()
calc.write(f"{output_dir}/excited_sic.gpw", mode="all")

print(f"Ground state energy SIC: {e} eV")

orbital_ids_after, coefficients_after, radial_dist_after = get_coefficints_and_extents(atoms=atoms, calc=calc, a=0.2, density_type="sum_rho_i", occupation_method="keep", prenormalize=True)

print(f"Orbitals after: {orbital_ids_after}")
print(f"Coefficients after: {coefficients_after}")
print(f"Radial distribution after: {radial_dist_after}")

