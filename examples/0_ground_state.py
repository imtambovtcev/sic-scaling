import os

import numpy as np
from ase import Atoms
from ase.build import molecule
from gpaw import FD, GPAW
from gpaw.directmin.etdm_fdpw import FDPWETDM

# Create directory for output
output_dir = 'ground_state'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up H2O molecule using ASE's built-in molecule function
water = molecule('H2O')
water.center(vacuum=3.0)  # Add some vacuum around the molecule

mode = FD(force_complex_dtype=True)

calc = GPAW(mode='fd',
            h=0.2,
            xc='PBE',
            maxiter=1000,
            spinpol=True,
            eigensolver='cg',
            occupations={'name': 'fixed-uniform'},
            nbands=-10,
            symmetry='off',
            txt=f'{output_dir}/h2o.txt')


water.calc = calc
water.get_potential_energy()

calc.set(eigensolver=FDPWETDM(),
         mixer={'backend': 'no-mixing'})

# Run the calculation
energy = water.get_potential_energy()
print(f'Potential energy: {energy:.3f} eV')

# Save the calculation state
calc.write(f'{output_dir}/h2o.gpw', mode='all')