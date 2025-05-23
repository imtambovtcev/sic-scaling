
import json
import os

import numpy as np
from gpaw import GPAW, restart, setup_paths
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.tools import excite
from gpaw.mom import prepare_mom_calculation

output_dir = 'excited_state'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

atoms, calc = restart("ground_state/h2o.gpw", txt=None)
calc.wfs.initialize_wave_functions_from_restart_file()
f_sn = excite(calc, 0, 0, (0, 0)) #first excited state 0, 1, (0, 0) - second


calc.set(eigensolver=FDPWETDM(excited_state=True), txt = f'{output_dir}/excited.txt')
prepare_mom_calculation(calc, atoms, f_sn)
e = atoms.get_potential_energy()
calc.write(f'{output_dir}/excited.gpw', mode = 'all')

print(f'Excited state energy: {e} eV')

