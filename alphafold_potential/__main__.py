
from model import G
from potential import Potential
from supplementary import read_pickle
from scipy.optimize import minimize
import numpy as np

if __name__ == "__main__":
    p1 = "./test_pdbs_results_40_50/contacts_pdb1h1j_2020_07_22_12_10_37/pasted/1h1j.pickle"
    p2 = "./test_pdbs_results_40_50/contacts_pdb1h1j_2020_07_22_12_10_37/background_distogram/ensemble/1h1j.pickle"
    p3 = "./test_pdbs_results_40_50/contacts_pdb1h1j_2020_07_22_12_10_37/torsion/ensemble/1h1j.pickle"
    p4 = "./test_pdbs_results_40_50/contacts_pdb1h1j_2020_07_22_12_10_37/torsion/0/torsions/1h1j.torsions"

    pot1 = Potential(read_pickle(p4)["sequence"], p1, p2, p4, init_angles_by_dists=False)
    print("pot value at init:", pot1.calc(pot1.get_initial_angles()))
    min_res = minimize(pot1.calc,
                       x0=pot1.get_initial_angles(),
                       bounds=[[-np.pi, np.pi]] * 44 * 2,
                       jac=pot1.derivative,
                       method="L-BFGS-B",
                       )
    print("pot value after minimization:", pot1.calc(min_res.x))
