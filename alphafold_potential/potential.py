import numpy as np
from .model import G
from Bio.PDB import calc_angle, Vector
from .supplementary import *

class Potential:
    _model = None
    _sequence = None
    _init_angles = None

    def __init__(self,
                 sequence,
                 pasted_pickle_path,
                 background_pickle_path,
                 torsion_pickle_path,
                 init_angles_by_dists=True):
        self._model = G(sequence)
        self._sequence = sequence
        self.pasted_distributions = convert_discrete_potential_to_spline(read_pickle(pasted_pickle_path)["probs"])
        self.background_distributions = convert_discrete_potential_to_spline(
            read_pickle(background_pickle_path)["probs"])
        self.torsion_distributions = read_pickle(torsion_pickle_path)["probs"].reshape((len(sequence), 36, 36))

        self.pasted_distributions = self.pasted_distributions[np.triu_indices_from(self.pasted_distributions[:, :, 0],
                                                                                   1)]
        self.background_distributions = self.background_distributions[
            np.triu_indices_from(self.background_distributions[:, :, 0],
                                 1)]
        # TODO: check for error in next line?
        self.torsion_distributions = [interp2d(np.arange(-np.pi + np.pi / 36, np.pi, np.pi / 18),
                                               np.arange(-np.pi + np.pi / 36, np.pi, np.pi / 18),
                                               distrib, kind='cubic')
                                      for distrib in self.torsion_distributions]

        if init_angles_by_dists:
            self._init_angles = self.get_initial_angles_by_dists(read_pickle(pasted_pickle_path)["probs"])
        else:
            self._init_angles = self.get_initial_angles_by_torsion(
                read_pickle(torsion_pickle_path)["probs"].reshape((len(sequence), 36, 36)))

    def get_initial_angles_by_dists(self, distogram):
        angles = np.array([np.pi, np.pi])
        print("ANGLE INIT STARTED")
        for i in range(2, len(self._sequence)):
            print("STEP {}/{} STARTED".format(i, len(self._sequence)))
            angles = np.hstack((angles, [np.pi, np.pi]))
            md = G(self._sequence[:i])
            loop_step_distogram = distogram[:i, :i, :]
            def f(angles):
                matr = md.get_cb_dist_matrix(angles[1:-1])
                u_tri_inds = np.triu_indices_from(matr, 1)
                nonlocal loop_step_distogram
                # print(loc_distogram)
                loc_distogram = loop_step_distogram[u_tri_inds]
                matr = matr[u_tri_inds]
                loc_distogram = np.argmax(loc_distogram, axis=1) * DIST_STEP
                return np.sum((loc_distogram - matr) ** 2)

            angles = minimize(f, x0=angles, bounds=[[-np.pi, np.pi]] * len(angles)).x
            print("\tangles:", angles)
        return angles

    def get_initial_angles_by_torsion(self, torsion):
        angles = np.array([])
        grid_step = 10 / 360 * (2 * np.pi)
        for i in range(len(torsion)):
            argmax = np.argmax(torsion[i])
            angles = np.hstack((angles, [(argmax // 36) * grid_step - np.pi,
                                         (argmax % 36) * grid_step - np.pi]))
        return angles

    def _calc_distance_potential(self, phi_psi_vals):
        dist_matr = self._model.get_cb_dist_matrix(phi_psi_vals[1:-1])
        dist_matr = dist_matr[np.triu_indices_from(dist_matr, 1)]
        # print(dist_matr)
        pot1 = np.array(
            [self.pasted_distributions[i, 1] if dist_matr[i] > 18 else self.pasted_distributions[i, 0](dist_matr[i])
             for i in range(len(dist_matr))])
        pot2 = np.array([self.background_distributions[i, 1] if dist_matr[i] > 18 else self.background_distributions[
            i, 0](dist_matr[i])
                         for i in range(len(dist_matr))])
        pot1[np.isnan(pot1)] = 1
        pot2[np.isnan(pot2)] = 1

        pot1[pot1 <= 0] = 1
        pot2[pot2 <= 0] = 1

        return -np.sum(np.log(pot1) - np.log(pot2))

    def _calc_torsion_potential(self, phi_psi_vals):
        #         grid_step = 10/360*(2*np.pi)
        #         discr_angles = ((phi_psi_vals + np.pi) // grid_step).astype(int)
        #         discr_angles[discr_angles >= 36] = 35
        #         discr_angles = discr_angles.reshape((-1,2))

        #         pot = np.array([self.torsion_distributions[i, p[0], p[1]] for i,p in enumerate(discr_angles)])
        #         return -np.sum(np.log(pot))
        pot = np.array([self.torsion_distributions[i](phi_psi_vals[2 * i], phi_psi_vals[2 * i + 1])
                        for i in range(len(phi_psi_vals) // 2)])
        pot[pot <= 0] = 1 # so that ans would be zero
        return -np.sum(np.log(pot))

    def calc(self, phi_psi_vals):
        return self._calc_distance_potential(phi_psi_vals) + self._calc_torsion_potential(phi_psi_vals)

    def get_initial_angles(self):
        return self._init_angles

    def _ARM(self, axis, last_dihedral_data, last_dihedral_ind, der_axis=None):
        # print("ENTERED ARM")
        last_dihedral, last_dih_is_const = last_dihedral_data
        appendices = None
        x, y, z = axis
        # print("ARM axis shape: {}; els: {}, {}, {}".format(axis.shape, x,y,z))
        if der_axis is not None:
            trig1, trig2 = -np.sin(last_dihedral), np.cos(last_dihedral)
        else:
            trig1, trig2 = np.cos(last_dihedral), np.sin(last_dihedral)
        last_der = (1 - trig1) * np.array([[x ** 2, x * y, x * z],
                                           [y * x, y ** 2, y * z],
                                           [z * x, z * y, z ** 2]]) + np.array([[trig1, -z * trig2, y * trig2],
                                                                                [z * trig2, trig1, -x * trig2],
                                                                                [-y * trig2, x * trig2, trig1]])
        # print("ARM: COUNTED INIT DER OF SHAPE {}".format(last_der.shape))
        if der_axis is None:
            return last_der

        ans = np.zeros((der_axis.shape[1], 3, 3))

        if last_dih_is_const:
            last_der = np.zeros((3, 3))

        if last_dihedral_ind is not None:
            ans[last_dihedral_ind] = last_der

        # ans[-1] += last_der

        # print("ARM: der axis SHAPE: {}".format(der_axis.shape))
        dx, dy, dz = der_axis
        # print("ARM d: {}, {}, {}".format(dx, dy, dz))
        trig1, trig2 = np.cos(last_dihedral), np.sin(last_dihedral)
        zero_el = np.zeros(dx.shape)
        last_der = (1 - trig1) * np.array([[2 * x * dx, dx * y + x * dy, dx * z + x * dz],
                                           [dy * x + y * dx, 2 * y * dy, dy * z + y * dz],
                                           [dz * x + z * dx, dz * y + z * dy, 2 * z * dz]]) + np.array(
            [[zero_el, -dz * trig2, dy * trig2],
             [dz * trig2, zero_el, -dx * trig2],
             [-dy * trig2, dx * trig2, zero_el]])
        # print("ARM: COUNTED FULL DER OF SHAPE {}".format(last_der.shape))
        last_der = np.moveaxis(last_der, [0, 1, 2], [1, 2, 0])
        ans = ans + last_der
        # print("ARM: RETURN SHAPE: {}".format(ans.shape))
        return ans

    def _dp_dx(self, prev_ders, coords, dihedral, last_dih_num, is_peptide_bond=False):
        A, B, C, D = coords

        planar = calc_angle(Vector(D),
                            Vector(C),
                            Vector(B))
        dist = np.linalg.norm(D - C)
        der = np.zeros((3, prev_ders.shape[2]))
        der += prev_ders[-1]

        # dv0/dx
        dv0_dx = 1 / np.linalg.norm(C - B) * (prev_ders[-1] - prev_ders[-2])
        der -= dist * np.cos(planar) * dv0_dx

        # dv1/dx
        arm = self._ARM(C - B, (dihedral, is_peptide_bond), last_dih_num)
        darm_dx = self._ARM(C - B, (dihedral, is_peptide_bond), last_dih_num,
                            der_axis=prev_ders[-1] - prev_ders[-2])
        A1 = B + arm.dot(A - B)
        dA1_dx = prev_ders[-2] + (darm_dx.dot(A - B)).T + arm.dot(prev_ders[-3] - prev_ders[-2])

        E = C - (A1 - C).dot(C - B) / (np.linalg.norm(C - B) ** 2) * (C - B)

        dE_dx = (np.full((2*len(self._sequence), 3),
                         (dA1_dx - prev_ders[-1]).T.dot(C - B).reshape((der.shape[1],
                                                                        1))) * (C - B)).T
        dE_dx += (np.full((2*len(self._sequence), 3),
                          (prev_ders[-1] - prev_ders[-2]).T.dot(A1 - C).reshape((der.shape[1],
                                                                                 1))) * (C - B)).T
        dE_dx += (A1 - C).dot(C - B) * (prev_ders[-1] - prev_ders[-2])
        dE_dx = prev_ders[-1] - 1 / (np.linalg.norm(C - B) ** 2) * dE_dx

        len_EA1 = np.linalg.norm(E - A1)
        dv1_dx = 1 / (len_EA1 ** 2) * (1 / len_EA1 - (A1 - E) ** 2 / (len_EA1 ** 3)) * (dA1_dx - dE_dx).T
        dv1_dx = dv1_dx.T
        der += dist * np.sin(planar) * dv1_dx

        return der

    def derivative(self, phi_psi_vals):
        der = np.zeros(len(phi_psi_vals))
        cb_coords, bb_coords = self._model.get_cb_coords(phi_psi_vals[1:-1])
        bb_coords = np.reshape(bb_coords, (-1, 3))
        dists = self._model.get_cb_dist_matrix(phi_psi_vals[1:-1])
        inds = np.vstack(np.triu_indices_from(dists, 1)).T
        dists = dists[np.triu_indices_from(dists, 1)]

        bb_ders = np.zeros((3 * len(phi_psi_vals) // 2, 3, len(phi_psi_vals)))
        cb_ders = np.zeros((len(phi_psi_vals) // 2, 3, len(phi_psi_vals)))

        for i in range(3, len(bb_ders)):
            # print(i, "started")
            is_peptide = False
            last_angle_num = None
            cur_atom = ""
            if i % 3 == 1:
                # peptide bond for CA, ang = np.pi
                dih = np.pi
                is_peptide = True
                cur_atom = "CA"
            elif i % 3 == 0:
                # N
                dih = phi_psi_vals[2 * (i // 3 - 1) + 1]
                last_angle_num = 2 * (i // 3 - 1) + 1
                cur_atom = "N"
            else:
                # C
                dih = phi_psi_vals[2 * (i // 3)]
                last_angle_num = 2 * (i // 3)
                cur_atom = "C"

            d_der = self._dp_dx(bb_ders[i - 3:i],
                                bb_coords[i - 3:i + 1],
                                dih,
                                last_angle_num,
                                is_peptide_bond=is_peptide)
            bb_ders[i] = d_der

            # cb:
            if cur_atom == "CA" and self._sequence[i // 3] == "G":
                # if cur residue is GLY, set CB derivative equal to CA derivative
                cb_ders[i // 3] = d_der
                # print("\tres {} is GLY. Setting derivative of CA to CB".format(i//3))
            elif cur_atom == "C" and self._sequence[i // 3] != "G":
                # CB for non-GLY residues is computed by the same derivatives.
                # Atoms: 3 same + CB_(i//3)
                # Only dihedral is different.
                d_der = self._dp_dx(bb_ders[i - 3:i],
                                    np.vstack((bb_coords[i - 3:i],
                                               [cb_coords[i // 3]])),
                                    self._model._geom.C_N_CA_CB(dih),
                                    last_angle_num,
                                    is_peptide_bond=False)
                cb_ders[i // 3] = d_der
                # print("\tres {} CB derivative set".format(i//3))

        # aggregating data
        init_val = np.array([self.pasted_distributions[pot_index, 0].derivative()(dists[pot_index]) / (
            self.pasted_distributions[pot_index, 1] if dists[pot_index] > 18 else self.pasted_distributions[
                pot_index, 0](dists[pot_index])) for pot_index, v in enumerate(inds)])
        init_val -= np.array([self.background_distributions[pot_index, 0].derivative()(dists[pot_index]) / (
            self.background_distributions[pot_index, 1] if dists[pot_index] > 18 else self.background_distributions[
                pot_index, 0](dists[pot_index])) for pot_index, v in enumerate(inds)])
        init_val *= -1

        # print(cb_ders)

        dists_dx = np.array([((cb_ders[v[1]] -
                               cb_ders[v[0]]).T * (cb_coords[v[1]] -
                                                   cb_coords[v[0]]) / dists[i]).sum(axis=1)
                             for i, v in enumerate(inds)])
        dists_dx = (dists_dx.T * init_val).T

        # PART 2. torsion derivative
        torsion_pot = np.array([self.torsion_distributions[i](phi_psi_vals[2 * i],
                                                              phi_psi_vals[2 * i + 1])
                                for i in range(len(phi_psi_vals) // 2)])
        torsion_pot = np.vstack((torsion_pot, torsion_pot)).T.reshape(-1)

        torsion_pot_ders = np.zeros(len(phi_psi_vals))
        x = np.arange(-np.pi + np.pi / 36, np.pi, np.pi / 18)
        for i in range(len(self._sequence)):
            # phi. psi should be fixed!
            y = self.torsion_distributions[i](x, phi_psi_vals[2 * i + 1])
            torsion_pot_ders[2 * i] = CubicSpline(x, y).derivative()(phi_psi_vals[2 * i])
            # psi. Phi should be fixed!
            y = self.torsion_distributions[i](phi_psi_vals[2 * i], x)
            torsion_pot_ders[2 * i + 1] = CubicSpline(x, y).derivative()(phi_psi_vals[2 * i + 1])

        torsion_pot_ders = -1 * torsion_pot_ders / torsion_pot

        # print(dists_dx)

        return dists_dx.sum(axis=0) + torsion_pot_ders

