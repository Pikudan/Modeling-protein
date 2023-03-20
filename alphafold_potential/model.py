
import numpy as np
from .basic_geometry import *
from .protein_geometry import Geometry

class G:
    _geom = None
    _sequence = None
    def __init__(self, sequence):
        self._geom = Geometry()
        self._sequence = sequence

    def _place_next_dihedral_atom(self, prev_atoms, dih_ang_val, planar_ang_val, dist):
        A1 = prev_atoms[-2] + get_axis_rotation_matrix(prev_atoms[-1] - prev_atoms[-2],
                                                       dih_ang_val).dot(prev_atoms[-3] - prev_atoms[-2])
        v0 = (prev_atoms[-1] - prev_atoms[-2]) / np.linalg.norm(prev_atoms[-1] - prev_atoms[-2])
        E = project_point_on_line(A1,
                                  (prev_atoms[-1],
                                   prev_atoms[-1] - prev_atoms[-2]))
        v1 = (A1 - E) / np.linalg.norm(A1 - E)
        return prev_atoms[-1] - v0 * dist * np.cos(planar_ang_val) + v1 * dist * np.sin(planar_ang_val)

    def get_cb_dist_matrix(self, phi_psi_vals):
        cb_coords, _ = self.get_cb_coords(phi_psi_vals)
        M = np.zeros((len(cb_coords), len(cb_coords)))
        for i in range(len(cb_coords) - 1):
            for j in range(i, len(cb_coords)):
                M[i ,j] = M[j ,i] = np.linalg.norm(cb_coords[i] - cb_coords[j])
        return M

    def get_cb_coords(self, phi_psi_values):
        at_coords = np.zeros((len(self._sequence) ,3 ,3))
        cb_coords = np.zeros((len(self._sequence) ,3))

        n_coord = np.array([0 ,0 ,0])
        ca_coord = np.array([self._geom.N_CA ,0 ,0])
        c_coord = ca_coord + self._geom.CA_C * np.array([-np.cos(self._geom.N_CA_C),
                                                         np.sin(self._geom.N_CA_C),
                                                         0])
        at_coords[0] = [n_coord, ca_coord, c_coord]

        # WARN: CB cannot be placed for first residue, so set CA!
        # TODO: maybe can be placed somehow?
        cb_coords[0] = ca_coord


        for i in range(1, len(self._sequence)):
            prev_psi = phi_psi_values[ 2 *( i -1)]
            cur_phi = phi_psi_values[ 2 *( i -1 ) +1]
            n_coord = self._place_next_dihedral_atom(at_coords[ i -1],
                                                     prev_psi,
                                                     self._geom.CA_C_N,  # (prev_psi),
                                                     self._geom.C_N)
            ca_coord = self._place_next_dihedral_atom([at_coords[ i -1][1], at_coords[ i -1][2], n_coord],
                                                      np.pi,
                                                      self._geom.C_N_CA,
                                                      self._geom.N_CA)
            c_coord = self._place_next_dihedral_atom([at_coords[ i -1][2], n_coord, ca_coord],
                                                     cur_phi,
                                                     self._geom.N_CA_C,
                                                     self._geom.CA_C)
            at_coords[i] = [n_coord, ca_coord, c_coord]
            if self._sequence[i] != "G":
                cb_coord = self._place_next_dihedral_atom([at_coords[ i -1][2], n_coord, ca_coord],
                                                          self._geom.C_N_CA_CB(cur_phi),
                                                          self._geom.N_CA_CB,
                                                          self._geom.CA_CB)
                cb_coords[i] = cb_coord
            else:
                cb_coords[i] = ca_coord

        return cb_coords, at_coords