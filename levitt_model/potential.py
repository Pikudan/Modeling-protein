import numpy as _np
import pandas as _pd
from scipy.optimize import minimize as _minimize

# from .LevittModel import LevittModel as _lvtmd

_Kss = 10
_r_0_SS = 4.2
_eps_p = 0.2
_r_p = 4.6
_q_p2 = 0.74 ** 2
_K_alpha = 2

_sigma_hold = 3


class Potential:
    _aminoacids = None
    _model = None

    _vdwCoeffs = None
    _vdwType = None
    _solventCoeffs = None
    _fourierCoeffs = None
    _dist_ns_os = None

    # variables for minimization
    _V_init = None
    _init_alphas = None
    _non_sec_struct_alpha_inds = []

    # _potentialType = None

    def __init__(self, aminoacids, geometryModel, vdwCoeffsPath, solventCoeffsPath, fourierCoeffsPath, vdwType='A',
                 dist_ns_os=1):
        self._aminoacids = _np.array(aminoacids)
        self._model = geometryModel
        self._vdwCoeffs = _pd.read_csv(vdwCoeffsPath)
        self._vdwType = vdwType
        self._solventCoeffs = _pd.read_csv(solventCoeffsPath)
        self._fourierCoeffs = _pd.read_csv(fourierCoeffsPath)
        self._dist_ns_os = dist_ns_os
        # self._potentialType = potentialType

    def _calculateVanDerWaalsAndSSEnergy(self, RcentroidCoords, verbose=False):
        if verbose:
            print('*** VAN DER WAALS CALC STARTED ***')
        result = 0
        for i in range(1, len(self._aminoacids) - 2):
            for j in range(i + 2, len(self._aminoacids) - 1):
                r_ij = _np.linalg.norm(RcentroidCoords[i - 1] - RcentroidCoords[j - 1])

                if self._aminoacids[i] == 'CYS' and self._aminoacids[j] == 'CYS':
                    result += _Kss * (r_ij - _r_0_SS) ** 2
                    continue

                iCoeefs = self._vdwCoeffs[self._vdwCoeffs['amino_acid'] == self._aminoacids[i]]
                jCoeefs = self._vdwCoeffs[self._vdwCoeffs['amino_acid'] == self._aminoacids[j]]
                eps_ij = _np.sqrt(
                    iCoeefs['eps_' + self._vdwType].astype(float).values[0] *
                    jCoeefs['eps_' + self._vdwType].astype(float).values[0]
                )
                r0_ij = _np.sqrt(
                    iCoeefs['r0_' + self._vdwType].astype(float).values[0] *
                    jCoeefs['r0_' + self._vdwType].astype(float).values[0]
                )

                result += eps_ij * (3 * (r0_ij / r_ij) ** 8 - 4 * (r0_ij / r_ij) ** 6)

        if verbose:
            print('*** VAN DER WAALS CALC OVER ***')

        return result

    def _g(self, r_ij):
        x = r_ij / 9
        return 1 - 1 / 2 * (7 * x ** 2 -
                            9 * x ** 3 +
                            5 * x ** 6 -
                            x ** 8)

    def _calculateSolventContactEnergy(self, RcentroidCoords):
        result = 0
        for i in range(0, len(self._aminoacids) - 2):
            s_i = self._solventCoeffs[self._solventCoeffs['amino_acid'] == self._aminoacids[i]]['s'].values[0]
            jInds = i + 1 + _np.where(_np.linalg.norm(RcentroidCoords[i + 1:] - RcentroidCoords[i], axis=1) < 9)[0]
            if jInds.shape[0] == 0:
                continue
            jDists = _np.linalg.norm(RcentroidCoords[jInds] - RcentroidCoords[i], axis=1)
            s_j_all = [self._solventCoeffs[self._solventCoeffs['amino_acid'] == acid]['s'].values[0]
                       for acid in self._aminoacids[jInds]]
            gs = _np.array([self._g(dist) for dist in jDists])

            result += gs.dot(s_j_all + s_i)

        return result

    def _calculateHydrogenBondEnergy(self, Nss, Oss, verbose=False):
        if verbose:
            print('*** HYDROGEN CALC STARTED ***')
        result = 0
        for i in range(0, len(Nss) - 3):
            if self._aminoacids[i + 1] == 'PRO':  # or self._aminoacids[i + 1] == 'PRO':
                continue
            for j in range(i + 3, len(Nss)):
                if self._aminoacids[j + 1] == 'PRO':  # or self._aminoacids[j + 1] == 'PRO':
                    continue
                r_NN = _np.linalg.norm(Nss[i] - Nss[j])
                r_OO = _np.linalg.norm(Oss[i] - Oss[j])
                r_NO = _np.linalg.norm(Nss[i] - Oss[j])
                r_ON = _np.linalg.norm(Oss[i] - Nss[j])

                if verbose:
                    print((i, j), 'DISTANCES (' + str(_r_p) + '):\n\t', r_NN, r_OO, r_NO, r_ON)

                result += _eps_p * (
                        (_r_p / r_NO) ** 12 -
                        2 * (_r_p / r_NO) ** 6 +
                        (_r_p / r_ON) ** 12 -
                        2 * (_r_p / r_ON) ** 6)
                result += 332 * _q_p2 * (1 / r_NN + 1 / r_NO - 1 / r_OO - 1 / r_ON)

                if verbose:
                    print('\tpart1 =', _eps_p * (
                            (_r_p / r_NO) ** 12 - 2 * (_r_p / r_NO) ** 6 + (_r_p / r_ON) ** 12 - 2 * (
                            _r_p / r_ON) ** 6))
                    print('\tpart2 =', 332 * _q_p2 * (1 / r_NN +
                                                      1 / r_OO -
                                                      1 / r_NO -
                                                      1 / r_ON))

        if verbose:
            print('*** HYDROGEN CALC OVER ***')

        return result

    def _calculateNonBondedEnergy(self, alphas):
        result = 0
        for i in range(len(alphas)):
            potentialType = 'ALA'
            if self._aminoacids[i + 2] == 'GLY':
                potentialType = 'GLY'
            elif self._aminoacids[i + 2] == 'PRO':
                potentialType = 'PRO'
            coeffs = self._fourierCoeffs[['A_' + potentialType, 'B_' + potentialType]].values
            arg = (_np.arange(1, 7) - 1) * alphas[i]
            c = _np.cos(arg)
            s = _np.sin(arg)
            result += 2 * (coeffs[:, 0].dot(c) + coeffs[:, 1].dot(s))

        return result

    def calculateEnergyPotential(self, alphas, calcVDW=True, calcSolv=True, calcHydr=True, calcNoB=True, verbose=False):
        _, Rs, Nss, Oss = _np.array(self._model.dihedralsToCoords(alphas, self._dist_ns_os))
        result = 0
        if verbose:
            print('COMPONENTS:')

        if calcVDW:
            energy = self._calculateVanDerWaalsAndSSEnergy(Rs)
            if verbose:
                print('\tvan der Waals and SS:', energy)
            result += energy
        if calcSolv:
            energy = self._calculateSolventContactEnergy(Rs)
            if verbose:
                print('\tSolvent:', energy)
            result += energy
        if calcHydr:
            energy = self._calculateHydrogenBondEnergy(Nss, Oss)
            if verbose:
                print('\tHydrogen:', energy)
            result += energy
        if calcNoB:
            energy = self._calculateNonBondedEnergy(alphas)
            if verbose:
                print('\tNon bonded:', energy)
            result += energy

        return result

    def searchPotentialMinimum(self, x0, minimizationMethod=None, verbose=False):
        minRes = _minimize(self.calculateEnergyPotential,
                           method=minimizationMethod,
                           x0=x0,
                           bounds=[[-_np.pi, _np.pi]] * len(x0))
        if verbose:
            print(minRes)
        return minRes

    """
    ##################################################
    ###                                            ###
    ###              DERIVATIVES                   ###
    ###                                            ###
    ##################################################
    """

    def _calculate_van_der_Waals_and_SS_derivative(self, RcentroidCoords, verbose=False):
        result = _np.zeros((RcentroidCoords.shape[0], 3))
        for i in range(1, len(self._aminoacids) - 2):
            for j in range(i + 2, len(self._aminoacids) - 1):
                r_ij = _np.linalg.norm(RcentroidCoords[i - 1] - RcentroidCoords[j - 1])
                # base = 0
                if self._aminoacids[i] == 'CYS' and self._aminoacids[j] == 'CYS':
                    base = 2 * _Kss * (r_ij - _r_0_SS)
                else:
                    iCoeefs = self._vdwCoeffs[self._vdwCoeffs['amino_acid'] == self._aminoacids[i]]
                    jCoeefs = self._vdwCoeffs[self._vdwCoeffs['amino_acid'] == self._aminoacids[j]]
                    eps_ij = _np.sqrt(
                        iCoeefs['eps_' + self._vdwType].astype(float).values[0] *
                        jCoeefs['eps_' + self._vdwType].astype(float).values[0]
                    )
                    r0_ij = _np.sqrt(
                        iCoeefs['r0_' + self._vdwType].astype(float).values[0] *
                        jCoeefs['r0_' + self._vdwType].astype(float).values[0]
                    )
                    base = eps_ij * 24 * ((r0_ij ** 6) / (r_ij ** 7) - (r0_ij ** 8) / (r_ij ** 9))

                result[i - 1] += base / r_ij * (RcentroidCoords[i - 1] - RcentroidCoords[j - 1])
                result[j - 1] -= base / r_ij * (RcentroidCoords[i - 1] - RcentroidCoords[j - 1])

        return result

    def _dg_drij(self, r_ij):
        x = r_ij / 9
        return - 1 / 2 * (14 * x -
                          27 * x ** 2 / 9 +
                          30 * x ** 5 / 9 -
                          8 * x ** 7 / 9)

    def _calculate_solvent_derivative(self, RcentroidCoords, verbose=False):
        result = _np.zeros((RcentroidCoords.shape[0], 3))
        for i in range(1, len(self._aminoacids) - 2):
            s_i = self._solventCoeffs[self._solventCoeffs['amino_acid'] == self._aminoacids[i]]['s'].values[0]
            jInds = i + _np.where(_np.linalg.norm(RcentroidCoords[i:] - RcentroidCoords[i - 1], axis=1) < 9)[0]
            if jInds.shape[0] == 0:
                continue
            jDists = _np.repeat([_np.linalg.norm(RcentroidCoords[jInds] - RcentroidCoords[i - 1], axis=1)], 3, axis=0).T
            s_j_all = [self._solventCoeffs[self._solventCoeffs['amino_acid'] == acid]['s'].values[0] for acid in
                       self._aminoacids[jInds + 1]]
            dgs = _np.array([self._dg_drij(dist) for dist in jDists])

            result[jInds] -= _np.repeat([s_j_all + s_i], 3, axis=0).T * dgs / jDists * (
                    -RcentroidCoords[jInds] + RcentroidCoords[i - 1]) / _np.linalg.norm(
                -RcentroidCoords[jInds] + RcentroidCoords[i - 1], axis=0)
            result[i - 1] += _np.sum(_np.repeat([s_j_all + s_i], 3, axis=0).T * dgs / jDists * (
                    -RcentroidCoords[jInds] + RcentroidCoords[i - 1]) / _np.linalg.norm(
                -RcentroidCoords[jInds] + RcentroidCoords[i - 1],
                axis=0),
                                     axis=0)
            if verbose:
                print('\tSTEP', i, '\n\t\t', result[jInds - 1], '\n\t\t', result[i - 1])

        return result

    def _calculate_hydrogen_bond_derivative(self, Nss, Oss, verbose=False):
        result_nss = _np.zeros((Nss.shape[0], 3))
        result_oss = _np.zeros((Nss.shape[0], 3))
        for i in range(0, len(Nss) - 3):
            if self._aminoacids[i + 1] == 'PRO':
                continue
            for j in range(i + 3, len(Nss)):
                if self._aminoacids[j + 1] == 'PRO':
                    continue
                r_NN = _np.linalg.norm(Nss[i] - Nss[j])
                r_OO = _np.linalg.norm(Oss[i] - Oss[j])
                r_NO = _np.linalg.norm(Nss[i] - Oss[j])
                r_ON = _np.linalg.norm(Oss[i] - Nss[j])

                result_nss[i] += _eps_p * (-12 * _r_p ** 12 / r_NO ** 13 + 12 * _r_p ** 6 / r_NO ** 7) / r_NO * (
                        Nss[i] - Oss[j])
                result_oss[j] -= _eps_p * (-12 * _r_p ** 12 / r_NO ** 13 + 12 * _r_p ** 6 / r_NO ** 7) / r_NO * (
                        Nss[i] - Oss[j])
                result_nss[j] -= _eps_p * (-12 * _r_p ** 12 / r_ON ** 13 + 12 * _r_p ** 6 / r_ON ** 7) / r_ON * (
                        Oss[i] - Nss[j])
                result_oss[i] += _eps_p * (-12 * _r_p ** 12 / r_ON ** 13 + 12 * _r_p ** 6 / r_ON ** 7) / r_ON * (
                        Oss[i] - Nss[j])

                # q = 332 * _q_p2 * (1 / r_NN + 1 / r_NO - 1 / r_OO - 1 / r_ON)

                result_nss[i] += 332 * _q_p2 * (
                        (-1 / r_NN ** 2) * (Nss[i] - Nss[j]) / r_NN + (-1 / r_NO ** 2) * (Nss[i] - Oss[j]) / r_NO)
                result_oss[i] += 332 * _q_p2 * (
                        (-1 / r_OO ** 2) * (Oss[i] - Oss[j]) / r_OO + (-1 / r_ON ** 2) * (Oss[i] - Nss[j]) / r_ON)
                result_nss[j] -= 332 * _q_p2 * (
                        (-1 / r_NN ** 2) * (Nss[i] - Nss[j]) / r_NN + (-1 / r_ON ** 2) * (Oss[i] - Nss[j]) / r_ON)
                result_oss[j] -= 332 * _q_p2 * (
                        (1 / r_NO ** 2) * (Nss[i] - Oss[j]) / r_NO + (-1 / r_OO ** 2) * (Oss[i] - Oss[j]) / r_OO)

        return result_nss, result_oss

    def _calculate_nonbonded_derivative(self, alphas):
        result = _np.zeros(len(alphas))
        for i in range(len(alphas)):
            potentialType = 'ALA'
            if self._aminoacids[i + 2] == 'GLY':
                potentialType = 'GLY'
            elif self._aminoacids[i + 2] == 'PRO':
                potentialType = 'PRO'
            coeffs = self._fourierCoeffs[['A_' + potentialType, 'B_' + potentialType]].values
            arg = (_np.arange(1, 7) - 1) * alphas[i]
            c = _np.cos(arg)
            s = _np.sin(arg)
            result[i] = 2 * (coeffs[:, 0].dot(c) + coeffs[:, 1].dot(s))

        return result

    def calculate_potential_alpha_derivative(self, alphas, verbose=False):
        sec_struct_mode = False
        if len(alphas) < len(self._init_alphas):
            sec_struct_mode = True
            tmp_alphas = _np.array(self._init_alphas)
            tmp_alphas[self._non_sec_struct_alpha_inds] = alphas
            alphas = _np.array(tmp_alphas)

        Cas, Rs, Nss, Oss = _np.array(self._model.dihedralsToCoords(alphas, self._dist_ns_os))

        if verbose:
            print('Derivative calc started')
        dV_dalpha = _np.empty(0)

        dV_Rs = self._calculate_van_der_Waals_and_SS_derivative(Rs, verbose)
        if verbose:
            print('vdW dV:', dV_Rs)
        dV_Rs += self._calculate_solvent_derivative(Rs, verbose)
        if verbose:
            print('vdW and solvent dV:', dV_Rs)
        dV_Nss, dV_Oss = self._calculate_hydrogen_bond_derivative(Nss, Oss, verbose)
        if verbose:
            print('Nss, Oss:', Nss, Oss)
        dV_all = _np.concatenate((dV_Rs, dV_Nss, dV_Oss))
        if verbose:
            print('dV_all:', dV_all)

        r_k_all = _np.concatenate((Rs, Nss, Oss))

        n_i_all = Cas[2:-1] - Cas[1:-2]
        n_i_all = n_i_all / self._model._distAA  # _np.repeat([_np.linalg.norm(n_i_all, axis=1)], 3, axis=0).T
        if verbose:
            print('n_i_all:', n_i_all)

        for i in range(len(alphas)):
            # print(_np.repeat([n_i_all[i]], r_k_all.shape[0], axis=0).T.shape, (r_k_all - Cas[i + 1]).shape)
            cross_prod = _np.cross(_np.repeat([n_i_all[i]], r_k_all.shape[0], axis=0), r_k_all - Cas[i + 1])
            dV_dalpha = _np.append(dV_dalpha, _np.sum(_np.diag(dV_all.dot(cross_prod.T))))

        if sec_struct_mode:
            return dV_dalpha[self._non_sec_struct_alpha_inds]
        else:
            return dV_dalpha

    def _K(self, v_init):
        if v_init > 0:
            return v_init / (_sigma_hold ** 2)
        else:
            return 0

    def _derivative_dists_for_coords(self, Rs, Nss, Oss):
        result = _np.empty(0)
        for i in range(0, len(Rs) - 1):
            for j in range(i + 1, len(Rs)):
                result = _np.append(result, _np.linalg.norm(Rs[i] - Rs[j]))

        for i in range(0, len(Nss) - 4):
            for j in range(i + 3, len(Nss)):
                result = _np.append(result, _np.linalg.norm(Nss[i] - Nss[j]))
                result = _np.append(result, _np.linalg.norm(Oss[i] - Oss[j]))
                result = _np.append(result, _np.linalg.norm(Nss[i] - Oss[j]))
                result = _np.append(result, _np.linalg.norm(Oss[i] - Nss[j]))

        return result

    def RMS(self, vector):
        return _np.sqrt(_np.mean(vector ** 2))

    def holding_potentials_energy(self, alphas):
        coords = self._model.dihedralsToCoords(alphas)
        init_coords = self._model.dihedralsToCoords(self._init_alphas)

        result = self.calculateEnergyPotential(alphas) + self._K(self._V_init) * self.RMS(
            self._derivative_dists_for_coords(coords[1],
                                              coords[2],
                                              coords[3]) -
            self._derivative_dists_for_coords(init_coords[1],
                                              init_coords[2],
                                              init_coords[3])
        )
        #print(result)

        return result

    def fixed_sec_structs_energy(self, non_sec_struct_alphas):
        alphas = _np.array(self._init_alphas)
        alphas[self._non_sec_struct_alpha_inds] = non_sec_struct_alphas

        coords = self._model.dihedralsToCoords(alphas)
        init_coords = self._model.dihedralsToCoords(self._init_alphas)

        result = self.calculateEnergyPotential(alphas) + self._K(self._V_init) * self.RMS(
            self._derivative_dists_for_coords(coords[1],
                                              coords[2],
                                              coords[3]) -
            self._derivative_dists_for_coords(init_coords[1],
                                              init_coords[2],
                                              init_coords[3])
        )

        return result

    def minimize_potential_by_holding_potentials(self, alphas_init, minimizationMethod, verbose=False):
        self._init_alphas = alphas_init
        self._V_init = self.calculateEnergyPotential(alphas_init)

        min_res = _minimize(self.holding_potentials_energy,
                            method=minimizationMethod,
                            x0=alphas_init,
                            jac=self.calculate_potential_alpha_derivative,
                            bounds=[[-_np.pi, _np.pi]] * len(alphas_init))
        if verbose:
            print(min_res)
        return min_res

    def minimize_potential_w_sec_structs(self, alphas_init, non_sec_struct_alpha_inds, minimizationMethod, verbose=False):
        self._init_alphas = alphas_init
        self._non_sec_struct_alpha_inds = non_sec_struct_alpha_inds

        self._V_init = self.calculateEnergyPotential(alphas_init)
        min_res = _minimize(self.fixed_sec_structs_energy,
                            method=minimizationMethod,
                            x0=_np.array(alphas_init)[non_sec_struct_alpha_inds],
                            jac=self.calculate_potential_alpha_derivative,
                            bounds=[[-_np.pi, _np.pi]] * len(non_sec_struct_alpha_inds))


        if verbose:
            print(min_res)
        return min_res



