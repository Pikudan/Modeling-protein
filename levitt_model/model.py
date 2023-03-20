import numpy as _np
import pandas as _pd
from .geometry import getPlaneByThreePoints as _getPlaneByThreePoints,\
    getAxisRotationMatrix as _getAxisRotationMatrix,\
    projectPointOnLine as _projectPointOnLine


class LevittModel:
    _tab1PD = None

    _distAA = 3.808
    # _distNsOs = 1
    # _tauAAA = 106.3 / 180 * _np.pi

    _residues = None

    def __init__(self, tab1csvPath, resNames):
        self._tab1PD = _pd.read_csv(tab1csvPath)
        self._residues = _np.array(resNames)

    def _tauAAA(self, alpha):
        # result = 112 + 25 * _np.cos(0.9*(alpha % (2*_np.pi)) + 2*_np.pi/3)
        # return result / 180 * _np.pi
        a, b, c, d = [ 1.95861985, -0.4580599 ,  1.03085375,  0.1917959 ]
        return a + b*_np.sin(c*((alpha % (2*_np.pi)) + d))

    def dihedralsToCoords(self, alphas, dist_ns_os=1):
        CaCoords = _np.empty((0, 3))
        Rcoords = _np.empty((0, 3))
        NsCoords = _np.empty((0, 3))
        OsCoords = _np.empty((0, 3))

        CaCoords = _np.append(CaCoords, [[1, 0, 0]], axis=0)
        CaCoords = _np.append(CaCoords, [[1 + self._distAA, 0, 0]], axis=0)
        CaCoords = _np.append(CaCoords, [[1 + self._distAA - self._distAA*_np.cos(self._tauAAA(alphas[0])), self._distAA*_np.sin(self._tauAAA(alphas[0])), 0]], axis=0)

        for i in range(len(alphas)):
            # next Ca
            A1 = CaCoords[-2] + _getAxisRotationMatrix(_np.array(CaCoords[-1]) - _np.array(CaCoords[-2]), alphas[i]).dot(CaCoords[-3] - CaCoords[-2])
            #plane = _getPlaneByThreePoints(CaCoords[-1], CaCoords[-2], A1)
            v0 = (CaCoords[-1] - CaCoords[-2]) / _np.linalg.norm(CaCoords[-1] - CaCoords[-2])
            E = _projectPointOnLine(A1, (CaCoords[-1], CaCoords[-1] - CaCoords[-2]))
            v1 = (A1 - E) / _np.linalg.norm(A1 - E)
            tauAAA = self._tauAAA(alphas[i+1]) if i < len(alphas) - 1 else self._tauAAA(-_np.pi/4)
            CaCoords = _np.append(CaCoords, [CaCoords[-1] - v0 * self._distAA*_np.cos(tauAAA) + v1*self._distAA*_np.sin(tauAAA)], axis=0)

        for i in range(1, len(CaCoords) - 1):
            # i'th Ca's R
            if self._residues[i] == 'GLY':
                Rcoords = _np.append(Rcoords, [CaCoords[i]], axis=0)
                continue

            params = self._tab1PD[self._tab1PD.amino_acid == self._residues[i]]
            A1 = CaCoords[i-1] + _getAxisRotationMatrix(_np.array(CaCoords[i]) - _np.array(CaCoords[i-1]), params.phiAAAR.values[0]).dot(CaCoords[i+1] - CaCoords[i-1])
            # plane = _getPlaneByThreePoints(CaCoords[i-1], CaCoords[i], A1)
            v0 = (CaCoords[-2] - CaCoords[-3]) / _np.linalg.norm(CaCoords[-2] - CaCoords[-3])
            E = _projectPointOnLine(A1, (CaCoords[-2], CaCoords[-2] - CaCoords[-3]))
            v1 = (A1 - E) / _np.linalg.norm(A1 - E)
            Rcoords = _np.append(Rcoords, [CaCoords[i] - v0*params.dAR.values[0]*_np.cos(params.tauAAR.values[0]) + v1*params.dAR.values[0]*_np.sin(params.tauAAR.values[0])], axis=0)

        for i in range(1, len(CaCoords) - 1):
            # Ns and Os
            NsCoords = _np.append(NsCoords, [(CaCoords[i] + CaCoords[i-1]) / 2], axis=0)
            plane = _getPlaneByThreePoints(CaCoords[i], CaCoords[i+1], CaCoords[i-1])
            OsCoords = _np.append(OsCoords, [NsCoords[-1] + dist_ns_os * plane[:3]/_np.linalg.norm(plane[:3])], axis=0)

        NsCoords = _np.append(NsCoords, [(CaCoords[i+1] + CaCoords[i]) / 2], axis=0)
        OsCoords = _np.append(OsCoords, [NsCoords[-1] - dist_ns_os * plane[:3]/_np.linalg.norm(plane[:3])], axis=0)

        return CaCoords, Rcoords, NsCoords, OsCoords

    '''
    def plotSimplifiedProtein(this, Cas, Rs, Nss, Oss, ax=None, interactive=False):
        if interactive:
            %matplotlib notebook
        else:
            %matplotlib inline
        
        if type(ax) == type(None):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(1,1,1, projection='3d')
    
        ax.plot(Cas[:, 0], Cas[:, 1], Cas[:, 2], c='black')
        ax.scatter(*Cas.T, c='black', s=30)
        ax.scatter(*Rs.T, c='green', s=60)
        ax.scatter(*Nss.T, c='blue', s=30)
        ax.scatter(*Oss.T, c='red', s=30)
        
        for i in range(len(Rs)):
            ax.plot(*_np.vstack((Cas[i + 1], Rs[i])).T, c='green', linewidth=1)

        for i in range(len(Nss)):
            ax.plot(*_np.vstack((Nss[i], Oss[i])).T, c='blue', linewidth=1)
    '''


