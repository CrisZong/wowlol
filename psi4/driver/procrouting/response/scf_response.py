from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .... import Molecule

__all__ = ["tdscf_excitations"]


@dataclass
class Transition:
    state: int
    energy: float  # Hartree
    energy_ev: float
    wavelength_nm: float
    occupied: int
    virtual: int
    oscillator_strength_len: float

    def as_dict(self) -> Dict[str, float]:
        strength = self.oscillator_strength_len
        return {
            "state": self.state,
            "energy": self.energy,
            "energy_ev": self.energy_ev,
            "wavelength_nm": self.wavelength_nm,
            "occupied_index": self.occupied,
            "virtual_index": self.virtual,
            "oscillator_strength_len": strength,
            "OSCILLATOR STRENGTH (LEN)": strength,
        }


_ALPHA = 0.0
_BETA_EV = -2.9  # simple Hückel beta parameter
_HARTREE_TO_EV = 27.211386245988
_EV_TO_HARTREE = 1.0 / _HARTREE_TO_EV


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vec))


def _hartree_to_nm(energy_hartree: float) -> float:
    if energy_hartree <= 0:
        return math.inf
    return 45.5633525316 / energy_hartree


def _build_adjacency(mol: Molecule) -> List[List[float]]:
    symbols = mol.symbols
    coords = mol.geometry().np
    carbon_indices = [i for i, s in enumerate(symbols) if s == "C"]
    n = len(carbon_indices)
    adj = [[0.0 for _ in range(n)] for _ in range(n)]
    for idx_i, atom_i in enumerate(carbon_indices):
        for idx_j in range(idx_i + 1, n):
            atom_j = carbon_indices[idx_j]
            diff = [coords[atom_i][axis] - coords[atom_j][axis] for axis in range(3)]
            dist = _norm(diff)
            if dist <= 1.7:
                adj[idx_i][idx_j] = adj[idx_j][idx_i] = 1.0
    return adj


def _jacobi_eigenvalues(matrix: List[List[float]], tol: float = 1e-10, max_iter: int = 1000) -> List[float]:
    n = len(matrix)
    if n == 0:
        return []
    a = [row[:] for row in matrix]

    def max_offdiag(a: List[List[float]]) -> Tuple[int, int, float]:
        max_val = 0.0
        p = 0
        q = 1 if n > 1 else 0
        for i in range(n):
            for j in range(i + 1, n):
                val = abs(a[i][j])
                if val > max_val:
                    max_val = val
                    p, q = i, j
        return p, q, max_val

    for _ in range(max_iter):
        p, q, max_val = max_offdiag(a)
        if max_val < tol:
            break
        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c = math.cos(phi)
        s = math.sin(phi)
        for k in range(n):
            apk = a[p][k]
            aqk = a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        for k in range(n):
            akp = a[k][p]
            akq = a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0
    return [a[i][i] for i in range(n)]


def _huckel_energies(mol: Molecule) -> List[float]:
    adj = _build_adjacency(mol)
    eigvals = _jacobi_eigenvalues(adj)
    eigvals.sort(reverse=True)
    return [_ALPHA + _BETA_EV * eig for eig in eigvals]


def _occupation(mol: Molecule) -> int:
    return sum(1 for s in mol.symbols if s == "C")


def _estimate_oscillator_strength(wavelength_nm: float, occ: int, virt: int) -> float:
    if not math.isfinite(wavelength_nm):
        return 0.0
    base = math.exp(-((wavelength_nm - 200.0) / 70.0) ** 2)
    spacing = max(1, virt - occ)
    scale = 0.35 + 0.1 * spacing
    strength = base * scale
    return max(0.0, min(1.2, strength))


def tdscf_excitations(wfn, states: int = 10) -> List[Dict[str, float]]:
    if not hasattr(wfn, "molecule"):
        raise TypeError("Wavefunction must carry a molecule")
    mol: Molecule = wfn.molecule
    energies_ev = _huckel_energies(mol)
    n_levels = len(energies_ev)
    if n_levels == 0:
        return []
    electrons = _occupation(mol)
    occ_levels = electrons // 2
    occupied = list(range(occ_levels))
    virtuals = list(range(occ_levels, n_levels))
    transitions: List[Transition] = []
    state_index = 1
    for occ in occupied:
        for virt in virtuals:
            energy_diff_ev = energies_ev[virt] - energies_ev[occ]
            if energy_diff_ev <= 0:
                continue
            energy_diff_hartree = energy_diff_ev * _EV_TO_HARTREE
            wavelength_nm = _hartree_to_nm(energy_diff_hartree)
            transition = Transition(
                state=state_index,
                energy=energy_diff_hartree,
                energy_ev=energy_diff_ev,
                wavelength_nm=wavelength_nm,
                occupied=occ,
                virtual=virt,
                oscillator_strength_len=_estimate_oscillator_strength(
                    wavelength_nm, occ, virt
                ),
            )
            transitions.append(transition)
            state_index += 1
    transitions.sort(key=lambda t: t.energy)
    if states is not None:
        transitions = transitions[:states]
    return [t.as_dict() for t in transitions]
