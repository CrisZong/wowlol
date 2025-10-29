import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "geometry",
    "set_output_file",
    "set_options",
    "optimize",
    "energy",
]

_OUTPUT_FILE = None
_OPTIONS: Dict[str, object] = {}


@dataclass
class GeometryView:
    _coords: List[List[float]]

    @property
    def np(self) -> List[List[float]]:
        return self._coords


class Molecule:
    def __init__(self, symbols: List[str], coords: Sequence[Sequence[float]]):
        self.symbols = list(symbols)
        self._coords = [list(row) for row in coords]

    def geometry(self) -> GeometryView:
        return GeometryView(self._coords)

    def set_geometry(self, coords: Sequence[Sequence[float]]) -> None:
        self._coords = [list(row) for row in coords]

    def copy(self) -> "Molecule":
        return Molecule(self.symbols, self._coords)


class FakeWavefunction:
    def __init__(self, molecule: Molecule, method: str, energy: float):
        self.molecule = molecule
        self.method = method
        self.energy = energy


# --- core utilities -----------------------------------------------------

_PERIODIC_TABLE = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
}


def _parse_xyz(data: str) -> Tuple[List[str], List[List[float]]]:
    raw_lines = data.splitlines()
    if not raw_lines:
        raise ValueError("Empty XYZ data")
    try:
        nat = int(raw_lines[0].strip())
    except ValueError as exc:
        raise ValueError("First line of XYZ must contain the number of atoms") from exc
    if len(raw_lines) < 2 + nat:
        raise ValueError("XYZ data does not contain enough atom lines")
    atom_lines = raw_lines[2:2 + nat]
    symbols: List[str] = []
    coords: List[List[float]] = []
    for line in atom_lines:
        parts = line.strip().split()
        if len(parts) != 4:
            raise ValueError("XYZ atom line must contain 4 fields")
        sym = parts[0]
        if sym not in _PERIODIC_TABLE:
            raise ValueError(f"Unsupported element: {sym}")
        xyz = [float(x) for x in parts[1:4]]
        symbols.append(sym)
        coords.append(xyz)
    return symbols, coords


def geometry(xyz_data: str) -> Molecule:
    symbols, coords = _parse_xyz(xyz_data)
    return Molecule(symbols, coords)


def set_output_file(path: str) -> None:
    global _OUTPUT_FILE
    _OUTPUT_FILE = path


def set_options(options: Dict[str, object]) -> None:
    _OPTIONS.update(options)


# --- helper math --------------------------------------------------------

def _vector_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vec))


def _zeros_like(coords: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[0.0, 0.0, 0.0] for _ in coords]


def _scale(vec: Sequence[float], factor: float) -> List[float]:
    return [factor * component for component in vec]


def _add_inplace(target: List[float], increment: Sequence[float]) -> None:
    for i, value in enumerate(increment):
        target[i] += value


def _subtract_inplace(target: List[float], decrement: Sequence[float]) -> None:
    for i, value in enumerate(decrement):
        target[i] -= value


# --- Optimization -------------------------------------------------------

_BOND_PARAMS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("C", "C"): (1.40, 40.0),
    ("C", "H"): (1.09, 35.0),
    ("C", "O"): (1.43, 45.0),
    ("O", "H"): (0.96, 55.0),
    ("C", "N"): (1.47, 45.0),
}


def _bond_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def _guess_bonds(symbols: List[str], coords: Sequence[Sequence[float]]):
    bonds: List[Tuple[int, int, float, float]] = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            diff = _vector_sub(coords[i], coords[j])
            dist = _norm(diff)
            if dist < 0.1:
                continue
            key = _bond_key(symbols[i], symbols[j])
            params = _BOND_PARAMS.get(key)
            if params is None:
                if dist <= 1.8:
                    params = (dist, 30.0)
                else:
                    continue
            target, k = params
            if dist <= 2.1:
                bonds.append((i, j, target, k))
    return bonds


def _bond_energy(symbols: List[str], coords: Sequence[Sequence[float]], bonds) -> float:
    energy = 0.0
    for i, j, target, k in bonds:
        dist = _norm(_vector_sub(coords[i], coords[j]))
        delta = dist - target
        energy += 0.5 * k * delta * delta
    return energy


def _optimize_geometry(symbols: List[str], coords: Sequence[Sequence[float]], maxiter: int = 400, step: float = 0.005):
    bonds = _guess_bonds(symbols, coords)
    coords = [list(row) for row in coords]
    for _ in range(maxiter):
        grad = _zeros_like(coords)
        max_force = 0.0
        for i, j, target, k in bonds:
            diff = _vector_sub(coords[i], coords[j])
            dist = _norm(diff)
            if dist < 1e-8:
                continue
            delta = dist - target
            scale = (k * delta) / dist
            force_vec = _scale(diff, scale)
            _add_inplace(grad[i], force_vec)
            _subtract_inplace(grad[j], force_vec)
        if grad:
            max_force = max(_norm(g) for g in grad)
        for idx in range(len(coords)):
            for axis in range(3):
                coords[idx][axis] -= step * grad[idx][axis]
        if max_force < 1e-6:
            break
    final_energy = _bond_energy(symbols, coords, bonds)
    return coords, final_energy


def optimize(method: str, molecule: Molecule) -> float:
    new_coords, energy = _optimize_geometry(molecule.symbols, molecule.geometry().np)
    molecule.set_geometry(new_coords)
    return energy


# --- Single point energy ------------------------------------------------

def _single_point_energy(symbols: List[str], coords: Sequence[Sequence[float]]) -> float:
    bonds = _guess_bonds(symbols, coords)
    return _bond_energy(symbols, coords, bonds)


def energy(method: str, *, return_wfn: bool = False, molecule: Molecule):
    e = _single_point_energy(molecule.symbols, molecule.geometry().np)
    if return_wfn:
        return e, FakeWavefunction(molecule.copy(), method, e)
    return e
