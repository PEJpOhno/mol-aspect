# Copyright (c) 2026 Mitsuru Ohno

# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 2026/08/22 M. Ohno
# MolAspectRatio version 0.2

from pathlib import Path

import numpy as np

from .read_mol import read_mol_file


def _zero_aspect():
    return {
        "var_ratio": [0, 0, 0],
        "length_ratio": [0, 0, 0],
        "variance": [0, 0, 0],
        "length": [0, 0, 0],
        "centroid": [0, 0, 0],
        "axes": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    }


ZERO_ASPECT = _zero_aspect()


def extract_heavy_atoms(mol):
    """Extract 3D coordinates of non-hydrogen atoms from a mol.

    Hydrogen isotopes (H, D, T; atomic number 1) are excluded.

    Args:
        mol: An RDKit Mol, or None.

    Returns:
        list: Nested list ``[["element_symbol", x, y, z], ...]``.
        Returns ``[["error", 0, 0, 0]]`` if ``mol`` is None, has no
        conformer, has no heavy atoms, or another error occurs.
    """
    if mol is None:
        return [["error", 0, 0, 0]]
    try:
        if mol.GetNumConformers() < 1:
            return [["error", 0, 0, 0]]
        conf = mol.GetConformer()
        got_coords = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            pos = conf.GetAtomPosition(atom.GetIdx())
            got_coords.append([
                atom.GetSymbol(),
                float(pos.x),
                float(pos.y),
                float(pos.z),
            ])
        if not got_coords:
            return [["error", 0, 0, 0]]
        return got_coords
    except Exception:
        return [["error", 0, 0, 0]]


def mol_aspect_ratio(got_coords):
    """Perform PCA on a heavy-atom coordinate list.

    The centroid is the unweighted mean of heavy-atom coordinates in
    the original frame. The molecule is not moved. ``L`` is the
    max-min projection onto each principal axis.

    Args:
        got_coords (list): Coordinate rows
            ``[["element_symbol", x, y, z], ...]``.

    Returns:
        dict: On success::

            {
                "var_ratio": [1.0, PC2/PC1, PC3/PC1],
                "length_ratio": [1.0, L2/L1, L3/L1],
                "variance": [PC1, PC2, PC3],
                "length": [L1, L2, L3],
                "centroid": [centroid_x, centroid_y, centroid_z],
                "axes": [PC1 vector, PC2 vector, PC3 vector],
            }

        Returns the same keys with zeros if ``got_coords`` contains
        ``["error", 0, 0, 0]`` or PCA fails. Does not include ``name``.
    """
    if not got_coords or ["error", 0, 0, 0] in got_coords:
        return _zero_aspect()
    try:
        coords = []
        for atom in got_coords:
            if len(atom) != 4:
                continue
            try:
                x, y, z = float(atom[1]), float(atom[2]), float(atom[3])
                coords.append([x, y, z])
            except Exception:
                continue
        if len(coords) < 2:
            return _zero_aspect()
        coords = np.array(coords)
        centroid = np.mean(coords, axis=0)
        coords_centered = coords - centroid
        cov = np.cov(coords_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        if eigvals[0] == 0:
            return _zero_aspect()
        pc1_var = float(eigvals[0])
        pc2_var = float(eigvals[1])
        pc3_var = float(eigvals[2])
        Ls = []
        for i in range(3):
            axis = eigvecs[:, i]
            proj = coords_centered.dot(axis)
            Ls.append(float(proj.max() - proj.min()))
        L1, L2, L3 = Ls
        if L1 == 0:
            return _zero_aspect()
        return {
            "var_ratio": [1.0, pc2_var / pc1_var, pc3_var / pc1_var],
            "length_ratio": [1.0, L2 / L1, L3 / L1],
            "variance": [pc1_var, pc2_var, pc3_var],
            "length": [L1, L2, L3],
            "centroid": [
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
            ],
            "axes": [
                [float(x) for x in eigvecs[:, 0]],
                [float(x) for x in eigvecs[:, 1]],
                [float(x) for x in eigvecs[:, 2]],
            ],
        }
    except Exception:
        return _zero_aspect()


def get_aspect_ratio(file_path):
    """Read a file, extract heavy atoms, and run PCA per molecule.

    Each result is ``{"name": filename, **mol_aspect_ratio(...)}``.
    Coordinate files use the file geometry as-is. ``.csv`` is
    always embedded via ``read_smiles`` (UFF off by default).

    Args:
        file_path (pathlib.Path or str): Path to a molecule file
            supported by ``read_mol_file``.

    Returns:
        list: One dict per structure. A ``ValueError`` from reading
        (unsupported extension or missing CSV ``SMILES`` column)
        returns one zero-filled dict with ``name``. Other exceptions
        return ``[]``.
    """
    file_path = Path(file_path)
    file_name = file_path.name
    try:
        results = []
        for name, mol in read_mol_file(file_path):
            got_coords = extract_heavy_atoms(mol)
            results.append({"name": name, **mol_aspect_ratio(got_coords)})
        return results
    except ValueError:
        return [{"name": file_name, **_zero_aspect()}]
    except Exception:
        return []
