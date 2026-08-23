# Copyright (c) 2026 Mitsuru Ohno

# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 2026/08/23 M. Ohno
# MolAspectRatio version 0.2

from pathlib import Path

from rdkit import Chem
import py3Dmol

from .calc_aspect import extract_heavy_atoms, mol_aspect_ratio
from .read_mol import read_mol_file

AXIS_COLORS = ("blue", "green", "magenta")
PCA_FAILED = [0, 0, 0, 0]


def _has_explicit_hydrogen(mol):
    return any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())


def _resolve_mol(path, name, mol):
    if mol is not None and path is not None:
        return "Specify either a file path or mol, not both."
    if mol is not None:
        return mol
    if path is None:
        return "Specify a file path or mol."

    file_path = Path(path)
    found = None
    try:
        for rec_name, rec_mol in read_mol_file(file_path):
            if name is None:
                found = rec_mol
                break
            if rec_name == name:
                found = rec_mol
                break
    except ValueError as exc:
        return str(exc)

    if found is None and name is not None:
        return (
            f"Filename and molecule name do not match: "
            f"'{file_path.name}' vs '{name}'."
        )
    if found is None:
        return "Specify a file path or mol."
    return found


def _prepare_display_mol(mol, centroid):
    disp = Chem.Mol(mol)
    if not _has_explicit_hydrogen(disp):
        disp = Chem.AddHs(disp, addCoords=True)
    conf = disp.GetConformer()
    cx, cy, cz = centroid
    for i in range(disp.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x - cx, pos.y - cy, pos.z - cz))
    return disp


def _draw_viewer(disp_mol, eigvecs, lengths, width, height):
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(Chem.MolToMolBlock(disp_mol), "mol")
    viewer.setStyle({"stick": {}})
    for vec, length, color in zip(eigvecs, lengths, AXIS_COLORS):
        if length == 0:
            continue
        half = length / 2.0
        start = [-half * vec[0], -half * vec[1], -half * vec[2]]
        end = [half * vec[0], half * vec[1], half * vec[2]]
        viewer.addCylinder({
            "start": {"x": start[0], "y": start[1], "z": start[2]},
            "end": {"x": end[0], "y": end[1], "z": end[2]},
            "radius": 0.08,
            "color": color,
            "fromCap": True,
            "toCap": True,
        })
    viewer.zoomTo()
    return viewer


def view_aspect3d(path=None, name=None, mol=None, width=400, height=400):
    """Show one molecule in py3Dmol with PC1-PC3 axes.

    Specify a file ``path`` (and ``name`` for multi-molecule files) or
    ``mol``. Single-molecule files may omit ``name``. Axes are ±L/2
    from the heavy-atom centroid (the display origin). Colors are
    PC1=blue, PC2=green, PC3=magenta.

    Args:
        path (pathlib.Path or str): Molecule file path. Mutually
            exclusive with ``mol``.
        name (str): Identifier from ``read_mol_file``
            (e.g. ``foo.sdf_1``). Optional for single-molecule files.
        mol: An RDKit Mol already in memory. Mutually exclusive with
            ``path``.
        width (int): Viewer width in pixels. Defaults to 400.
        height (int): Viewer height in pixels. Defaults to 400.

    Returns:
        py3Dmol.view or str: A viewer on success, or an English error
        string on failure (missing path/mol, name mismatch, no 3D
        conformer, or PCA failure).
    """
    resolved = _resolve_mol(path, name, mol)
    if isinstance(resolved, str):
        return resolved
    if resolved is None:
        return "PCA failed: mol is None."
    if resolved.GetNumConformers() < 1:
        return "No 3D coordinates: this molecule has no conformer."

    aspect = mol_aspect_ratio(extract_heavy_atoms(resolved))
    if aspect == PCA_FAILED:
        return (
            "PCA failed: need at least two non-hydrogen atoms "
            "with spatial extent."
        )

    lengths = aspect[3]
    centroid = aspect[4]
    eigvecs = aspect[5]
    disp = _prepare_display_mol(resolved, centroid)
    return _draw_viewer(disp, eigvecs, lengths, width, height)


def write_aspect3d_html(viewer, filename, dir=None):
    """Write a py3Dmol viewer to an HTML file.

    If ``viewer`` is an error string from ``view_aspect3d``, that
    string is returned and no file is written.

    Args:
        viewer: A py3Dmol viewer, or an error string.
        filename (str): Output file name only (not a directory).
        dir (pathlib.Path or str): Output directory. Defaults to the
            current working directory (often the notebook folder).

    Returns:
        str: Path of the written file, or the error string if
        ``viewer`` is a string.
    """
    if isinstance(viewer, str):
        return viewer

    out_dir = Path.cwd() if dir is None else Path(dir)
    out_path = out_dir / filename
    if hasattr(viewer, "write_html"):
        viewer.write_html(str(out_path))
    else:
        out_path.write_text(viewer._make_html(), encoding="utf-8")
    return str(out_path)
