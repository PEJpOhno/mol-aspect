# Copyright (c) 2026 Mitsuru Ohno

# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 2026/08/22 M. Ohno
# MolAspectRatio version 0.2

import csv

from rdkit import Chem
from rdkit.Chem import AllChem


def read_mol_file(file_path, seed=123, optimize=False):
    """Yield ``[filename, mol]`` one structure at a time.

    Single-molecule formats (``.xyz``, ``.mol``, ``.pdb``) yield one
    entry. ``.sdf`` and ``.mol2`` yield one entry per structure, with a
    1-based index appended to the filename (e.g. ``compounds.sdf_1``).

    Coordinate files (``.xyz``, ``.mol``, ``.pdb``, ``.sdf``,
    ``.mol2``) are treated as optimized 3D structures. File
    coordinates are used as-is; they are not embedded or
    force-field optimized. Explicit hydrogens are kept
    (``removeHs=False``) so display can use file H coordinates when
    present.

    ``.csv`` is a SMILES table. The first row is a header, and the
    column named ``SMILES`` (uppercase, exact match) is read. Each
    non-empty cell is passed to ``read_smiles``. Identifiers use the
    1-based file line number (header is line 1, so the first data row
    is ``filename_2``). Other columns are ignored. ``.csv`` always
    embeds; UFF runs only when ``optimize`` is True.

    Args:
        file_path (pathlib.Path): Path to the input file.
        seed (int): Random seed for SMILES embedding. Used only for
            ``.csv``. Defaults to 123.
        optimize (bool): If True, run UFF optimization after embedding
            SMILES. Used only for ``.csv``. Defaults to False.

    Yields:
        list: ``[name, mol]`` where ``mol`` is an RDKit Mol or None.

    Raises:
        ValueError: If the extension is unsupported, or a ``.csv``
            header has no ``SMILES`` column.
    """
    filename = file_path.name
    ext = file_path.suffix.lower()
    path = str(file_path)

    if ext == ".xyz":
        yield [filename, Chem.MolFromXYZFile(path)]
    elif ext == ".mol":
        yield [filename, Chem.MolFromMolFile(path, removeHs=False)]
    elif ext == ".pdb":
        yield [filename, Chem.MolFromPDBFile(path, removeHs=False)]
    elif ext == ".sdf":
        with open(path, "rb") as inf:
            for i, mol in enumerate(
                Chem.ForwardSDMolSupplier(inf, removeHs=False), start=1
            ):
                yield [f"{filename}_{i}", mol]
    elif ext == ".mol2":
        text = file_path.read_text(encoding="utf-8")
        chunks = text.split("@<TRIPOS>MOLECULE")
        for i, chunk in enumerate(chunks[1:], start=1):
            block = "@<TRIPOS>MOLECULE" + chunk
            yield [f"{filename}_{i}", Chem.MolFromMol2Block(block, removeHs=False)]
    elif ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as inf:
            reader = csv.DictReader(inf)
            if reader.fieldnames is None or "SMILES" not in reader.fieldnames:
                raise ValueError("CSV header must contain a SMILES column")
            for i, row in enumerate(reader, start=2):
                smi = (row.get("SMILES") or "").strip()
                if not smi:
                    continue
                for _, mol in read_smiles(smi, seed=seed, optimize=optimize):
                    yield [f"{filename}_{i}", mol]
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def read_smiles(smiles, seed=123, optimize=False):
    """Build a 3D mol from a single SMILES string.

    Always adds hydrogens and embeds with ``randomSeed=seed``.
    UFF optimization runs only when ``optimize`` is True. For
    multiple SMILES, use a ``.csv`` file with ``read_mol_file``.

    Args:
        smiles (str): A single SMILES string.
        seed (int): Random seed for embedding. Defaults to 123.
        optimize (bool): If True, run UFF optimization after
            embedding. Defaults to False.

    Yields:
        list: ``[smiles, mol]``. ``mol`` is None if parsing or
        embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        yield [smiles, None]
        return

    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if status != 0:
        yield [smiles, None]
        return

    if optimize:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

    yield [smiles, mol]
