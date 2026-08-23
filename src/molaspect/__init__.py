# Copyright (c) 2025 Mitsuru Ohno
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# 07/22/2025, M. Ohno
# mol_aspect __init__

from ._version import __version__

from .read_mol import read_mol_file, read_smiles
from .calc_aspect import extract_heavy_atoms, mol_aspect_ratio, get_aspect_ratio
from .view_aspect3d import view_aspect3d, write_aspect3d_html
