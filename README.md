# mol-aspect

Calculate molecular aspect ratios from heavy-atom coordinates by PCA, and optionally view the principal axes in 3D.

The import name is `molaspect`. The pip name is `mol-aspect`.

**Documentation:** https://pejpohno.github.io/mol-aspect/  
**How to cite:** Ohno, M. mol-aspect. GitHub. https://github.com/PEJpOhno/mol-aspect (2025).  

## Current version and requirements

current version = 0.2

requirements
- python >= 3.12
- numpy >= 2.0.2
- rdkit >= 2024.3.1
- py3Dmol >= 2.4.0

## Getting started

```
pip install git+https://github.com/PEJpOhno/mol-aspect.git
```
Then  
```python
import molaspect

molaspect.get_aspect_ratio("PATH_TO_YOUR_FILE.mol")
```

Coordinate files (`.xyz`, `.mol`, `.pdb`, `.sdf`, `.mol2`) are used as given 3D structures. Include the file extension in the path. `.sdf` and `.mol2` may contain multiple molecules.

SMILES can be read with `read_smiles`, or as a `.csv` table whose header has a column named `SMILES`. Those paths always generate 3D coordinates (`AllChem.EmbedMolecule`); UFF is off unless `optimize=True`.

For a worked example, see `examples/example_script.ipynb`. To display PC1–PC3 axes, use `view_aspect3d`.

## Acknowledgement  
This module and its accompanying documentation were developed with the support of Cursor’s AI-assisted tools.  

## Copyright and license

Copyright (c) 2025 Mitsuru Ohno
Released under the BSD-3 license, license that can be found in the LICENSE file.
