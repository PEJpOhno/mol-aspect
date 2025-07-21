# mol-aspect
This module provides functions and a class to extract heavy atom coordinates from .xyz, mol and .sdf files and to calculate the aspect ratio of molecules using PCA.  

## Current version and requirements
current version = 0.1  

requirements  
pyhon 3.10, 3.11, 3.12, 3.13  
numpy >= 1.22.4

## Getting Start  

Create an instance of a class.  
```python
from mol_aspect import MolAspectRatio
my_aspect = MolAspectRatio()
```
Calculate the aspect ratio of the molecule provided in the given .xyz file.  
For .mol and .sdf files as well, include the file extension when specifying the file name.  

```python
my_aspect.get_aspect_ratio('PATH_TO_YOUR_xyz_FILE.xyz')
```  

To extract the three-dimensional coordinates of a molecule, use the function extract_heavy_atoms_from_xyz() for .xyz files and extract_heavy_atoms_from_mol() for .mol files.  

```python
my_aspect.extract_heavy_atoms_from_xyz('PATH_TO_YOUR_xyz_FILE.xyz')
```

In an SDF file containing multiple molecules, the aspect ratio is returned as a nested list of values corresponding to each molecule, but the three-dimensional coordinates of individual molecules cannot be extracted.  

## Copyright and license
Copyright (c) 2025 Mitsuru Ohno
Released under the BSD-3 license, license that can be found in the LICENSE file.