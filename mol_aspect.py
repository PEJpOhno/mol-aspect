# Copyright (c) 2025 Mitsuru Ohno

# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 2025/07/20 M. Ohno
# MolAspectRatio

import os
import numpy as np

class MolAspectRatio:
    """
    This module provides functions and a class to extract heavy atom coordinates from
    .xyz, mol and .sdf files and to calculate the aspect ratio of molecules using PCA.

    Classes:
        MolAspectRatio: Class for calculating molecular aspect ratios

    Functions:
        extract_heavy_atoms_from_xyz(file_path): 
            Extracts 3D coordinates of non-hydrogen atoms from .xyz files and returns a nested list.
        extract_heavy_atoms_from_mol(file_path): 
            Extracts 3D coordinates of non-hydrogen atoms from .mol files and returns a nested list.
        mol_aspect_ratio(result): 
            Performs PCA analysis on the obtained 3D coordinate list and returns aspect ratio and molecular long axis variance.
    """

    def __init__(self):
        pass

    def extract_heavy_atoms_from_xyz(self, file_path):
        """
        Extracts 3D coordinates of non-hydrogen atoms from .xyz files and returns
        a nested list in the format [["element_symbol", x, y, z], ...].
        Returns [["error", 0, 0, 0]] on error.

        Args:
            file_path (str): Path to the input .xyz file.

        Returns:
            list: List in the format [["element_symbol", x, y, z], ...]. 
                  Returns [["error", 0, 0, 0]] on error.

        Raises:
            Exceptions are handled internally and [["error", 0, 0, 0]] is returned,
            so no exceptions are raised externally.
        """
        result = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) < 3:
                    return [["error", 0, 0, 0]]
                try:
                    atom_count = int(lines[0].strip())
                except Exception:
                    return [["error", 0, 0, 0]]
                for i in range(2, 2 + atom_count):
                    if i >= len(lines):
                        break
                    parts = lines[i].strip().split()
                    if len(parts) < 4:
                        continue
                    symbol = parts[0]
                    if symbol.upper() == "H":
                        continue
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        result.append([symbol, x, y, z])
                    except Exception:
                        continue
            if not result:
                return [["error", 0, 0, 0]]
            return result
        except Exception:
            return [["error", 0, 0, 0]]

    def extract_heavy_atoms_from_mol(self, file_path):
        """
        Extracts 3D coordinates of non-hydrogen atoms from .mol files and returns
        a nested list in the format [["element_symbol", x, y, z], ...].
        Returns [["error", 0, 0, 0]] on error.

        Args:
            file_path (str): Path to the input .mol file.

        Returns:
            list: List in the format [["element_symbol", x, y, z], ...]. 
                  Returns [["error", 0, 0, 0]] on error.

        Raises:
            Exceptions are handled internally and [["error", 0, 0, 0]] is returned,
            so no exceptions are raised externally.
        """
        result = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) < 4:
                    return [["error", 0, 0, 0]]
                counts_line = lines[3]
                try:
                    atom_count = int(counts_line[:3])
                except Exception:
                    return [["error", 0, 0, 0]]
                for i in range(4, 4 + atom_count):
                    if i >= len(lines):
                        break
                    parts = lines[i].strip().split()
                    if len(parts) < 4:
                        continue
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        symbol = parts[3]
                        if symbol.upper() == "H":
                            continue
                        result.append([symbol, x, y, z])
                    except Exception:
                        continue
            if not result:
                return [["error", 0, 0, 0]]
            return result
        except Exception:
            return [["error", 0, 0, 0]]

    def mol_aspect_ratio(self, result):
        """
        Performs PCA analysis on the obtained 3D coordinate list result and returns
        [PC2/PC1, PC3/PC1, PC1 variance]. Returns [0, 0, 0, 0] if ["error", 0, 0, 0] is included.

        Args:
            result (list): 3D coordinate list in the format [["element_symbol", x, y, z], ...].
                          May contain ["error", 0, 0, 0] on error.

        Returns:
            list: List of [PC2/PC1, PC3/PC1, PC1 variance].
                  Returns [0, 0, 0, 0] if input is invalid or error occurs.

        Raises:
            Exceptions are handled internally, so none are raised externally.
        """
        if not result or ["error", 0, 0, 0] in result:
            return [0, 0, 0, 0]
        try:
            coords = []
            for atom in result:
                if len(atom) != 4:
                    continue
                try:
                    x, y, z = float(atom[1]), float(atom[2]), float(atom[3])
                    coords.append([x, y, z])
                except Exception:
                    continue
            if len(coords) < 2:
                return [0, 0, 0, 0]
            coords = np.array(coords)
            coords_centered = coords - np.mean(coords, axis=0)
            cov = np.cov(coords_centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            if eigvals[0] == 0:
                return [0, 0, 0, 0]
            pc1_var = eigvals[0]
            pc2_var = eigvals[1]
            pc3_var = eigvals[2]
            aspect_ratios = [pc2_var/pc1_var, pc3_var/pc1_var, pc1_var]
            return [float(e) for e in aspect_ratios]
        except Exception:
            return [0, 0, 0, 0]

    def get_aspect_ratio(self, file_path):
        """
        Calls the appropriate extract function based on the file_path extension,
        performs PCA with mol_aspect_ratio, and returns [pc2_var/pc1_var, pc3_var/pc1_var, pc1_var].
        For SDF files, returns a list of [property_name, pc2_var/pc1_var, pc3_var/pc1_var, pc1_var] for all molecules.

        Args:
            file_path (str): Path to the input file. Extensions .xyz, .mol, .sdf are expected.

        Returns:
            list: 
                - For .xyz, .mol: List of [filename, pc2_var/pc1_var, pc3_var/pc1_var, pc1_var].
                - For .sdf: Nested list of [[property_name, pc2_var/pc1_var, pc3_var/pc1_var, pc1_var], ...].
                - Returns [filename, 0, 0, 0, 0] or [] if input is invalid or unsupported extension.

        Raises:
            Exceptions are handled internally, so none are raised externally.
        """
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        if ext == ".xyz":
            result = self.extract_heavy_atoms_from_xyz(file_path)
            aspect = self.mol_aspect_ratio(result)
            return [[file_name] + aspect]
        elif ext == ".mol":
            result = self.extract_heavy_atoms_from_mol(file_path)
            aspect = self.mol_aspect_ratio(result)
            return [[file_name] + aspect]
        elif ext == ".sdf":
            results = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                mol_blocks = []
                current_block = []
                for line in lines:
                    current_block.append(line)
                    if line.strip() == "$$$$":
                        mol_blocks.append(current_block)
                        current_block = []
                for idx, mol_block in enumerate(mol_blocks):
                    # プロパティ名取得（> <...> の最初のもの or インデックス）
                    prop_name = None
                    for i, l in enumerate(mol_block):
                        if l.startswith("> <"):
                            prop_name_lines = [l.strip()]
                            # 次の> <で始まる行までの間の文字列を改行を除いて取得
                            for l2 in mol_block[i+1:]:
                                if l2.startswith("> <"):
                                    break
                                prop_name_lines.append(l2.strip())
                            prop_name = "".join(prop_name_lines)[2:] # 最初の"> "を除去
                            break
                    if prop_name is None:
                        prop_name = f"{file_name}_mol{idx+1}"
                    # mol部分（M  ENDまで）を抽出
                    mol_lines = []
                    found_m_end = False
                    for l in mol_block:
                        mol_lines.append(l)
                        if l.strip() == "M  END":
                            found_m_end = True
                            break
                    if not found_m_end:
                        results.append([prop_name, 0, 0, 0, 0])
                        continue
                    # extract_heavy_atoms_from_molのロジックを直接利用
                    # mol_linesをパース
                    result = []
                    try:
                        # molファイルのヘッダーは3行、その後に原子数・結合数の行
                        if len(mol_lines) < 4:
                            result = [["error", 0, 0, 0]]
                        else:
                            counts_line = mol_lines[3]
                            try:
                                atom_count = int(counts_line[:3])
                            except Exception:
                                result = [["error", 0, 0, 0]]
                                atom_count = 0
                            atom_start = 4
                            atom_end = atom_start + atom_count
                            for i in range(atom_start, atom_end):
                                if i >= len(mol_lines):
                                    break
                                parts = mol_lines[i].rstrip('\n').split()
                                if len(parts) < 4:
                                    continue
                                symbol = parts[3] if len(parts) > 3 else ""
                                if symbol.upper() == "H":
                                    continue
                                try:
                                    x = float(parts[0])
                                    y = float(parts[1])
                                    z = float(parts[2])
                                    result.append([symbol, x, y, z])
                                except Exception:
                                    continue
                            if not result:
                                result = [["error", 0, 0, 0]]
                    except Exception:
                        result = [["error", 0, 0, 0]]
                    aspect = self.mol_aspect_ratio(result)
                    results.append([prop_name] + aspect)
                return results
            except Exception:
                return []
        else:
            return [[file_name, 0, 0, 0, 0]]
