# 動作の流れ

読み込みのあと、同じ mol を **PCA**（`get_aspect_ratio`）または **表示**（`view_aspect3d`）に渡す。表示も内部で同じ PCA を行う。

```text
read_mol_file / read_smiles / mol=
        │
        ├─► get_aspect_ratio 経路
        │     extract_heavy_atoms → mol_aspect_ratio
        │     dict（name + 比・分散・長さ・重心・軸）
        │
        └─► view_aspect3d 経路
              extract_heavy_atoms → mol_aspect_ratio   ※ 同じ PCA
              _prepare_display_mol → _draw_viewer
              py3Dmol（任意で write_aspect3d_html）
```

---

## 読み込みの処理流れ

座標ファイルは最適化済み 3D を入力することを前提としている。SMILES とそれを集めた CSV だけ、接続情報から 3D 構造を生成する（`AllChem.EmbedMolecule`）。

```text
get_aspect_ratio(file_path)
view_aspect3d(path=...)
        │
        ▼
read_mol_file(path, seed, optimize)   ※ seed / optimize は .csv のみ
        │
        ├─ .xyz / .mol / .pdb / .sdf / .mol2
        │     ファイルを読む（3D 構造は生成しない、UFF しない）
        │     removeHs=False
        │
        └─ .csv
              └─► read_smiles(smi, seed, optimize)
                    AddHs → EmbedMolecule（3D 構造生成）→ UFF（optimize=True のとき）
                    失敗: mol is None
```

### 呼び出し側

- `get_aspect_ratio` … 返ってきた mol で PCA（`read_mol_file` のデフォルト `optimize=False`）
- `view_aspect3d(path=...)` … 同じ `read_mol_file`。明示 H が無ければ表示用にだけ `AddHs(addCoords=True)`
- `view_aspect3d(mol=...)` … `read_mol_file` に入らない。渡された mol をそのまま使う

### 入力ファイル形式の違いによる処理差

1. 拡張子: 座標ファイルか CSV か
2. 座標ファイル: ファイルの 3D をそのまま使う
3. SMILES / CSV: 常に 3D 構造を生成する。構造最適化（UFF） は `optimize=True` のときだけ

---

## PCA の処理流れ

重原子（原子番号 ≠ 1）の座標だけを使う。幾何学的な形状を知りたいので、重心は単純平均。各軸の `L` は、その主軸への射影の max−min。

```text
mol
        │
        ▼
extract_heavy_atoms(mol)
        │
        ├─ mol is None / コンフォーマー無し / 重原子無し
        │     → [["error", 0, 0, 0]]
        │
        └─ 各原子
              原子番号 1 は除く
              [元素記号, x, y, z]
        │
        ▼
mol_aspect_ratio(got_coords)
        │
        ├─ error 行、または点が 2 個未満、または PC1/L1 が 0
        │     → _zero_aspect()（全キー 0）
        │
        └─ 成功
              重心 = 重原子座標の平均
              共分散 → 固有値分解（分散が大きい順が PC1–PC3）
              各軸に射影して L1, L2, L3
              var_ratio = [1, PC2/PC1, PC3/PC1]
              length_ratio = [1, L2/L1, L3/L1]
```

`get_aspect_ratio` は分子ごとに `{"name": name, **mol_aspect_ratio(...)}` を並べる。読込の `ValueError` は `name` 付きゼロ埋め 1 件。その他の例外は `[]`。

---

## 表示の処理流れ

PCA は計算側と同じ関数。表示用コピーだけ重心を原点に平行移動する。軸は ±L/2。色は PC1=青、PC2=緑、PC3=マゼンタ。`L=0` の軸は描かない。表示の際にも、PCAを再計算するので、PCA処理の結果を表示するために、オプションをそろえること。

```text
view_aspect3d(path, name, mol, width, height)
        │
        ▼
_resolve_mol(path, name, mol)
        │
        ├─ path と mol の両方 / どちらも無し / 名前不一致
        │     → 英語のエラー文字列
        ├─ mol=  → その mol
        └─ path= → read_mol_file（上の読み込み流れ）
        │
        ▼
mol is None / コンフォーマー無し / PCA がゼロ
        → 英語のエラー文字列
        │
        ▼
extract_heavy_atoms → mol_aspect_ratio   ※ PCA と同じ
        │
        ▼
_prepare_display_mol(mol, centroid)
        コピー
        明示 H が無ければ Chem.AddHs(..., addCoords=True)  ※ 表示のみ
        全原子を −centroid だけ平行移動
        │
        ▼
_draw_viewer(...)
        py3Dmol stick
        円柱で三軸（長さ 0 はスキップ）
        │
        ▼
（任意）write_aspect3d_html(viewer, filename, dir)
```

---

## 関数名つきの呼び出し関係

```text
get_aspect_ratio(file_path)
  └─► read_mol_file(file_path)
        └─► extract_heavy_atoms(mol)
              └─► mol_aspect_ratio(got_coords)
                    ├─ _zero_aspect()          ※ 失敗
                    └─ numpy の平均・共分散・eigh


view_aspect3d(path, name, mol, width, height)
  └─► _resolve_mol(path, name, mol)
        │
        ├─ mol= のとき  → その mol を返す（read_mol_file は呼ばない）
        └─ path= のとき
              └─► read_mol_file(file_path)
  └─► extract_heavy_atoms(resolved)
        └─► mol_aspect_ratio(got_coords)
  └─► _prepare_display_mol(resolved, centroid)
        └─► _has_explicit_hydrogen(disp)
        └─► Chem.AddHs(disp, addCoords=True)   ※ 明示 H が無いときだけ
  └─► _draw_viewer(disp, eigvecs, lengths, width, height)
        └─► py3Dmol.view / addModel / addCylinder
  └─► （任意）write_aspect3d_html(viewer, filename, dir)
```

```text
read_mol_file(file_path, seed, optimize)
  │
  ├─ .csv
  │     └─► read_smiles(smi, seed, optimize)
  │           ├─ Chem.MolFromSmiles
  │           ├─ Chem.AddHs
  │           ├─ AllChem.EmbedMolecule          ※ 3D 構造を生成（常に）
  │           └─ AllChem.UFFOptimizeMolecule    ※ optimize=True のとき
  │
  └─ .xyz / .mol / .pdb / .sdf / .mol2
        └─ Chem.MolFromXYZFile / MolFromMolFile / MolFromPDBFile
            ForwardSDMolSupplier / MolFromMol2Block
            （3D 構造は生成しない）
```
