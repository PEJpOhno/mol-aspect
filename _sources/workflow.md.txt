# 動作の流れ

読み込みのあと、同じ mol を **PCA**（`get_aspect_ratio`）または **表示**（`view_aspect3d`）に渡す。可視化に際して、再度アスペクト比の計算をおこなっている。`get_aspect_ratio` はファイルを受け取る。`view_aspect3d` はファイルパス、または `mol=`。SMILESを文字列としてそのまま与える場合には、`read_smiles` で mol を作り、可視化は `view_aspect3d(mol=mol)` で行う。

```text
read_mol_file / read_smiles / mol=
        │
        ├─► get_aspect_ratio 経路   ※ ファイルのみ（read_mol_file）
        │     extract_heavy_atoms → mol_aspect_ratio
        │     dict（name + 比・分散・長さ・重心・軸）
        │
        └─► view_aspect3d 経路     ※ path= または mol=
              extract_heavy_atoms → mol_aspect_ratio   ※ 同じ PCA
              _prepare_display_mol → _draw_viewer
              py3Dmol（任意で write_aspect3d_html）
```

---

## 読み込みの処理流れ

入力する構造は、最適化した3D構造であることを前提としている。与える3D構造（座標）は構造最適化などをおこない、妥当な空間配置としておくこと。SMILESと、それを集めた.csvだけ、接続情報から3D構造を発生させる（`AllChem.EmbedMolecule`）。

```text
get_aspect_ratio(file_path, seed, optimize)
view_aspect3d(path=..., seed, optimize)
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
                    AddHs → EmbedMolecule（3D 構造生成）
                    → UFFOptimize（デフォルト。optimize=False で停止）
                    失敗: mol is None
```

`get_aspect_ratio` が返す dict の `name` は、次のとおり。

- `.xyz` / `.mol` / `.pdb`: ファイル名
- `.sdf` / `.mol2` / `.csv`: ファイル名＋通し番号（1始まり）。例: `compounds.sdf_1`、先頭の SMILES 行は `filename_1`
- 空の SMILES は読み飛ばすが、通し番号は進む
- SMILESを文字列としてそのまま与える場合には、`get_aspect_ratio` が付ける `ファイル名_通し番号` の `name` はない。戻り値の第一要素は与えた SMILES 文字列である

### 呼び出し側

- `get_aspect_ratio` … 分子ごとに dict を返す。デフォルトは `seed=123`、`optimize=True`。`optimize=False` で構造最適化を停止することができる
- `view_aspect3d()` … 指定した一分子ずつ可視化する思想で設計している。`path=` のときは同じ `read_mol_file`。明示 H が無ければ表示用にだけ `AddHs(addCoords=True)`
- sdf, mol2のように、複数分子が１ファイルに含まれる場合の可視化は、ファイルパスと、ファイル名＋通し番号からなるIDを指定する。IDは、`get_aspect_ratio` が返すdictの `name` で確認できる。.csvの場合も同様。`name` を省略すると先頭の一分子だけ
- `view_aspect3d(mol=...)` … `read_mol_file` に入らない。渡された mol をそのまま使う。`seed` / `optimize` は使わない

### 入力ファイル形式の違いによる処理差

1. 拡張子: 座標ファイルか CSV か
2. 座標ファイル（.xyz, .mol, .pdb および .sdf, .mol2）: ファイルの 3D をそのまま使う。sdfとmol2は複数分子を含んでよい。`seed` / `optimize` は使わない
3. SMILESで構造を与えた場合、3D構造を発生させ、デフォルトではその構造をUFFOptimizeする。ただし、発生する3D構造は、seedで指定した乱数に従って発生したものであって（デフォルトは `seed=123`）、初期構造として適切なものである保証はない。csv形式で与える場合も同様。`optimize=False` で構造最適化を停止することができる。SMILESを.csvファイルに記述する場合、一行目をヘッダにし、列名は `"SMILES"`（大文字であること）とする

---

## PCA の処理流れ

重原子（原子番号 ≠ 1）の座標だけを使う。座標の原点は、重原子の幾何学的な重心。幾何的な特徴を捉えるために、原子量による重みづけはおこなっていない。各軸の `L` は、その主軸への射影の max−min。

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

`get_aspect_ratio` は分子ごとに `{"name": name, **mol_aspect_ratio(...)}` を並べる。キーは name, var_ratio（分散比 1.0, PC2/PC1, PC3/PC1）, length_ratio（長さ比 1.0, L2/L1, L3/L1）, variance（三軸の分散）, length（三軸の長さ）, centroid（重心座標）, axes（三軸の固有ベクトル）。読込の `ValueError` は `name` 付きゼロ埋め 1 件。その他の例外は `[]`。

---

## 表示の処理流れ

`view_aspect3d()` 関数を用いる。指定した一分子ずつ可視化する思想で設計している。座標の原点は、重原子の幾何学的な重心。幾何的な特徴を捉えるために、原子量による重みづけはおこなっていない。±L/2。PC1=青, PC2=緑, PC3=マゼンタ。`L=0` の軸は描かない。

可視化に際して、再度アスペクト比の計算をおこなっている。特に、SMILESや、.csvでget_aspect_ratioで求めた結果を目視して、その構造を可視化したい場合には、引数（seed, optimize）を統一すること。SMILESを文字列として与えて可視化する場合、先に 3D mol を作り、`mol=` で渡す。

```text
view_aspect3d(path, name, mol, width, height, seed, optimize)
        │
        ▼
_resolve_mol(path, name, mol, seed, optimize)
        │
        ├─ path と mol の両方 / どちらも無し / 名前不一致
        │     → 英語のエラー文字列
        ├─ mol=  → その mol（seed / optimize は使わない）
        └─ path= → read_mol_file(..., seed, optimize)
              name 省略 → 先頭の一分子
              name 指定 → ファイル名＋通し番号からなるIDと一致する分子
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
get_aspect_ratio(file_path, seed, optimize)
  └─► read_mol_file(file_path, seed, optimize)
        └─► extract_heavy_atoms(mol)
              └─► mol_aspect_ratio(got_coords)
                    ├─ _zero_aspect()          ※ 失敗
                    └─ numpy の平均・共分散・eigh


view_aspect3d(path, name, mol, width, height, seed, optimize)
  └─► _resolve_mol(path, name, mol, seed, optimize)
        │
        ├─ mol= のとき  → その mol を返す（read_mol_file は呼ばない）
        └─ path= のとき
              └─► read_mol_file(file_path, seed, optimize)
                    name 省略 → 先頭の一分子で break
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
  │           ├─ AllChem.EmbedMolecule          ※ 3D 構造を発生（常に）
  │           └─ AllChem.UFFOptimizeMolecule    ※ デフォルトで実行。optimize=False で停止
  │
  └─ .xyz / .mol / .pdb / .sdf / .mol2
        └─ Chem.MolFromXYZFile / MolFromMolFile / MolFromPDBFile
            ForwardSDMolSupplier / MolFromMol2Block
            （3D 構造は生成しない。seed / optimize は使わない）
```
