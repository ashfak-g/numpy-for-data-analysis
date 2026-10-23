# Day 2: Indexing, Slicing, Masking & Memory Management (`02_indexing_and_memory`)

This folder contains the Day 2 practice notebook and documentation for **NumPy Indexing, Slicing, Boolean Masking, Fancy Indexing, Conditional Operations, and Memory Behavior (Views vs. Copies)**. All concepts are practiced on both synthetic arrays and the UCI Red Wine Quality dataset (`winequality-red.csv`, 1,599 rows × 12 features).

---

## Folder Contents

| File | Description |
| :--- | :--- |
| [02_indexing_and_memory.ipynb](./02_indexing_and_memory.ipynb) | Complete Day 2 practice Jupyter Notebook with code and outputs |
| `README.md` | Detailed walkthrough of all concepts, outputs, and memory rules |

> **Dataset path note:** The notebook uses `"/content/winequality-red.csv"` from Google Colab. When running locally inside this folder, point to `"../01_array_fundamentals/winequality-red.csv"`.

---

## Code Walkthrough

### 1. Dataset Setup & Feature Reference

The Red Wine Quality dataset (`(1599, 12)`, `float64`) is loaded via `np.loadtxt(..., delimiter=";", skiprows=1)`, and all 12 column names are stored in a reference list (`0`: `fixed acidity`, ..., `10`: `alcohol`, `11`: `quality`) for clear column indexing throughout the notebook.

---

### 2. 1D, 2D, and 3D Indexing & Slicing

#### 1D Array (`arr_1d` = `[10, 20, ..., 100]`)
- **Positive & Negative Indexing:** `arr_1d[0]` (`10`), `arr_1d[9]` (`100`), `arr_1d[-1]` (`100`), `arr_1d[-3]` (`80`).
- **Basic Slicing (`arr[start:stop:step]`):** `start` is inclusive and `stop` is exclusive (`arr_1d[2:7]` → `[30 40 50 60 70]`).
- **Step Slicing & Reversing:**
  - `arr_1d[::2]` → `[10 30 50 70 90]` (every 2nd element)
  - `arr_1d[1::2]` → `[20 40 60 80 100]` (every 2nd element starting at index 1)
  - `arr_1d[::-1]` → reverses the entire array without a loop.

#### 2D Array (`4×4` Matrix & Wine Dataset `data`)
- **Row access:** `arr_2d[0]`, `arr_2d[-1]`, `data[:5]` (first 5 wine records).
- **Element access (`arr[row, col]`):** `arr_2d[1, 2]` (`70`), `arr_2d[-1, -1]` (`160`).
- **Column extraction (`arr[:, col]`):** `data[:, 0]` (`fixed acidity`), `data[:, 10]` (`alcohol`), `data[:, 11]` (`quality`).
- **Submatrix slicing:** `arr_2d[1:3, 1:4]` and `data[:10, :3]` (first 3 features of the first 10 wines).

#### 3D Array (`np.arange(1, 25).reshape(2, 3, 4)`)
- Represents **2 matrices**, each with **3 rows and 4 columns** (`shape: (2, 3, 4)`, `ndim: 3`).
- Indexed as `arr_3d[depth, row, col]`:
  - `arr_3d[0, 1, 2]` → `7` (1st matrix, 2nd row, 3rd column)
  - `arr_3d[:, 0, :]` → first row from both matrices
  - `arr_3d[0:2, 0:2, 1:3]` → extracts a `(2, 2, 2)` 3D sub-block across both matrices.

---

### 3. Views vs. Copies (Memory Sharing)

A critical NumPy rule demonstrated in both synthetic arrays and `data`:

| Operation | Creates | `.base` | `np.shares_memory()` | Mutating it affects original? |
| :--- | :---: | :--- | :---: | :---: |
| Basic slicing (`original[1:4]`, `data[:10, :]`) | **View** | Points to original array | `True` | **Yes** |
| Explicit copy (`original[1:4].copy()`, `data[:10, :].copy()`) | **Copy** | `None` | `False` | **No** |
| Fancy indexing (`data[[0, 1, 2]]`) | **Copy** | `None` | `False` | **No** |

**Wine dataset verification test:**
```python
original_value = data[0, 0]
wine_view[0, 0] = -999   # Mutates data[0, 0] -> -999.0
wine_copy[0, 0] = -500   # Leaves data[0, 0] untouched at -999.0
data[0, 0] = original_value  # Restores original value safely
```

---

### 4. Boolean Masking (Conditional Filtering)

Filtering arrays using element-wise boolean conditions and bitwise operators (`&` for AND, `|` for OR, `~` for NOT—always wrapping each condition in parentheses):

**Wine Dataset Filtering Results (`1,599` total records):**

| Condition | Expression | Matching Rows |
| :--- | :--- | :---: |
| High alcohol | `data[:, 10] > 10` | **852** |
| Good quality | `data[:, 11] >= 7` | **217** |
| High alcohol **AND** good quality | `(data[:, 10] > 10) & (data[:, 11] >= 7)` | **196** |
| Quality is `5` **OR** `8` | `(data[:, 11] == 5) \| (data[:, 11] == 8)` | **699** |
| Quality is **NOT** below `5` | `~(data[:, 11] < 5)` | **1,536** |

---

### 5. Fancy Indexing (Integer Array Indexing)

Selecting arbitrary, non-contiguous positions using lists/arrays of integer indices:

- **1D selection:** `arr[[0, 2, 4, 6]]` → `[10 30 50 70]`
- **2D rows / columns:** `arr_2d[[0, 2, 4]]` (specific rows), `arr_2d[:, [0, 2]]` (specific columns)
- **Paired coordinates (diagonal):** `arr_2d[[0, 1, 2], [0, 1, 2]]` → `[10 50 90]` (pairs `(0,0)`, `(1,1)`, `(2,2)`)
- **Cartesian subgrid (`np.ix_`):** `arr_2d[np.ix_([0, 2, 4], [0, 2])]` extracts the full 3×2 intersection of rows `[0, 2, 4]` and columns `[0, 2]`.
- **Wine dataset subsetting:**
  - Specific samples: `data[[0, 10, 50, 100, 500]]`
  - Selected feature subset (`fixed acidity`, `volatile acidity`, `alcohol`, `quality`): `data[:, [0, 1, 10, 11]]` → shape `(1599, 4)`.
- **Memory check:** `np.shares_memory(data, data[[0, 1, 2]])` returns `False`—fancy indexing always creates an independent **Copy**.

---

### 6. Conditional Selection, Multi-Class Labeling, Clipping & Coordinate Search

| Function | Purpose | Notebook Example & Result |
| :--- | :--- | :--- |
| **`np.where(cond, x, y)`** | Vectorized `if-else` assignment (or index lookup when only `cond` is passed) | Classified `quality >= 7` into `"Good"` / `"Average"`, `alcohol > 12` into `"Premium"` / `"Regular"`, and retrieved indices where `salary > 30000` (`[2, 3, 5, 7]`). |
| **`np.select(conds, choices, default)`** | Clean multi-condition branching (replaces nested `np.where`) | Grouped Wine quality into 3 tiers: **Low (`<= 4`)**: `63`, **Medium (`5–6`)**: `1,319`, **High (`>= 7`)**: `217` (verified with `np.unique(..., return_counts=True)`). |
| **`np.clip(arr, min, max)`** | Caps outliers below `min` or above `max` to boundary values | Clamped `salary` to `[20000, 80000]` (`10000 → 20000`, `120000 → 80000`) and constrained Wine `alcohol` to `[8, 14]`. |
| **`np.argwhere(cond)`** | Returns a 2D array of exact index coordinates where `cond` is `True` | Located 2D elements `> 50` (`[[1 0], [1 2], [2 1], [2 2]]`) and found all **18** top-rated wines with `quality >= 8` (first index: `267`). |

---

## Key Takeaways

1. Basic slicing returns a **View** (zero-copy, fast, shares memory), whereas `.copy()`, Boolean masking, and Fancy indexing return independent **Copies**. Always check with `.base` or `np.shares_memory()` when mutating slices.
2. In Boolean masking, always use `&`, `|`, `~` with parentheses around each condition instead of Python's `and`, `or`, `not`.
3. Use `np.ix_(rows, cols)` when you need a rectangular submatrix from non-contiguous row and column indices—plain `arr[rows, cols]` pairs indices element-by-element instead.
4. Combine `np.where()`, `np.select()`, `np.clip()`, and `np.argwhere()` to handle conditional feature engineering, outlier capping, and index lookup without writing Python loops.
