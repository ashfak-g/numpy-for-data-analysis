# Day 1: NumPy Array Fundamentals (`01_array_fundamentals`)

This folder contains the Day 1 practice notebook, dataset, and notes for **NumPy Fundamentals**. Using the UCI Red Wine Quality dataset (`winequality-red.csv`) alongside synthetic examples, it covers loading numeric CSV data, creating arrays from scratch, inspecting array attributes, optimizing memory with downcasting, and comparing Python Lists against NumPy Arrays.

---

## Folder Contents

| File | Description |
| :--- | :--- |
| [01_array_fundamentals.ipynb](./01_array_fundamentals.ipynb) | Complete Day 1 practice Jupyter Notebook with code and outputs |
| [winequality-red.csv](./winequality-red.csv) | UCI Red Wine Quality dataset (1,599 rows, 12 numeric features, semicolon-separated) |
| `README.md` | Detailed code walkthrough and Day 1 Final Assignment |

---

## Code Walkthrough

### 1. Creating a NumPy Array from a Dataset

```python
import numpy as np

data = np.loadtxt("/content/winequality-red.csv", delimiter=";", skiprows=1)
print(data)
```

**How it works:**
- `import numpy as np`: Imports NumPy using the standard alias `np`.
- `np.loadtxt(...)`: Loads numeric data from a text/CSV file directly into a 2D NumPy array (`float64` by default).
- `delimiter=";"`: Unlike standard comma-separated files, `winequality-red.csv` separates columns with semicolons (`;`).
- `skiprows=1`: Skips the header row containing column names (`fixed acidity`, `volatile acidity`, ..., `quality`) because `np.loadtxt()` expects numeric values.

> **Local execution note:** The notebook was originally run in Google Colab using `"/content/winequality-red.csv"`. When running locally inside this folder, use `"winequality-red.csv"`.

---

### 2. Array Creation Without a Dataset

Seven core NumPy array creation functions are practiced in the notebook:

| Code | Output (Summary) | Explanation |
| :--- | :--- | :--- |
| `np.array([10, 20, 30, 40])` | `array([10, 20, 30, 40])` | Converts a standard Python list into a 1D NumPy array. |
| `np.zeros((3, 4))` | 3×4 grid of `0.` | Creates a 3-row by 4-column 2D array filled with zeros (`float64` by default). |
| `np.ones((3, 3))` | 3×3 grid of `1.` | Creates a 3×3 2D array filled with ones (`float64` by default). |
| `np.full((2, 3), 99)` | 2×3 grid of `99` | Creates a 2×3 2D array filled with a custom constant value (`99`). |
| `np.arange(0, 101, 5)` | `[0, 5, 10, ..., 100]` | Generates 21 evenly stepped integers from `0` up to (but excluding) `101` with a step size of `5`. |
| `np.linspace(0, 100, 11)` | `[0., 10., 20., ..., 100.]` | Generates exactly `11` evenly spaced floating-point numbers between `0` and `100` (both endpoints inclusive). |
| `np.eye(4)` | 4×4 Identity Matrix | Creates a 4×4 square matrix with `1.` along the main diagonal and `0.` elsewhere. |

---

### 3. Inspecting Array Attributes (`data`)

Five essential attributes are inspected on the loaded `winequality-red.csv` array:

```python
data.ndim    # 2
data.shape   # (1599, 12)
data.size    # 19188
data.dtype   # dtype('float64')
data.nbytes  # 153504
```

**Explanation:**
- **`data.ndim` → `2`**: The array is 2-dimensional (rows and columns).
- **`data.shape` → `(1599, 12)`**: Contains **1,599 rows** (wine samples) and **12 columns** (11 physicochemical features + 1 `quality` score).
- **`data.size` → `19188`**: Total number of individual elements (`1599 × 12 = 19,188`).
- **`data.dtype` → `float64`**: Each value is stored as a 64-bit (8-byte) floating-point number.
- **`data.nbytes` → `153504`**: Total RAM consumed by the array's data buffer (`19,188 × 8 bytes = 153,504 bytes` ≈ 150 KB).

---

### 4. Data Type & Memory Optimization

To reduce memory usage on large datasets, numeric downcasting with `astype()` is demonstrated:

#### (a) Downcasting `float64` to `float32`
```python
data_float32 = data.astype(np.float32)

print(data.dtype)          # float64
print(data_float32.dtype)  # float32

print(data.nbytes)         # 153504
print(data_float32.nbytes) # 76752
```
- `data.astype(np.float32)` returns a **new array** without modifying the original `data` array (`data.dtype` remains `float64`).
- Reducing each element from 8 bytes (`float64`) to 4 bytes (`float32`) cuts memory usage from **153,504 bytes to 76,752 bytes**—an exact **50% memory reduction**.

#### (b) Slicing the `quality` Column and Casting to `int8`
```python
quality = data[:, -1].astype(np.int8)
print(quality.dtype)   # int8
print(quality.nbytes)  # 1599
```
- `data[:, -1]` extracts all rows of the final column (`quality`).
- Because wine quality ratings are small integers (ranging from 3 to 8), they easily fit inside the signed `int8` range (`-128` to `127`).
- Storing 1,599 values in `float64` takes `1,599 × 8 = 12,792` bytes, whereas `int8` (1 byte per value) uses only **1,599 bytes**—an **87.5% memory reduction** for that column.

---

### 5. Python List vs. NumPy Array

Four side-by-side experiments highlight why NumPy arrays outperform Python lists for numerical computing:

```python
my_list = [10, 20, 30, 40]
my_array = np.array([10, 20, 30, 40])
```

| Test | Python List (`my_list`) | NumPy Array (`my_array`) | Key Insight |
| :--- | :--- | :--- | :--- |
| **1. Object Type** (`type()`) | `<class 'list'>` | `<class 'numpy.ndarray'>` | NumPy uses a dedicated C-backed `ndarray` structure. |
| **2. Math Operation** (`* 2`) | `[10, 20, 30, 40, 10, 20, 30, 40]` | `[20 40 60 80]` | Lists repeat elements on `* 2`; NumPy performs **vectorized** element-wise multiplication. |
| **3. Memory Footprint** | `200` bytes (`sys.getsizeof` sum) | `32` bytes (`my_array.nbytes`) | Lists store pointers and full Python `int` objects; NumPy packs raw values contiguously (`4 × 8 = 32` bytes). |
| **4. Mixed Types** (`[10, "hello", 3.14]`) | `[10, 'hello', 3.14]` (mixed types kept) | `['10' 'hello' '3.14']` (all cast to string) | Lists are **heterogeneous**; NumPy arrays are **homogeneous** (all elements share a single `dtype`). |

---

## Day 1 Final Assignment

Self-assessment checklist covering all core concepts from Day 1 (try answering each question before expanding the solution):

- [ ] **1. How do you create a 2D NumPy array?**
  <details>
  <summary>Click to verify answer</summary>

  Pass a nested list (list of lists) into `np.array()`, or pass a 2-element shape tuple `(rows, cols)` to factory functions such as `np.zeros()`, `np.ones()`, `np.full()`, `np.eye()`, or load tabular numeric data with `np.loadtxt()`.
  ```python
  arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
  zeros_2d = np.zeros((3, 4))
  ```
  </details>

- [ ] **2. What is the difference between `zeros()` and `ones()`?**
  <details>
  <summary>Click to verify answer</summary>

  Both create arrays of a given shape (defaulting to `float64`), but `np.zeros()` fills every element with **`0.0`**, while `np.ones()` fills every element with **`1.0`**.
  </details>

- [ ] **3. When should you use `arange()` vs. `linspace()`?**
  <details>
  <summary>Click to verify answer</summary>

  - Use **`np.arange(start, stop, step)`** when you know the exact **step size** between consecutive values (e.g., `np.arange(0, 101, 5)`). The `stop` value is exclusive.
  - Use **`np.linspace(start, stop, num)`** when you know the exact **number of evenly spaced points (`num`)** needed across a range (e.g., `np.linspace(0, 100, 11)`). Both `start` and `stop` are inclusive by default.
  </details>

- [ ] **4. What kind of matrix does `eye(4)` create?**
  <details>
  <summary>Click to verify answer</summary>

  `np.eye(4)` creates a **4×4 Identity Matrix** with `1.` along the main diagonal and `0.` in all off-diagonal positions.
  </details>

- [ ] **5. What is the difference between `shape` and `size`?**
  <details>
  <summary>Click to verify answer</summary>

  - **`shape`** returns a tuple describing the length of each dimension (e.g., `data.shape` → `(1599, 12)` for 1,599 rows and 12 columns).
  - **`size`** returns a single integer representing the total number of elements across the entire array (e.g., `data.size` → `19188`, equal to `1599 × 12`).
  </details>

- [ ] **6. How much memory is saved when converting from `float64` to `float32`?**
  <details>
  <summary>Click to verify answer</summary>

  Memory usage is reduced by exactly **50% (cut in half)**, because `float64` uses 8 bytes (64 bits) per element while `float32` uses 4 bytes (32 bits). In this notebook, memory dropped from `153,504` bytes to `76,752` bytes.
  </details>

- [ ] **7. When can `int8` be used for integer data?**
  <details>
  <summary>Click to verify answer</summary>

  `int8` can be used when all values are small whole numbers strictly within the signed 8-bit range **`-128` to `+127`** (such as ages, ratings, or the wine `quality` scores). Storing values outside this range causes **integer overflow**.
  </details>

- [ ] **8. What are the main differences between a NumPy array and a Python list?**
  <details>
  <summary>Click to verify answer</summary>

  1. **Data types:** Python lists are heterogeneous (can mix `int`, `str`, `float`); NumPy arrays are homogeneous (every element shares one `dtype`).
  2. **Math operations:** NumPy arrays support direct vectorized math (`arr * 2`), whereas Python lists require explicit loops (`list * 2` repeats elements).
  3. **Memory & speed:** NumPy stores values in contiguous memory using C-level operations, making it much more memory-efficient and significantly faster for numerical computing.
  </details>

- [ ] **9. Does `astype()` modify the original array?**
  <details>
  <summary>Click to verify answer</summary>

  **No.** `astype()` does not operate in-place; it always returns a **new copy** of the array cast to the requested `dtype`. In this notebook, `data.dtype` remained `float64` even after running `data_float32 = data.astype(np.float32)`.
  </details>

- [ ] **10. What does `nbytes` measure?**
  <details>
  <summary>Click to verify answer</summary>

  `nbytes` measures the total number of **bytes** consumed in memory by all elements of a NumPy array (`nbytes = size × itemsize`).
  </details>

---

## Key Takeaways

1. `np.loadtxt()` loads structured numeric CSV files directly into 2D arrays with explicit delimiter and header handling.
2. Factory functions (`zeros`, `ones`, `full`, `arange`, `linspace`, `eye`) provide fast, readable ways to initialize arrays and matrices.
3. Always inspect `ndim`, `shape`, `size`, `dtype`, and `nbytes` right after loading data to understand its layout and memory footprint.
4. Choosing appropriate data types (`float32`, `int8`) drastically cuts RAM usage (50% to 87.5% in these exercises) without losing needed precision.
5. Contiguous homogeneous storage and vectorized operations are the core reasons NumPy forms the foundation of the Python data science stack.
