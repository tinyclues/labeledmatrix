# Basic operations available on `LabeledMatrix`
## Available constructors to initialize matrix
### `__init__`
### `from_zip_occurrence`
### `from_dict`
### `from_diagonal`
### `from_pivot`
### `from_ragged_tensor`
### `from_pyarrow_list_array`
### `from_xarray`
### `from_random`
## Available exporters to other formats
### `to_vectorial_dataframe`
### `to_flat_dataframe`
### `to_ragged_tensor`
### `to_pyarrow`
### `to_xarray`
### `to_tensorflow_model`
## Attributes
### `deco`
### `shape`
### `dtype`
### `is_square`
### `is_sparse`
### `nnz`
### `density`
## Modify matrix content or its labels
### `to_dense`
### `to_sparse`
### `to_optimal_format`
### `align`
### `extend`
### `restrict`
### `exclude`
### `symmetrize_label`
### `sort`
### `sort_by_hierarchical_clustering`
### `rename_row`
### `rename_column`
### `sample_rows`, `sample_columns`
### `transpose`
### `zeros`, `without_zeros`
### `diagonal`, `without_diagonal`
### `without_mask`
