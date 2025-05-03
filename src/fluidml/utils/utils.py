import numpy as np

from itertools import permutations
from typing import Dict, List, Iterator, Tuple


__DTYPE_MAP: Dict[str, np.dtype] = {
    "f32": np.float32,
    "f64": np.float64,
    "i1": np.bool_,
    "i32": np.int32,
    "i64": np.int64,
    "u32": np.uint32,
    "u64": np.uint64,
}


def map_str_dtype(dtype: str) -> np.dtype:
    return __DTYPE_MAP[dtype]


def permute_shape(shape: Tuple[int]) -> Iterator[Tuple[int]]:
    fixed_indices: List[int] = [idx for idx, dim in enumerate(shape) if dim == 1]
    return filter(
        lambda elem: all(map(lambda idx: idx == elem[idx], fixed_indices)),
        permutations(range(len(shape))),
    )
