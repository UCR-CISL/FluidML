from itertools import permutations
from typing import List, Iterator, Tuple


def permute_shape(shape: Tuple[int]) -> Iterator[Tuple[int]]:
    fixed_indices: List[int] = [idx for idx, dim in enumerate(shape) if dim == 1]
    return filter(
        lambda elem: all(map(lambda idx: idx == elem[idx], fixed_indices)),
        permutations(range(len(shape))),
    )
